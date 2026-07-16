"""专项高危与异常行为统计模块 (A2)。

实现 5 项专项行为检测器：
  1. SpatialTrajectoryMap — 轨迹空间建图（基础组件）
  2. RepetitivePathDetector — 重复路线/无目的徘徊
  3. RepeatedActionDetector — 反复开关/寻找行为
  4. ProlongedInactivityDetector — 长时间静止/异常久坐久卧
  5. CircadianRhythmAnalyzer — 昼夜节律偏移
  6. SocialInteractionAnalyzer — 社交互动强度

全部纯 CPU 几何/统计算法。每个检测器输出必须包含:
  time_window, valid_duration, confidence_score
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. 轨迹空间建图
# ============================================================

class SpatialTrajectoryMap:
    """网格化空间轨迹映射器。

    将连续空间划分为网格，记录人体质心在网格间的移动轨迹。
    用于后续徘徊检测和热点分析。
    """

    def __init__(
        self,
        grid_resolution: int = 200,
        image_width: float = 640.0,
        image_height: float = 480.0,
        max_history_sec: float = 600.0,  # 保留 10 分钟历史
        fps: float = 15.0,
    ) -> None:
        self._grid_resolution = grid_resolution
        self._image_width = image_width
        self._image_height = image_height
        self._max_history = int(max_history_sec * fps)
        self._fps = fps

        # 网格访问计数
        self._cell_visits: Dict[Tuple[int, int], int] = defaultdict(int)
        # 路径序列: [(grid_cell, timestamp, track_id), ...]
        self._path_history: deque = deque(maxlen=self._max_history)
        # 网格间转移计数: {(cell_a, cell_b): count}
        self._transitions: Dict[Tuple, int] = defaultdict(int)

    @property
    def grid_resolution(self) -> int:
        return self._grid_resolution

    def to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """将像素坐标映射到网格单元。"""
        return (int(x / self._grid_resolution), int(y / self._grid_resolution))

    def add_position(
        self,
        centroid_x: float,
        centroid_y: float,
        timestamp: float,
        track_id: int = 0,
    ) -> Optional[Tuple[int, int]]:
        """记录一个位置，更新轨迹。

        Returns:
            当前网格单元，坐标无效时返回 None。
        """
        if centroid_x is None or centroid_y is None:
            return None
        if np.isnan(centroid_x) or np.isnan(centroid_y):
            return None

        cell = self.to_grid(centroid_x, centroid_y)
        self._cell_visits[cell] += 1

        if self._path_history:
            prev_cell, _, _ = self._path_history[-1]
            if prev_cell != cell:
                key = (prev_cell, cell)
                self._transitions[key] += 1

        self._path_history.append((cell, timestamp, track_id))
        return cell

    def get_path_sequence(
        self, start_sec: Optional[float] = None, end_sec: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """获取指定时间窗口内的路径网格序列。"""
        path = [
            (cell, ts) for cell, ts, _ in self._path_history
            if (start_sec is None or ts >= start_sec)
            and (end_sec is None or ts <= end_sec)
        ]
        return path

    def get_top_cells(self, n: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """返回访问次数最多的网格单元。"""
        return sorted(self._cell_visits.items(), key=lambda x: -x[1])[:n]

    def get_transition_count(self, cell_a: Tuple[int, int], cell_b: Tuple[int, int]) -> int:
        return self._transitions.get((cell_a, cell_b), 0)

    def clear(self) -> None:
        self._cell_visits.clear()
        self._path_history.clear()
        self._transitions.clear()

    def __repr__(self) -> str:
        return (
            f"SpatialTrajectoryMap(grid={self._grid_resolution}px, "
            f"cells_visited={len(self._cell_visits)}, "
            f"path_len={len(self._path_history)})"
        )


# ============================================================
# 2. 徘徊检测
# ============================================================

class RepetitivePathDetector:
    """重复路线/无目的徘徊检测器。

    检测原理:
      1. 在固定时间窗内，分析路径网格序列
      2. 查找重复出现的子路径（连续 3+ 网格单元）
      3. 计算路径重合度 = 重复边数 / 总边数
      4. 若重合度超过阈值，标记为徘徊事件

    输出:
      {
        "timestamp": float,
        "time_window": [start_sec, end_sec],
        "valid_duration": float,
        "repetitive_path_count": int,
        "path_overlap_ratio": float,
        "is_wandering": bool,
        "confidence_score": float,
      }
    """

    def __init__(
        self,
        window_sec: float = 600.0,         # 10 分钟检测窗口
        stride_sec: float = 120.0,         # 2 分钟步长
        min_path_length: int = 5,           # 最短路径长度（网格单元数）
        min_repetition_count: int = 3,      # 同一子路径最少重复次数
        overlap_threshold: float = 0.4,     # 路径重合度阈值
        fps: float = 15.0,
        grid_resolution: int = 200,
    ) -> None:
        self._window_sec = window_sec
        self._stride_sec = stride_sec
        self._min_path_length = min_path_length
        self._min_repetition_count = min_repetition_count
        self._overlap_threshold = overlap_threshold
        self._fps = fps

        self._trajectory = SpatialTrajectoryMap(
            grid_resolution=grid_resolution,
            max_history_sec=window_sec * 2,
            fps=fps,
        )
        self._last_check_ts = -stride_sec
        self._events: List[Dict[str, Any]] = []

    def update(
        self,
        centroid_x: Optional[float],
        centroid_y: Optional[float],
        timestamp: float,
        track_id: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """添加新位置，到达检查周期时触发检测。

        Returns:
            检测结果 dict，未到周期时返回 None。
        """
        if centroid_x is not None and centroid_y is not None:
            self._trajectory.add_position(centroid_x, centroid_y, timestamp, track_id)

        if timestamp - self._last_check_ts < self._stride_sec:
            return None

        self._last_check_ts = timestamp
        return self._detect(timestamp)

    def _detect(self, current_ts: float) -> Dict[str, Any]:
        """执行徘徊检测。"""
        window_start = current_ts - self._window_sec
        path = self._trajectory.get_path_sequence(window_start, current_ts)

        n = len(path)
        valid_duration = self._window_sec if n > 0 else 0.0
        path_overlap_ratio = 0.0
        repetitive_count = 0
        is_wandering = False

        if n >= self._min_path_length:
            # 提取网格序列
            cells = [c for c, _ in path]

            # 统计重复边
            edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = defaultdict(int)
            for i in range(len(cells) - 1):
                edge = (cells[i], cells[i + 1])
                edges[edge] += 1

            total_edges = len(cells) - 1
            repeated_edges = sum(1 for c in edges.values() if c >= self._min_repetition_count)

            if total_edges > 0:
                path_overlap_ratio = repeated_edges / total_edges
            is_wandering = path_overlap_ratio >= self._overlap_threshold
            repetitive_count = int(repeated_edges)

        # 置信度
        if n < self._min_path_length:
            confidence = 0.3
        else:
            confidence = min(0.95, 0.5 + path_overlap_ratio * 0.5)

        result = {
            "timestamp": current_ts,
            "time_window": [window_start, current_ts],
            "valid_duration": valid_duration,
            "repetitive_path_count": repetitive_count,
            "path_overlap_ratio": round(path_overlap_ratio, 4),
            "is_wandering": is_wandering,
            "confidence_score": round(confidence, 4),
        }
        self._events.append(result)
        return result

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def get_daily_repetitive_count(self) -> int:
        """日级徘徊事件总数。"""
        return sum(1 for e in self._events if e["is_wandering"])

    def reset(self) -> None:
        self._trajectory.clear()
        self._events.clear()
        self._last_check_ts = -self._stride_sec


# ============================================================
# 3. 重复动作检测
# ============================================================

class RepeatedActionDetector:
    """反复开关/寻找行为检测器。

    检测原理:
      1. 使用空间聚类识别高频访问的「兴趣区域」
      2. 统计在短时间内反复进入/离开同一区域的次数
      3. 若某区域在时间窗内的进出次数超过阈值，标记为重复行为

    输出:
      {
        "timestamp": float,
        "time_window": [start_sec, end_sec],
        "valid_duration": float,
        "hotspot_count": int,
        "hotspot_actions": [{"cell": (x, y), "visits": N, "enter_exit_pairs": N}],
        "is_repetitive": bool,
        "confidence_score": float,
      }
    """

    def __init__(
        self,
        window_sec: float = 600.0,
        stride_sec: float = 120.0,
        hotspot_min_visits: int = 5,         # 热点区域最少访问次数
        hotspot_max_radius_cells: int = 1,    # 热点合并半径（网格单元数）
        enter_exit_threshold: int = 4,        # 进出次数阈值
        fps: float = 15.0,
        grid_resolution: int = 200,
    ) -> None:
        self._window_sec = window_sec
        self._stride_sec = stride_sec
        self._hotspot_min_visits = hotspot_min_visits
        self._hotspot_max_radius_cells = hotspot_max_radius_cells
        self._enter_exit_threshold = enter_exit_threshold
        self._fps = fps

        self._trajectory = SpatialTrajectoryMap(
            grid_resolution=grid_resolution,
            max_history_sec=window_sec * 2,
            fps=fps,
        )
        self._last_check_ts = -stride_sec
        self._events: List[Dict[str, Any]] = []

    def update(
        self,
        centroid_x: Optional[float],
        centroid_y: Optional[float],
        timestamp: float,
        track_id: int = 0,
    ) -> Optional[Dict[str, Any]]:
        if centroid_x is not None and centroid_y is not None:
            self._trajectory.add_position(centroid_x, centroid_y, timestamp, track_id)

        if timestamp - self._last_check_ts < self._stride_sec:
            return None

        self._last_check_ts = timestamp
        return self._detect(timestamp)

    def _detect(self, current_ts: float) -> Dict[str, Any]:
        window_start = current_ts - self._window_sec
        path = self._trajectory.get_path_sequence(window_start, current_ts)

        n = len(path)
        valid_duration = self._window_sec if n > 0 else 0.0

        hotspot_actions: List[Dict] = []
        hotspot_count = 0
        is_repetitive = False

        if n >= self._hotspot_min_visits:
            cells = [c for c, _ in path]

            # 统计每个网格单元的进出次数
            cell_transitions: Dict[Tuple[int, int], int] = defaultdict(int)
            prev_cell = None
            for cell in cells:
                if prev_cell is not None and cell != prev_cell:
                    # 离开 prev_cell → 进入 cell
                    cell_transitions[cell] += 1
                prev_cell = cell

            # 找出热点：访问次数 ≥ threshold
            hotspots = [
                (cell, count)
                for cell, count in cell_transitions.items()
                if count >= self._enter_exit_threshold
            ]
            hotspots.sort(key=lambda x: -x[1])

            # 合并邻近热点
            merged = self._merge_nearby_hotspots(hotspots)

            hotspot_count = len(merged)
            is_repetitive = hotspot_count > 0

            for cell, count in merged[:10]:  # 最多报告 10 个
                hotspot_actions.append({
                    "cell": list(cell),
                    "enter_exit_pairs": count,
                })

        confidence = min(0.95, 0.4 + hotspot_count * 0.15) if hotspot_count > 0 else 0.3

        result = {
            "timestamp": current_ts,
            "time_window": [window_start, current_ts],
            "valid_duration": valid_duration,
            "hotspot_count": hotspot_count,
            "hotspot_actions": hotspot_actions,
            "is_repetitive": is_repetitive,
            "confidence_score": round(confidence, 4),
        }
        self._events.append(result)
        return result

    def _merge_nearby_hotspots(
        self, hotspots: List[Tuple[Tuple[int, int], int]],
    ) -> List[Tuple[Tuple[int, int], int]]:
        """合并邻近的热点区域。"""
        if not hotspots:
            return []
        merged: List[Tuple[Tuple[int, int], int]] = []
        used = set()
        r = self._hotspot_max_radius_cells

        for i, (cell_i, count_i) in enumerate(hotspots):
            if i in used:
                continue
            total_count = count_i
            for j, (cell_j, count_j) in enumerate(hotspots):
                if j <= i or j in used:
                    continue
                dist = max(abs(cell_i[0] - cell_j[0]), abs(cell_i[1] - cell_j[1]))
                if dist <= r:
                    total_count += count_j
                    used.add(j)
            merged.append((cell_i, total_count))
            used.add(i)

        return sorted(merged, key=lambda x: -x[1])

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def get_daily_hotspot_count(self) -> int:
        return sum(e["hotspot_count"] for e in self._events)

    def reset(self) -> None:
        self._trajectory.clear()
        self._events.clear()
        self._last_check_ts = -self._stride_sec
