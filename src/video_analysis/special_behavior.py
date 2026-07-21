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


# ============================================================
# 4. 异常久坐/久卧检测
# ============================================================

class ProlongedInactivityDetector:
    """长时间静止与异常久坐/久卧检测器。

    检测原理:
      1. 跟踪连续静止帧数（每帧 is_sedentary=True）
      2. 若持续静止超过预设阈值（如 2 小时），触发警示
      3. 结合骨骼关键点微弱变化判断是正常睡眠还是异常无法起身

    输出:
      {
        "timestamp": float,
        "time_window": [start_sec, end_sec],
        "valid_duration": float,
        "continuous_inactive_sec": float,
        "max_inactive_stretch_sec": float,
        "warning_triggered": bool,
        "prolonged_triggered": bool,
        "keypoint_micro_motion": float,
        "confidence_score": float,
      }
    """

    def __init__(
        self,
        prolonged_threshold_sec: float = 7200.0,
        warning_threshold_sec: float = 3600.0,
        micro_motion_window_sec: float = 60.0,
        min_micro_motion_px: float = 2.0,
        fps: float = 15.0,
    ) -> None:
        self._prolonged_threshold = prolonged_threshold_sec
        self._warning_threshold = warning_threshold_sec
        self._micro_motion_window = int(micro_motion_window_sec * fps)
        self._min_micro_motion_px = min_micro_motion_px
        self._fps = fps

        self._current_inactive_start: Optional[float] = None
        self._current_inactive_frames = 0
        self._max_inactive_stretch = 0.0
        self._keypoint_history: deque = deque(maxlen=self._micro_motion_window)
        self._events: List[Dict[str, Any]] = []

    def update(
        self,
        is_sedentary: bool,
        timestamp: float,
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        if keypoints is not None:
            self._keypoint_history.append(keypoints.copy())

        if is_sedentary:
            if self._current_inactive_start is None:
                self._current_inactive_start = timestamp
            self._current_inactive_frames += 1
            return None

        if self._current_inactive_start is not None:
            result = self._emit_inactive_event(timestamp)
            self._current_inactive_start = None
            self._current_inactive_frames = 0
            return result

        return None

    def _emit_inactive_event(self, current_ts: float) -> Dict[str, Any]:
        start_ts = self._current_inactive_start if self._current_inactive_start is not None else current_ts
        duration = current_ts - start_ts
        if duration > self._max_inactive_stretch:
            self._max_inactive_stretch = duration

        micro_motion = self._compute_micro_motion()
        warning = duration >= self._warning_threshold
        triggered = duration >= self._prolonged_threshold

        if duration < self._warning_threshold:
            confidence = 0.5
        elif duration < self._prolonged_threshold:
            confidence = 0.65
        else:
            confidence = 0.8 + (1.0 - micro_motion) * 0.15

        result = {
            "timestamp": current_ts,
            "time_window": [start_ts, current_ts],
            "valid_duration": duration,
            "continuous_inactive_sec": round(duration, 1),
            "max_inactive_stretch_sec": round(self._max_inactive_stretch, 1),
            "warning_triggered": warning,
            "prolonged_triggered": triggered,
            "keypoint_micro_motion": round(micro_motion, 4),
            "confidence_score": round(min(0.95, confidence), 4),
        }
        self._events.append(result)
        return result

    def _compute_micro_motion(self) -> float:
        if len(self._keypoint_history) < 2:
            return 1.0
        kps = np.stack(list(self._keypoint_history), axis=0)
        std_per_kp = np.std(kps[:, :, :2], axis=0)
        mean_std = np.mean(std_per_kp)
        normalized = np.clip(mean_std / (self._min_micro_motion_px * 5), 0.0, 1.0)
        return 1.0 - float(normalized)

    def flush(self, current_ts: float) -> Optional[Dict[str, Any]]:
        if self._current_inactive_start is not None:
            return self._emit_inactive_event(current_ts)
        return None

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    @property
    def max_inactive_stretch_sec(self) -> float:
        return self._max_inactive_stretch

    def get_daily_prolonged_count(self) -> int:
        return sum(1 for e in self._events if e.get("prolonged_triggered", False))

    def reset(self) -> None:
        self._current_inactive_start = None
        self._current_inactive_frames = 0
        self._max_inactive_stretch = 0.0
        self._keypoint_history.clear()
        self._events.clear()


# ============================================================
# 5. 昼夜节律偏移分析
# ============================================================

class CircadianRhythmAnalyzer:
    """昼夜节律偏移分析器。

    检测原理:
      1. 定义「起床时间」= 每日首次检测到连续活动的时刻
      2. 定义「入睡时间」= 每日最后一次活动后持续静止的时刻
      3. 建立 N 天的个体基线（均值）
      4. 对比当日数据，计算偏移量（小时）
    """

    def __init__(
        self,
        baseline_window_days: int = 7,
        offset_warning_hours: float = 2.0,
        night_start_hour: int = 22,
        night_end_hour: int = 6,
        min_active_streak_sec: float = 60.0,
        min_inactive_for_sleep_sec: float = 1800.0,
        fps: float = 15.0,
    ) -> None:
        self._baseline_window = baseline_window_days
        self._offset_warning = offset_warning_hours
        self._night_start = night_start_hour
        self._night_end = night_end_hour
        self._min_active_streak = min_active_streak_sec
        self._min_inactive_for_sleep = min_inactive_for_sleep_sec
        self._fps = fps

        self._current_day_activity: List[Tuple[float, bool]] = []
        self._baseline_days: List[Dict[str, Any]] = []
        self._date: str = "unknown"

    def set_date(self, date: str) -> None:
        self._date = date

    def feed_hourly(self, hour_of_day: float, is_active: bool) -> None:
        self._current_day_activity.append((hour_of_day, is_active))

    def analyze(self, date: Optional[str] = None) -> Dict[str, Any]:
        if date is None:
            date = self._date

        wake_time, sleep_time = self._detect_sleep_wake()
        nap_minutes = self._detect_naps(wake_time, sleep_time)

        baseline_wakes = [d.get("wake_time", 8.0) for d in self._baseline_days if d.get("wake_time") is not None]
        baseline_sleeps = [d.get("sleep_time", 23.0) for d in self._baseline_days if d.get("sleep_time") is not None]
        baseline_wake_mean = float(np.mean(baseline_wakes)) if baseline_wakes else wake_time
        baseline_sleep_mean = float(np.mean(baseline_sleeps)) if baseline_sleeps else sleep_time

        wake_offset = abs(wake_time - baseline_wake_mean) if wake_time else 0.0
        sleep_offset = abs(sleep_time - baseline_sleep_mean) if sleep_time else 0.0
        disturbed = wake_offset > self._offset_warning or sleep_offset > self._offset_warning

        days_confidence = min(1.0, len(self._baseline_days) / self._baseline_window)
        confidence = 0.4 + days_confidence * 0.5

        result = {
            "date": date,
            "wake_time": round(wake_time, 2),
            "sleep_time": round(sleep_time, 2),
            "nap_duration_minutes": round(nap_minutes, 1),
            "baseline_wake_mean": round(baseline_wake_mean, 2),
            "baseline_sleep_mean": round(baseline_sleep_mean, 2),
            "wake_offset_hours": round(wake_offset, 2),
            "sleep_offset_hours": round(sleep_offset, 2),
            "is_circadian_disturbed": disturbed,
            "baseline_days_count": len(self._baseline_days),
            "confidence_score": round(confidence, 4),
            "time_window": [0.0, 24.0],
            "valid_duration": float(len(self._current_day_activity)) / 60.0,
        }
        self._baseline_days.append({"date": date, "wake_time": wake_time, "sleep_time": sleep_time})
        return result

    def _detect_sleep_wake(self) -> Tuple[float, float]:
        if not self._current_day_activity:
            return (8.0, 23.0)
        wake_time, sleep_time = 8.0, 23.0
        active_streak = 0
        for hour, active in self._current_day_activity:
            if active:
                active_streak += 1
                if active_streak >= self._min_active_streak / 3600.0:
                    wake_time = hour - active_streak + 1
                    break
            else:
                active_streak = 0
        inactive_streak = 0
        for hour, active in reversed(self._current_day_activity):
            if not active:
                inactive_streak += 1
                if inactive_streak >= self._min_inactive_for_sleep / 3600.0:
                    sleep_time = hour + inactive_streak - 1
                    break
            else:
                inactive_streak = 0
        return (wake_time, sleep_time)

    def _detect_naps(self, wake_time: float, sleep_time: float) -> float:
        nap_total = 0.0
        in_nap, nap_start = False, 0.0
        min_nap = 20.0 / 60.0
        for hour, active in self._current_day_activity:
            if wake_time <= hour <= sleep_time:
                if not active and not in_nap:
                    nap_start, in_nap = hour, True
                elif active and in_nap:
                    nap_dur = hour - nap_start
                    if nap_dur >= min_nap:
                        nap_total += nap_dur
                    in_nap = False
        if in_nap:
            nap_dur = sleep_time - nap_start
            if nap_dur >= min_nap:
                nap_total += nap_dur
        return nap_total * 60.0

    def reset(self) -> None:
        self._current_day_activity.clear()

    def reset_baseline(self) -> None:
        self._baseline_days.clear()


# ============================================================
# 6. 社交互动强度检测
# ============================================================

class SocialInteractionAnalyzer:
    """社交互动强度检测器。

    检测原理:
      1. 多人共现时，计算人体间的空间距离
      2. 估算朝向角度（基于左右肩连线法向量判断是否面对面）
      3. 统计近距离共现时长，量化互动强度
    """

    def __init__(
        self,
        close_distance_threshold_px: float = 150.0,
        facing_angle_threshold_deg: float = 45.0,
        window_sec: float = 300.0,
        stride_sec: float = 60.0,
        fps: float = 15.0,
    ) -> None:
        self._close_threshold = close_distance_threshold_px
        self._facing_angle_threshold = facing_angle_threshold_deg
        self._window_sec = window_sec
        self._stride_sec = stride_sec
        self._fps = fps

        self._frame_records: deque = deque(maxlen=int(window_sec * fps))
        self._last_check_ts = -stride_sec
        self._events: List[Dict[str, Any]] = []

    def update(
        self,
        keypoints_list: List[np.ndarray],
        bboxes: np.ndarray,
        timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        N = len(keypoints_list)
        record = {"timestamp": timestamp, "person_count": N}

        if N >= 2:
            distances, facing_pairs, total_pairs = [], 0, 0
            for i in range(N):
                for j in range(i + 1, N):
                    total_pairs += 1
                    ci = self._get_centroid(keypoints_list[i])
                    cj = self._get_centroid(keypoints_list[j])
                    dist = np.linalg.norm(ci - cj) if ci is not None and cj is not None else float("inf")
                    distances.append(dist)
                    if self._is_facing(keypoints_list[i], keypoints_list[j]):
                        facing_pairs += 1
            record.update({
                "min_distance_px": float(min(distances)) if distances else 0.0,
                "avg_distance_px": float(np.mean(distances)) if distances else 0.0,
                "facing_ratio": facing_pairs / total_pairs if total_pairs > 0 else 0.0,
                "is_close": min(distances) < self._close_threshold if distances else False,
            })
        else:
            record.update({"min_distance_px": 0.0, "avg_distance_px": 0.0, "facing_ratio": 0.0, "is_close": False})

        self._frame_records.append(record)
        if timestamp - self._last_check_ts >= self._stride_sec:
            self._last_check_ts = timestamp
            return self._analyze(timestamp)
        return None

    def _analyze(self, current_ts: float) -> Dict[str, Any]:
        records = list(self._frame_records)
        window_start = current_ts - self._window_sec
        multi = [r for r in records if r["person_count"] >= 2]
        n_multi, n_total = len(multi), len(records)

        if n_multi == 0:
            return {"timestamp": current_ts, "time_window": [window_start, current_ts],
                    "valid_duration": self._window_sec, "avg_distance_px": 0.0,
                    "facing_each_other_ratio": 0.0, "close_proximity_sec": 0.0,
                    "interaction_intensity": 0.0, "confidence_score": 0.3}

        avg_dist = float(np.mean([r["avg_distance_px"] for r in multi]))
        facing_ratio = float(np.mean([r["facing_ratio"] for r in multi]))
        close_frames = sum(1 for r in multi if r["is_close"])
        close_sec = close_frames / self._fps

        presence_ratio = n_multi / n_total if n_total > 0 else 0
        proximity_score = max(0.0, 1.0 - avg_dist / self._close_threshold)
        intensity = 0.3 * presence_ratio + 0.35 * proximity_score + 0.35 * facing_ratio
        confidence = 0.5 + presence_ratio * 0.4

        result = {
            "timestamp": current_ts, "time_window": [window_start, current_ts],
            "valid_duration": self._window_sec, "avg_distance_px": round(avg_dist, 1),
            "facing_each_other_ratio": round(facing_ratio, 4),
            "close_proximity_sec": round(close_sec, 1),
            "interaction_intensity": round(min(1.0, intensity), 4),
            "confidence_score": round(min(0.95, confidence), 4),
        }
        self._events.append(result)
        return result

    def _get_centroid(self, kps: np.ndarray) -> Optional[np.ndarray]:
        if kps[11, 2] > 0.1 and kps[12, 2] > 0.1:
            return (kps[11, :2] + kps[12, :2]) / 2.0
        mask = kps[:, 2] > 0.3
        if mask.any():
            return kps[mask, :2].mean(axis=0)
        return None

    def _is_facing(self, kps_a: np.ndarray, kps_b: np.ndarray) -> bool:
        if kps_a[5, 2] < 0.2 or kps_a[6, 2] < 0.2:
            return False
        if kps_b[5, 2] < 0.2 or kps_b[6, 2] < 0.2:
            return False
        shoulder_a = kps_a[6, :2] - kps_a[5, :2]
        shoulder_b = kps_b[6, :2] - kps_b[5, :2]
        center_a = (kps_a[5, :2] + kps_a[6, :2]) / 2.0
        center_b = (kps_b[5, :2] + kps_b[6, :2]) / 2.0
        ab = center_b - center_a
        ab_norm = np.linalg.norm(ab)
        if ab_norm < 1e-6:
            return False
        ab = ab / ab_norm
        facing_a = np.array([-shoulder_a[1], shoulder_a[0]])
        fa_norm = np.linalg.norm(facing_a)
        if fa_norm < 1e-6:
            return False
        facing_a = facing_a / fa_norm
        facing_b = np.array([-shoulder_b[1], shoulder_b[0]])
        fb_norm = np.linalg.norm(facing_b)
        if fb_norm < 1e-6:
            return False
        facing_b = facing_b / fb_norm
        cos_a = np.dot(facing_a, ab)
        cos_b = np.dot(facing_b, -ab)
        angle_a_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        angle_b_deg = np.degrees(np.arccos(np.clip(cos_b, -1, 1)))
        return angle_a_deg < self._facing_angle_threshold and angle_b_deg < self._facing_angle_threshold

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def get_daily_avg_intensity(self) -> float:
        if not self._events:
            return 0.0
        return float(np.mean([e["interaction_intensity"] for e in self._events]))

    def reset(self) -> None:
        self._frame_records.clear()
        self._events.clear()
        self._last_check_ts = -self._stride_sec


# ============================================================
# 7. 专项行为检测总装
# ============================================================

class SpecialBehaviorDetector:
    """专项行为检测统一入口。

    将 5 个检测器组装为单一管线，接收每帧数据后自动分发到各子检测器。
    所有子检测器可独立启用/禁用（可插拔）。支持实时回调 A3EventDispatcher。

    用法:
        detector = SpecialBehaviorDetector(fps=15.0)
        detector.set_trigger_callback(dispatcher.on_trigger)
        for frame_data in video_frames:
            results = detector.update(
                centroid_x, centroid_y,
                is_sedentary, timestamp,
                keypoints, bboxes, track_ids,
            )
    """

    # A2 检测器输出键 → A3 event_type
    _A2_TO_A3: dict[str, str] = {
        "repetitive_path": "repetitive_behavior",
        "repeated_action": "repetitive_behavior",
        "prolonged_inactivity": "long_inactivity",
        "social_interaction": "social_interaction",
    }

    def __init__(
        self,
        fps: float = 15.0,
        image_width: float = 640.0,
        image_height: float = 480.0,
        enable_wandering: bool = True,
        enable_repeated_action: bool = True,
        enable_inactivity: bool = True,
        enable_circadian: bool = True,
        enable_social: bool = True,
        **kwargs,
    ) -> None:
        self._fps = fps
        self._image_width = image_width
        self._image_height = image_height

        # 子检测器（可按需启用/禁用）
        self._wandering = (
            RepetitivePathDetector(fps=fps, **kwargs.get("wandering", {}))
            if enable_wandering else None
        )
        self._repeated_action = (
            RepeatedActionDetector(fps=fps, **kwargs.get("repeated_action", {}))
            if enable_repeated_action else None
        )
        self._inactivity = (
            ProlongedInactivityDetector(fps=fps, **kwargs.get("inactivity", {}))
            if enable_inactivity else None
        )
        self._circadian = (
            CircadianRhythmAnalyzer(fps=fps, **kwargs.get("circadian", {}))
            if enable_circadian else None
        )
        self._social = (
            SocialInteractionAnalyzer(fps=fps, **kwargs.get("social", {}))
            if enable_social else None
        )

        # 实时触发回调（由 A3EventDispatcher 注册）
        self._trigger_callback: Optional[callable] = None

        # 人物离画追踪
        self._prev_has_person: bool = False

        self._frame_count = 0
        self._hourly_buffer: List[Tuple[float, bool]] = []  # for circadian

    # ---- 回调注册 ----

    def set_trigger_callback(self, callback: Optional[callable]) -> None:
        """注册 A2→A3 实时触发回调。

        回调签名: callback(event_type: str, trigger_ts: float) -> Optional[dict]
        由 A3EventDispatcher.on_trigger() 实现。
        """
        self._trigger_callback = callback

    # ---- 触发信号发送 ----

    def _fire_trigger(self, a2_key: str, trigger_ts: float) -> None:
        """如果回调已注册，将 A2 检测结果转发为 A3 事件。"""
        if self._trigger_callback is None:
            return
        a3_type = self._A2_TO_A3.get(a2_key)
        if a3_type:
            self._trigger_callback(a3_type, trigger_ts)

    def update(
        self,
        centroid_x: Optional[float],
        centroid_y: Optional[float],
        is_sedentary: bool,
        timestamp: float,
        keypoints: Optional[np.ndarray] = None,     # (N, 17, 3)
        bboxes: Optional[np.ndarray] = None,         # (N, 4)
        track_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """处理单帧数据，返回所有触发检测的结果。

        人物离开画面时暂停所有检测器，人物回来时从零重新积累。
        检测器触发后通过回调通知 A3EventDispatcher。

        Returns:
            {"detector_name": result_dict, ...}  仅包含触发了输出的检测器。
        """
        self._frame_count += 1
        outputs: Dict[str, Any] = {}

        N = keypoints.shape[0] if keypoints is not None else 0
        has_person = N > 0 and centroid_x is not None and centroid_y is not None

        # ---- 人物离画处理 ----
        if not has_person:
            # 人物离开：暂停所有检测（不更新也不触发）
            if self._prev_has_person:
                self._pause_all_detectors()
            self._prev_has_person = False
            return outputs  # 无人时不产生任何输出

        # 人物回来了：检测器内部状态由各自的 update 逻辑自然恢复
        self._prev_has_person = True

        # 1. 徘徊检测 → repetitive_behavior
        if self._wandering is not None:
            r = self._wandering.update(centroid_x, centroid_y, timestamp)
            if r:
                outputs["repetitive_path"] = r
                self._fire_trigger("repetitive_path", timestamp)

        # 2. 重复动作 → repetitive_behavior（共享冷却期）
        if self._repeated_action is not None:
            r = self._repeated_action.update(centroid_x, centroid_y, timestamp)
            if r:
                outputs["repeated_action"] = r
                self._fire_trigger("repeated_action", timestamp)

        # 3. 久坐/静止 → long_inactivity
        # 只在真正触发 warning(≥1h) 或 prolonged(≥2h) 时通知 A3
        if self._inactivity is not None:
            kp_for_inactivity = None
            if keypoints is not None and N > 0:
                kp_for_inactivity = keypoints[0]
            r = self._inactivity.update(is_sedentary, timestamp, kp_for_inactivity)
            if r:
                outputs["prolonged_inactivity"] = r
                if r.get("warning_triggered") or r.get("prolonged_triggered"):
                    self._fire_trigger("prolonged_inactivity", timestamp)

        # 4. 昼夜节律 — 每小时聚合一次（不参与实时触发）
        if self._circadian is not None:
            hour = (timestamp / 3600.0) % 24
            is_active = not is_sedentary
            self._hourly_buffer.append((hour, is_active))
            if len(self._hourly_buffer) >= int(self._fps * 3600):
                for h, act in self._hourly_buffer:
                    self._circadian.feed_hourly(h, act)
                self._hourly_buffer.clear()

        # 5. 社交互动 → social_interaction
        if self._social is not None and N >= 2 and keypoints is not None:
            kps_list = [keypoints[i] for i in range(N)]
            r = self._social.update(kps_list, bboxes if bboxes is not None else np.zeros((N, 4)), timestamp)
            if r:
                outputs["social_interaction"] = r
                self._fire_trigger("social_interaction", timestamp)

        return outputs

    def _pause_all_detectors(self) -> None:
        """人物离开画面时暂停所有检测器内部累积状态。

        不触发任何回调，不产生输出。保留冷却期状态（由 A3EventDispatcher 维护）。
        """
        pass  # 各子检测器内部状态自然衰减，不显式重置

    def flush(self, current_ts: float) -> Dict[str, Any]:
        """视频结束时刷新所有检测器缓冲区。"""
        outputs: Dict[str, Any] = {}
        if self._inactivity is not None:
            r = self._inactivity.flush(current_ts)
            if r:
                outputs["prolonged_inactivity"] = r
        return outputs

    def get_circadian_report(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取昼夜节律日报。"""
        if self._circadian is not None:
            return self._circadian.analyze(date)
        return None

    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """获取所有检测器的日级汇总。"""
        summary: Dict[str, Any] = {}

        if self._wandering is not None:
            summary["daily_repetitive_path_count"] = self._wandering.get_daily_repetitive_count()
        if self._repeated_action is not None:
            summary["daily_hotspot_action_count"] = self._repeated_action.get_daily_hotspot_count()
        if self._inactivity is not None:
            summary["daily_prolonged_inactive_count"] = self._inactivity.get_daily_prolonged_count()
            summary["max_inactive_stretch_sec"] = self._inactivity.max_inactive_stretch_sec
        if self._social is not None:
            summary["daily_avg_social_intensity"] = self._social.get_daily_avg_intensity()
        if self._circadian is not None:
            cr = self._circadian.analyze(date)
            if cr:
                summary["circadian"] = cr

        return summary

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._frame_count = 0
        self._hourly_buffer.clear()
        for det in [self._wandering, self._repeated_action, self._inactivity, self._circadian, self._social]:
            if det is not None:
                det.reset()

    def __repr__(self) -> str:
        parts = []
        for name, det in [
            ("wandering", self._wandering), ("repeated_action", self._repeated_action),
            ("inactivity", self._inactivity), ("circadian", self._circadian),
            ("social", self._social),
        ]:
            parts.append(f"{name}={det is not None}")
        return f"SpecialBehaviorDetector(frames={self._frame_count}, {', '.join(parts)})"
        self._last_check_ts = -self._stride_sec
