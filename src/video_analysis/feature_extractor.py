"""视频特征提取器。

消费标准化的 PerFrameData 流，通过滑动窗口计算 6 项基础行为指标：
  1. activity_minutes — 活动分钟数
  2. sedentary_ratio — 久坐/静止比例
  3. room_transitions — 房间/区域切换次数
  4. movement_velocity — 平均/瞬时运动速度
  5. night_activity_stats — 夜间活动统计
  6. multi_person_duration — 多人共现时长

全部计算在 CPU 上完成，不依赖 GPU。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.skeleton_parser import SkeletonParser
from src.video_analysis.sliding_window import SlidingWindow

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """视频行为特征提取器。

    消费标准化的人体关键点帧数据，通过滑动窗口累计计算
    6 项基础行为指标。同时支持生产模式（RGB 视频 + YOLO 推理）
    和验证模式（直接读取骨骼数据）。

    用法:
        extractor = VideoFeatureExtractor(config)
        for frame_data in data_loader.frames():
            window_metrics = extractor.process_frame(frame_data)
            if window_metrics:
                print(window_metrics)
        daily = extractor.get_daily_summary()
    """

    def __init__(
        self,
        window_size_sec: float = 300.0,
        window_stride_sec: float = 60.0,
        fps: float = 15.0,
        night_start_hour: int = 22,
        night_end_hour: int = 6,
        image_width: float = 640.0,
        image_height: float = 480.0,
        sedentary_max_displacement_px: float = 50.0,
        sedentary_min_duration_sec: float = 300.0,
        grid_resolution: int = 50,
        max_history_windows: int = 288,
    ) -> None:
        """
        Args:
            window_size_sec: 滑动窗口大小（秒），默认 300 (5分钟)。
            window_stride_sec: 窗口输出步长（秒），默认 60 (1分钟)。
            fps: 数据帧率。
            night_start_hour: 夜间开始小时，默认 22。
            night_end_hour: 夜间结束小时，默认 6。
            image_width: 图像宽度（用于坐标归一化参考）。
            image_height: 图像高度。
            sedentary_max_displacement_px: 判定静止的质心最大位移（像素）。
            sedentary_min_duration_sec: 判定久坐的最短累计静止时长（秒）。
            grid_resolution: 房间切换检测的空间网格分辨率（像素）。
            max_history_windows: 保留的最大历史窗口数（用于日级聚合）。
        """
        self._window_size_sec = window_size_sec
        self._window_stride_sec = window_stride_sec
        self._fps = fps
        self._night_start_hour = night_start_hour
        self._night_end_hour = night_end_hour
        self._image_width = image_width
        self._image_height = image_height
        self._sedentary_max_displacement_px = sedentary_max_displacement_px
        self._sedentary_min_duration_sec = sedentary_min_duration_sec
        self._grid_resolution = grid_resolution

        # 工具
        self._skel_parser = SkeletonParser(image_width, image_height)

        # 状态
        self._frame_count = 0
        self._elapsed_sec = 0.0
        self._prev_centroids: Dict[int, np.ndarray] = {}  # track_id → (x, y)
        self._last_grid_cell: Optional[Tuple[int, int]] = None
        self._night_activity_count = 0
        self._total_valid_frames = 0

        # 滑动窗口 — 存储每帧的基础特征
        self._window = SlidingWindow(
            max_size=int(window_size_sec * fps),
            max_duration_sec=window_size_sec,
            timestamp_key="timestamp",
        )

        # 窗口输出历史（用于日级聚合）
        self._window_history: List[Dict[str, Any]] = []
        self._max_history_windows = max_history_windows
        self._last_emit_ts = -window_stride_sec  # 确保第 0 秒即可输出

        # 全局累计器
        self._cumulative = _CumulativeMetrics()

    # ---- 核心处理 ----

    def process_frame(
        self,
        keypoints: np.ndarray,
        bboxes: np.ndarray,
        track_ids: List[int],
        timestamp: float,
        frame_index: int,
    ) -> Optional[Dict[str, Any]]:
        """处理单帧数据并更新窗口。

        Args:
            keypoints: (N, 17, 3) 关键点数组。
            bboxes: (N, 4) 检测框 xyxy。
            track_ids: 每个检测目标的 track ID 列表。
            timestamp: 帧时间戳（秒）。
            frame_index: 帧序号。

        Returns:
            若到达窗口输出周期则返回窗口级指标，否则返回 None。
        """
        self._frame_count += 1
        N = len(track_ids)
        self._elapsed_sec = timestamp

        # --- 每帧基础特征 ---
        frame_features: Dict[str, Any] = {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "person_count": N,
            "is_multi_person": N >= 2,
            "is_night": self._is_night_time(timestamp),
        }

        if N > 0:
            # 质心（髋部中点）
            centroids = self._skel_parser.get_centroid_sequence(keypoints)  # (N, 2)
            # get_centroid_sequence expects (T, K, 3), but we have (N, K, 3) per frame
            # Actually, for per-frame data, let me compute directly
            centroids = self._compute_centroids(keypoints)  # (N, 2)

            # 运动速度（与上一帧相比）
            velocities = []
            displacements = []
            for i, tid in enumerate(track_ids):
                c = centroids[i]
                if not np.isnan(c).any():
                    if tid in self._prev_centroids:
                        prev_c = self._prev_centroids[tid]
                        disp = np.linalg.norm(c - prev_c)
                    else:
                        disp = 0.0
                    displacements.append(disp)
                    velocities.append(disp * self._fps)  # px/s
                else:
                    displacements.append(0.0)
                    velocities.append(0.0)
                self._prev_centroids[tid] = c

            mean_centroid = np.nanmean(centroids, axis=0) if N > 0 else np.array([np.nan, np.nan])
            max_disp = max(displacements) if displacements else 0.0
            mean_vel = float(np.mean(velocities)) if velocities else 0.0

            # 网格位置（用于检测房间切换）
            grid_cell = self._compute_grid_cell(mean_centroid)
            has_room_transition = (
                self._last_grid_cell is not None
                and grid_cell != self._last_grid_cell
                and not np.isnan(mean_centroid).any()
            )
            if has_room_transition:
                self._last_grid_cell = grid_cell
            elif self._last_grid_cell is None:
                self._last_grid_cell = grid_cell

            frame_features.update({
                "centroid_x": float(mean_centroid[0]) if not np.isnan(mean_centroid[0]) else None,
                "centroid_y": float(mean_centroid[1]) if not np.isnan(mean_centroid[1]) else None,
                "max_displacement_px": max_disp,
                "mean_velocity_px_s": mean_vel,
                "is_sedentary": max_disp < self._sedentary_max_displacement_px,
                "room_transition": has_room_transition,
                "grid_cell": grid_cell,
            })
        else:
            frame_features.update({
                "centroid_x": None,
                "centroid_y": None,
                "max_displacement_px": 0.0,
                "mean_velocity_px_s": 0.0,
                "is_sedentary": True,  # 无人时视为静止
                "room_transition": False,
                "grid_cell": None,
            })

        # 累积有效帧
        if N > 0:
            self._total_valid_frames += 1
        self._cumulative.update(frame_features)

        # 入窗
        self._window.append(frame_features)

        # 检查是否到达输出周期
        if timestamp - self._last_emit_ts >= self._window_stride_sec:
            self._last_emit_ts = timestamp
            return self._compute_window_metrics(timestamp)

        return None

    def _compute_centroids(self, keypoints: np.ndarray) -> np.ndarray:
        """从 (N, 17, 3) 关键点计算 N 个质心（髋部中点）。

        Args:
            keypoints: (N, 17, 3)

        Returns:
            (N, 2) 质心 (x, y)
        """
        N = keypoints.shape[0]
        centroids = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            left_hip = keypoints[i, 11, :2]
            right_hip = keypoints[i, 12, :2]
            left_conf = keypoints[i, 11, 2]
            right_conf = keypoints[i, 12, 2]
            valid = left_conf > 0 and right_conf > 0
            if valid:
                centroids[i] = (left_hip + right_hip) / 2.0
            else:
                # 回退到所有关键点均值
                conf_mask = keypoints[i, :, 2] > 0.3
                if conf_mask.any():
                    centroids[i] = keypoints[i, conf_mask, :2].mean(axis=0)
                else:
                    centroids[i] = [np.nan, np.nan]
        return centroids

    def _compute_grid_cell(self, centroid: np.ndarray) -> Optional[Tuple[int, int]]:
        """将质心坐标映射到空间网格单元。"""
        if np.isnan(centroid).any():
            return None
        gx = int(centroid[0] / self._grid_resolution)
        gy = int(centroid[1] / self._grid_resolution)
        return (gx, gy)

    def _is_night_time(self, timestamp: float) -> bool:
        """判断时间戳是否处于夜间时段。

        使用累计秒数配合基准时间。由于视频没有绝对时钟，
        这里基于 elapsed_sec 推算虚拟小时（假设 00:00 为第 0 秒）。
        """
        hours = (timestamp / 3600.0) % 24
        if self._night_start_hour < self._night_end_hour:
            return self._night_start_hour <= hours < self._night_end_hour
        else:
            # 跨日（如 22:00 – 06:00）
            return hours >= self._night_start_hour or hours < self._night_end_hour

    # ---- 窗口级指标计算 ----

    def _compute_window_metrics(self, timestamp: float) -> Dict[str, Any]:
        """计算当前窗口的聚合指标。"""
        records = self._window.get_all()
        if not records:
            return self._empty_window_metrics(timestamp)

        n = len(records)
        window_start = records[0]["timestamp"]
        window_end = records[-1]["timestamp"]
        valid_duration = window_end - window_start

        # 有效帧数（有人体检测的帧）
        frames_with_person = sum(1 for r in records if r["person_count"] > 0)
        coverage_ratio = frames_with_person / n if n > 0 else 0.0
        coverage_sec = valid_duration * coverage_ratio

        # 活动帧（非静止）
        active_frames = sum(
            1 for r in records
            if r["person_count"] > 0 and not r.get("is_sedentary", True)
        )
        active_ratio = active_frames / n if n > 0 else 0.0

        # 久坐帧
        sedentary_frames = sum(
            1 for r in records
            if r.get("is_sedentary", True) and r["person_count"] > 0
        )
        sedentary_ratio = sedentary_frames / n if n > 0 else 0.0

        # 房间切换
        room_transitions = sum(1 for r in records if r.get("room_transition", False))

        # 运动速度
        velocities = [
            r["mean_velocity_px_s"]
            for r in records
            if r["person_count"] > 0 and r["mean_velocity_px_s"] is not None
        ]
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0

        # 夜间活动
        night_activity = sum(
            1 for r in records
            if r.get("is_night", False)
            and r["person_count"] > 0
            and not r.get("is_sedentary", True)
        )

        # 多人共现
        multi_person_frames = sum(1 for r in records if r.get("is_multi_person", False))
        multi_person_sec = multi_person_frames / self._fps if self._fps > 0 else 0.0

        metrics = {
            "window_start_sec": float(window_start),
            "window_end_sec": float(window_end),
            "valid_duration_sec": valid_duration,
            "coverage_ratio": coverage_ratio,
            "active_ratio": active_ratio,
            "sedentary_ratio": sedentary_ratio,
            "room_transitions": room_transitions,
            "movement_velocity_px_s": mean_velocity,
            "night_activity_frames": night_activity,
            "multi_person_sec": multi_person_sec,
            "person_count_mean": np.mean([r["person_count"] for r in records]),
            "confidence_score": coverage_ratio * 0.7 + 0.3,  # 简化的置信度估计
        }

        # 保存到历史
        self._window_history.append(metrics)
        if len(self._window_history) > self._max_history_windows:
            self._window_history.pop(0)

        return metrics

    def _empty_window_metrics(self, timestamp: float) -> Dict[str, Any]:
        return {
            "window_start_sec": timestamp - self._window_stride_sec,
            "window_end_sec": timestamp,
            "valid_duration_sec": 0.0,
            "coverage_ratio": 0.0,
            "active_ratio": 0.0,
            "sedentary_ratio": 0.0,
            "room_transitions": 0,
            "movement_velocity_px_s": 0.0,
            "night_activity_frames": 0,
            "multi_person_sec": 0.0,
            "person_count_mean": 0.0,
            "confidence_score": 0.0,
        }

    # ---- 日级汇总 ----

    def get_daily_summary(
        self, user_id: str = "unknown", date: str = "1970-01-01"
    ) -> Dict[str, Any]:
        """基于窗口历史输出日级指标（符合 §6.1 Schema）。

        若窗口历史为空，基于全局累计值计算。

        Returns:
            符合 daily_metrics Schema 的字典。
        """
        if self._window_history:
            return self._summary_from_windows(user_id, date)
        else:
            return self._summary_from_cumulative(user_id, date)

    def _summary_from_windows(
        self, user_id: str, date: str
    ) -> Dict[str, Any]:
        """从窗口历史聚合日级指标。"""
        total_sec = sum(
            max(0, w["valid_duration_sec"]) for w in self._window_history
        )
        coverage_sec = sum(
            w["valid_duration_sec"] * w["coverage_ratio"]
            for w in self._window_history
        )
        active_sec = sum(
            w["valid_duration_sec"] * w["active_ratio"]
            for w in self._window_history
        )
        sedentary_ratio = (
            sum(w["sedentary_ratio"] for w in self._window_history)
            / len(self._window_history)
            if self._window_history else 0.0
        )
        total_transitions = sum(w["room_transitions"] for w in self._window_history)
        night_frames = sum(w["night_activity_frames"] for w in self._window_history)
        night_events = int(night_frames / (self._fps * 30))  # 每 30 秒活动 = 1 次事件
        multi_person_sec = sum(w["multi_person_sec"] for w in self._window_history)
        mean_vel = (
            np.mean([w["movement_velocity_px_s"] for w in self._window_history])
            if self._window_history else 0.0
        )
        conf = (
            np.mean([w["confidence_score"] for w in self._window_history])
            if self._window_history else 0.0
        )

        return {
            "user_id": user_id,
            "date": date,
            "daily_metrics": {
                "active_minutes": round(active_sec / 60.0, 2),
                "sedentary_ratio": round(float(sedentary_ratio), 4),
                "room_transition_count": int(total_transitions),
                "night_activity_count": int(night_events),
                "social_interaction_minutes": round(multi_person_sec / 60.0, 2),
                "repetitive_path_count": 0,  # A2 中实现
                "movement_speed": round(float(mean_vel) / self._image_width, 4),  # 归一化
                "coverage_minutes": round(coverage_sec / 60.0, 2),
                "feature_confidence": round(float(conf), 4),
            },
        }

    def _summary_from_cumulative(
        self, user_id: str, date: str
    ) -> Dict[str, Any]:
        """从累计数据估算日级指标（窗口不足时回退）。"""
        cum = self._cumulative
        total_sec = max(self._elapsed_sec, 1.0)

        return {
            "user_id": user_id,
            "date": date,
            "daily_metrics": {
                "active_minutes": round(cum.active_sec / 60.0, 2),
                "sedentary_ratio": round(
                    cum.sedentary_frames / max(cum.frames_with_person, 1), 4
                ),
                "room_transition_count": cum.room_transitions,
                "night_activity_count": cum.night_activity_events,
                "social_interaction_minutes": round(
                    cum.multi_person_frames / max(self._fps, 1) / 60.0, 2
                ),
                "repetitive_path_count": 0,
                "movement_speed": round(cum.mean_velocity / self._image_width, 4),
                "coverage_minutes": round(
                    cum.frames_with_person / max(self._fps, 1) / 60.0, 2
                ),
                "feature_confidence": round(
                    min(1.0, cum.frames_with_person / max(self._frame_count, 1) + 0.3), 4
                ),
            },
        }

    def reset(self) -> None:
        """重置提取器状态。"""
        self._frame_count = 0
        self._elapsed_sec = 0.0
        self._prev_centroids.clear()
        self._last_grid_cell = None
        self._night_activity_count = 0
        self._total_valid_frames = 0
        self._window.clear()
        self._window_history.clear()
        self._cumulative = _CumulativeMetrics()
        self._last_emit_ts = -self._window_stride_sec

    def __repr__(self) -> str:
        return (
            f"VideoFeatureExtractor(window={self._window_size_sec}s, "
            f"stride={self._window_stride_sec}s, "
            f"frames={self._frame_count}, "
            f"history_windows={len(self._window_history)})"
        )


class _CumulativeMetrics:
    """累计指标辅助类。"""

    def __init__(self) -> None:
        self.frames_with_person = 0
        self.active_sec = 0.0
        self.sedentary_frames = 0
        self.room_transitions = 0
        self.multi_person_frames = 0
        self.night_activity_events = 0
        self.mean_velocity = 0.0
        self._vel_sum = 0.0
        self._vel_count = 0

    def update(self, frame: Dict[str, Any]) -> None:
        N = frame.get("person_count", 0)
        if N > 0:
            self.frames_with_person += 1
            if not frame.get("is_sedentary", True):
                self.active_sec += 1.0  # 1 frame
            if frame.get("is_sedentary", True):
                self.sedentary_frames += 1
            if frame.get("room_transition", False):
                self.room_transitions += 1
            if frame.get("is_night", False) and not frame.get("is_sedentary", True):
                self.night_activity_events += 1
        if frame.get("is_multi_person", False):
            self.multi_person_frames += 1
        v = frame.get("mean_velocity_px_s", 0.0)
        if v is not None and v > 0:
            self._vel_sum += v
            self._vel_count += 1
            self.mean_velocity = self._vel_sum / max(self._vel_count, 1)
