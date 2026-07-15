"""日级/周期级指标聚合器。

将 VideoFeatureExtractor 输出的窗口级指标聚合为日级统计，
输出严格符合 video_tasks.md §6.1 定义的 JSON Schema。

纯 CPU 实现。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.schema_validator import SchemaValidator, get_validator

logger = logging.getLogger(__name__)


class DailyAggregator:
    """日级指标聚合器。

    收集一天内所有窗口级指标，生成符合 §6.1 Schema 的日报。

    用法:
        aggregator = DailyAggregator()
        for window_metrics in extractor_output:
            aggregator.add_window(window_metrics)
        daily_report = aggregator.aggregate(user_id="P001", date="2026-07-15")
    """

    def __init__(
        self,
        fps: float = 15.0,
        image_width: float = 640.0,
        night_start_hour: int = 22,
        night_end_hour: int = 6,
    ) -> None:
        """
        Args:
            fps: 数据帧率。
            image_width: 图像宽度（用于速度归一化）。
            night_start_hour: 夜间开始小时。
            night_end_hour: 夜间结束小时。
        """
        self._fps = fps
        self._image_width = image_width
        self._night_start_hour = night_start_hour
        self._night_end_hour = night_end_hour

        self._windows: List[Dict[str, Any]] = []
        self._validator = get_validator()

    # ---- 数据输入 ----

    def add_window(self, window_metrics: Dict[str, Any]) -> None:
        """添加一个窗口的指标。

        Args:
            window_metrics: VideoFeatureExtractor.process_frame() 的输出。
        """
        self._windows.append(window_metrics)

    def add_windows(self, windows: List[Dict[str, Any]]) -> None:
        """批量添加窗口指标。"""
        self._windows.extend(windows)

    # ---- 聚合 ----

    def aggregate(
        self,
        user_id: str = "unknown",
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """聚合所有窗口指标为日级统计。

        Args:
            user_id: 用户标识。
            date: 日期字符串 (YYYY-MM-DD)，None 则使用今天。

        Returns:
            符合 §6.1 daily_metrics Schema 的字典。
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if not self._windows:
            logger.warning("No window data to aggregate, returning zero metrics")
            return self._empty_daily(user_id, date)

        n_windows = len(self._windows)

        # --- 基础统计 ---
        total_valid_sec = sum(
            max(0, w.get("valid_duration_sec", 0)) for w in self._windows
        )

        # 有效覆盖时长（有人体检测的时间）
        coverage_sec = sum(
            w.get("valid_duration_sec", 0) * w.get("coverage_ratio", 0)
            for w in self._windows
        )

        # 活动分钟数
        active_sec = sum(
            w.get("valid_duration_sec", 0) * w.get("active_ratio", 0)
            for w in self._windows
        )
        active_minutes = active_sec / 60.0

        # 久坐比例
        sedentary_ratios = [
            w.get("sedentary_ratio", 0) for w in self._windows
            if w.get("coverage_ratio", 0) > 0
        ]
        mean_sedentary = float(np.mean(sedentary_ratios)) if sedentary_ratios else 0.0

        # 房间切换次数
        total_transitions = sum(
            w.get("room_transitions", 0) for w in self._windows
        )

        # 夜间活动次数
        total_night_frames = sum(
            w.get("night_activity_frames", 0) for w in self._windows
        )
        # 每连续 30 秒活动算一次事件
        night_events = int(total_night_frames / max(self._fps * 30, 1))

        # 社交互动时长（多人共现）
        total_multi_person_sec = sum(
            w.get("multi_person_sec", 0) for w in self._windows
        )
        social_minutes = total_multi_person_sec / 60.0

        # 重复路径次数（A1 不做专项计算，默认 0）
        repetitive_path_count = 0

        # 平均运动速度（归一化）
        velocities = [
            w.get("movement_velocity_px_s", 0)
            for w in self._windows
            if w.get("movement_velocity_px_s", 0) > 0
        ]
        mean_vel = float(np.mean(velocities)) if velocities else 0.0
        movement_speed = mean_vel / self._image_width  # 归一化到 [0, 1+)

        # 特征置信度
        confidences = [
            w.get("confidence_score", 0) for w in self._windows
        ]
        feature_confidence = float(np.mean(confidences)) if confidences else 0.0

        # 输出
        result = {
            "user_id": user_id,
            "date": date,
            "daily_metrics": {
                "active_minutes": round(active_minutes, 2),
                "sedentary_ratio": round(mean_sedentary, 4),
                "room_transition_count": int(total_transitions),
                "night_activity_count": int(night_events),
                "social_interaction_minutes": round(social_minutes, 2),
                "repetitive_path_count": int(repetitive_path_count),
                "movement_speed": round(movement_speed, 4),
                "coverage_minutes": round(coverage_sec / 60.0, 2),
                "feature_confidence": round(feature_confidence, 4),
            },
        }

        # 校验
        is_valid, errors = self._validator.validate_daily_metrics(result)
        if not is_valid:
            logger.warning(f"Daily metrics validation failed: {errors}")
            # 尝试修复
            result = self._repair_daily_metrics(result, errors)

        return result

    def aggregate_range(
        self,
        start_date: str,
        end_date: str,
        user_id: str = "unknown",
        daily_windows: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """聚合多日数据。

        Args:
            start_date: 起始日期 YYYY-MM-DD。
            end_date: 结束日期 YYYY-MM-DD。
            user_id: 用户标识。
            daily_windows: {date: [windows]} 多日窗口数据。

        Returns:
            每日报告列表。
        """
        reports = []

        if daily_windows is None:
            # 单日模式：把所有窗口当作同一天
            reports.append(self.aggregate(user_id, start_date))
        else:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            current = start
            while current <= end:
                day_str = current.strftime("%Y-%m-%d")
                day_windows = daily_windows.get(day_str, [])
                day_agg = DailyAggregator(
                    fps=self._fps,
                    image_width=self._image_width,
                )
                day_agg.add_windows(day_windows)
                reports.append(day_agg.aggregate(user_id, day_str))
                current += timedelta(days=1)

        return reports

    # ---- 辅助 ----

    def _empty_daily(self, user_id: str, date: str) -> Dict[str, Any]:
        """返回全零日级指标。"""
        return {
            "user_id": user_id,
            "date": date,
            "daily_metrics": {
                "active_minutes": 0.0,
                "sedentary_ratio": 0.0,
                "room_transition_count": 0,
                "night_activity_count": 0,
                "social_interaction_minutes": 0.0,
                "repetitive_path_count": 0,
                "movement_speed": 0.0,
                "coverage_minutes": 0.0,
                "feature_confidence": 0.0,
            },
        }

    def _repair_daily_metrics(
        self, data: Dict[str, Any], errors: List[str]
    ) -> Dict[str, Any]:
        """尝试修复不合规的日级指标。"""
        metrics = data.get("daily_metrics", {})

        # 确保所有必填字段存在且类型正确
        defaults = {
            "active_minutes": 0.0, "sedentary_ratio": 0.0,
            "room_transition_count": 0, "night_activity_count": 0,
            "social_interaction_minutes": 0.0, "repetitive_path_count": 0,
            "movement_speed": 0.0, "coverage_minutes": 0.0,
            "feature_confidence": 0.0,
        }
        for key, default in defaults.items():
            if key not in metrics:
                metrics[key] = default
            elif not isinstance(metrics[key], type(default)):
                try:
                    metrics[key] = type(default)(metrics[key])
                except (ValueError, TypeError):
                    metrics[key] = default

        # 范围约束
        metrics["sedentary_ratio"] = max(0.0, min(1.0, metrics["sedentary_ratio"]))
        metrics["feature_confidence"] = max(0.0, min(1.0, metrics["feature_confidence"]))
        metrics["active_minutes"] = max(0.0, metrics["active_minutes"])
        metrics["coverage_minutes"] = max(0.0, metrics["coverage_minutes"])
        metrics["room_transition_count"] = max(0, metrics["room_transition_count"])
        metrics["night_activity_count"] = max(0, metrics["night_activity_count"])

        data["daily_metrics"] = metrics
        return data

    def reset(self) -> None:
        """重置聚合器。"""
        self._windows.clear()

    @property
    def window_count(self) -> int:
        return len(self._windows)

    @property
    def total_coverage_sec(self) -> float:
        return sum(
            w.get("valid_duration_sec", 0) * w.get("coverage_ratio", 0)
            for w in self._windows
        )

    def __repr__(self) -> str:
        return (
            f"DailyAggregator(windows={len(self._windows)}, "
            f"coverage={self.total_coverage_sec / 60:.1f}min)"
        )
