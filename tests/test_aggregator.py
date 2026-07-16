"""日级聚合器测试。A1.9"""

import pytest
from src.video_analysis.aggregator import DailyAggregator


class TestDailyAggregator:
    """DailyAggregator 测试。"""

    def test_empty_aggregation(self):
        agg = DailyAggregator()
        report = agg.aggregate(user_id="P001", date="2026-07-15")
        assert report["user_id"] == "P001"
        assert report["date"] == "2026-07-15"
        # 所有指标应为零
        for v in report["daily_metrics"].values():
            assert v == 0.0 or v == 0

    def test_aggregate_with_windows(self, aggregator):
        """聚合多个窗口指标。"""
        windows = [
            {
                "window_start_sec": 0.0, "window_end_sec": 60.0,
                "valid_duration_sec": 60.0, "coverage_ratio": 0.8,
                "active_ratio": 0.5, "sedentary_ratio": 0.5,
                "room_transitions": 3, "movement_velocity_px_s": 30.0,
                "night_activity_frames": 0, "multi_person_sec": 10.0,
                "person_count_mean": 1.0, "confidence_score": 0.9,
            },
            {
                "window_start_sec": 60.0, "window_end_sec": 120.0,
                "valid_duration_sec": 60.0, "coverage_ratio": 0.9,
                "active_ratio": 0.6, "sedentary_ratio": 0.4,
                "room_transitions": 5, "movement_velocity_px_s": 50.0,
                "night_activity_frames": 0, "multi_person_sec": 0.0,
                "person_count_mean": 1.0, "confidence_score": 0.95,
            },
        ]
        aggregator.add_windows(windows)
        report = aggregator.aggregate(user_id="P001", date="2026-07-15")

        m = report["daily_metrics"]
        assert m["room_transition_count"] == 8
        assert m["social_interaction_minutes"] > 0  # 窗口1有 10s
        assert m["feature_confidence"] > 0.9

    def test_schema_validation(self, aggregator):
        """输出必须通过 §6.1 Schema 校验。"""
        from src.utils.schema_validator import get_validator

        aggregator.add_window({
            "window_start_sec": 0, "window_end_sec": 60,
            "valid_duration_sec": 60, "coverage_ratio": 1.0,
            "active_ratio": 0.5, "sedentary_ratio": 0.5,
            "room_transitions": 2, "movement_velocity_px_s": 10,
            "night_activity_frames": 0, "multi_person_sec": 0,
            "person_count_mean": 1, "confidence_score": 1.0,
        })
        report = aggregator.aggregate(user_id="P001", date="2026-07-15")

        val = get_validator()
        ok, errors = val.validate_daily_metrics(report)
        assert ok, f"Schema validation failed: {errors}"

    def test_repair_missing_fields(self, aggregator):
        """缺失字段应被自动修复。"""
        aggregator.add_window({
            "window_start_sec": 0, "window_end_sec": 60,
            "valid_duration_sec": 60, "coverage_ratio": 0.5,
            "active_ratio": 0.3, "sedentary_ratio": 0.7,
            "room_transitions": 0, "movement_velocity_px_s": 0,
            "night_activity_frames": 0, "multi_person_sec": 0,
            "person_count_mean": 0, "confidence_score": 0.3,
        })
        report = aggregator.aggregate(user_id="P001", date="2026-07-15")
        # 所有字段应存在
        required = [
            "active_minutes", "sedentary_ratio", "room_transition_count",
            "night_activity_count", "social_interaction_minutes",
            "repetitive_path_count", "movement_speed",
            "coverage_minutes", "feature_confidence",
        ]
        for field in required:
            assert field in report["daily_metrics"]

    def test_reset(self, aggregator):
        aggregator.add_window({"window_start_sec": 0, "window_end_sec": 60})
        aggregator.reset()
        assert aggregator.window_count == 0

    def test_repr(self, aggregator):
        r = repr(aggregator)
        assert "DailyAggregator" in r
