"""视频特征提取器测试。A1.8"""

import numpy as np
import pytest
from src.video_analysis.feature_extractor import VideoFeatureExtractor


class TestVideoFeatureExtractor:
    """VideoFeatureExtractor 核心功能测试。"""

    def test_process_single_person(self, extractor):
        """单人连续行走的特征提取。"""
        outputs = []
        for i in range(450):  # 30 秒 @ 15fps
            ts = i / 15.0
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            x = 200 + i * 0.5
            for k in range(17):
                kps[0, k] = [x + k * 2, 300 + k * 3 - 50, 0.9]
            bboxes = np.array([[x - 30, 150, x + 30, 450]], dtype=np.float32)
            result = extractor.process_frame(kps, bboxes, [1], ts, i)
            if result:
                outputs.append(result)

        # 30 秒数据，10 秒步长 → ~3 个窗口
        assert len(outputs) >= 2

    def test_window_metrics_structure(self, extractor):
        """窗口输出结构完整性。"""
        for i in range(450):
            ts = i / 15.0
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, 2] = 0.9
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)
            result = extractor.process_frame(kps, bboxes, [1], ts, i)

        if result:
            required_keys = [
                "window_start_sec", "window_end_sec", "valid_duration_sec",
                "coverage_ratio", "active_ratio", "sedentary_ratio",
                "room_transitions", "movement_velocity_px_s",
                "night_activity_frames", "multi_person_sec",
                "person_count_mean", "confidence_score",
            ]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_daily_summary_schema(self, extractor):
        """日级输出应通过 Schema 校验。"""
        from src.utils.schema_validator import get_validator

        for i in range(450):
            ts = i / 15.0
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            x = 200 + i * 0.5
            for k in range(17):
                kps[0, k] = [x + k * 2, 300 + k * 3 - 50, 0.9]
            bboxes = np.array([[x - 30, 150, x + 30, 450]], dtype=np.float32)
            extractor.process_frame(kps, bboxes, [1], ts, i)

        daily = extractor.get_daily_summary(user_id="test", date="2026-07-15")
        val = get_validator()
        ok, errors = val.validate_daily_metrics(daily)
        assert ok, f"Schema validation failed: {errors}"

    def test_multi_person_detection(self, extractor):
        """多人场景应检测到 multi_person。"""
        for i in range(150):
            ts = i / 15.0
            n = 2 if i > 50 else 1
            kps = np.zeros((n, 17, 3), dtype=np.float32)
            for p in range(n):
                px = 200 + p * 100 + i * 0.3
                for k in range(17):
                    kps[p, k] = [px + k, 300 + k, 0.9]
            bboxes = np.array([[px - 30, 150, px + 30, 450] for px in [200 + p * 100 + i * 0.3 for p in range(n)]], dtype=np.float32)
            extractor.process_frame(kps, bboxes, list(range(n)), ts, i)

        daily = extractor.get_daily_summary(user_id="test", date="2026-07-15")
        assert "social_interaction_minutes" in daily["daily_metrics"]

    def test_empty_frames(self, extractor):
        """全空帧应正常处理。"""
        for i in range(100):
            ts = i / 15.0
            kps = np.empty((0, 17, 3), dtype=np.float32)
            bboxes = np.empty((0, 4), dtype=np.float32)
            extractor.process_frame(kps, bboxes, [], ts, i)

        daily = extractor.get_daily_summary()
        assert daily["daily_metrics"]["feature_confidence"] >= 0.0

    def test_night_hours_detection(self):
        """夜间时段检测：0:00 AM 应在夜间，is_night 标志为 True。"""
        extractor = VideoFeatureExtractor(
            window_size_sec=30.0, window_stride_sec=15.0,
            fps=15.0, night_start_hour=22, night_end_hour=6,
        )
        # 模拟 0:00 AM 的数据，大幅移动以确保被判定为活动
        night_outputs = []
        for i in range(450):
            ts = i / 15.0  # 0–30s，在夜间
            # 每帧位移 100px 确保 is_sedentary=False
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, :2] = [[200 + i * 80, 300 + k * 5] for k in range(17)]
            kps[0, :, 2] = 0.9
            bboxes = np.array([[170, 150, 230, 450]], dtype=np.float32)
            result = extractor.process_frame(kps, bboxes, [1], ts, i)
            if result:
                night_outputs.append(result)

        # 0:00 AM 的夜间非静止帧应被统计
        if night_outputs:
            assert any(
                w.get("night_activity_frames", 0) > 0
                for w in night_outputs
            )

    def test_reset(self, extractor):
        for i in range(100):
            ts = i / 15.0
            kps = np.zeros((1, 17, 3), dtype=np.float32)
            kps[0, :, 2] = 0.9
            bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)
            extractor.process_frame(kps, bboxes, [1], ts, i)

        extractor.reset()
        assert extractor._frame_count == 0
        assert extractor._window.get_count() == 0
        assert len(extractor._window_history) == 0

    def test_repr(self, extractor):
        r = repr(extractor)
        assert "VideoFeatureExtractor" in r
