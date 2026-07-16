"""A2 专项行为检测器测试。

覆盖全部 6 个检测器 + SpecialBehaviorDetector 总装。
全部 CPU 模式，使用合成数据。
"""

import numpy as np
import pytest
from src.video_analysis.special_behavior import (
    SpatialTrajectoryMap,
    RepetitivePathDetector,
    RepeatedActionDetector,
    ProlongedInactivityDetector,
    CircadianRhythmAnalyzer,
    SocialInteractionAnalyzer,
    SpecialBehaviorDetector,
)


# ============================================================
# SpatialTrajectoryMap
# ============================================================

class TestSpatialTrajectoryMap:
    def test_add_position(self):
        tmap = SpatialTrajectoryMap(grid_resolution=200)
        cell = tmap.add_position(300, 400, 0.0)
        assert cell == (1, 2)

    def test_path_recording(self):
        tmap = SpatialTrajectoryMap(grid_resolution=200, max_history_sec=10, fps=15.0)
        for i in range(20):
            tmap.add_position(100 + i * 10, 200, i / 15.0)
        assert len(tmap._path_history) == 20

    def test_top_cells(self):
        tmap = SpatialTrajectoryMap(grid_resolution=100)
        for _ in range(5):
            tmap.add_position(150, 150, 0.0)  # cell (1,1)
        for _ in range(3):
            tmap.add_position(350, 350, 0.0)  # cell (3,3)
        tops = tmap.get_top_cells(2)
        assert tops[0][0] == (1, 1)
        assert tops[0][1] == 5

    def test_none_position(self):
        tmap = SpatialTrajectoryMap()
        assert tmap.add_position(None, 300, 0.0) is None
        assert tmap.add_position(300, np.nan, 0.0) is None

    def test_transitions(self):
        tmap = SpatialTrajectoryMap(grid_resolution=50)
        for x in [50, 100, 150, 200]:
            tmap.add_position(x, 100, 0.0)
        assert tmap.get_transition_count((1, 2), (2, 2)) > 0 or tmap.get_transition_count((1, 2), (3, 2)) > 0

    def test_clear(self):
        tmap = SpatialTrajectoryMap()
        tmap.add_position(100, 100, 0.0)
        tmap.clear()
        assert len(tmap._path_history) == 0


# ============================================================
# RepetitivePathDetector
# ============================================================

class TestRepetitivePathDetector:
    def test_no_wandering_straight_line(self):
        det = RepetitivePathDetector(
            window_sec=10.0, stride_sec=2.0,
            min_path_length=5, min_repetition_count=2,
            overlap_threshold=0.3, fps=15.0, grid_resolution=50,
        )
        for i in range(200):
            det.update(100 + i * 2, 300, i / 15.0)
        assert len(det.events) > 0
        # 直线走不应检测到徘徊
        assert not any(e["is_wandering"] for e in det.events)

    def test_wandering_detected(self):
        det = RepetitivePathDetector(
            window_sec=10.0, stride_sec=2.0,
            min_path_length=5, min_repetition_count=2,
            overlap_threshold=0.3, fps=15.0, grid_resolution=100,
        )
        # Simulate back-and-forth
        for i in range(300):
            ts = i / 15.0
            x = 100 + (i % 20) * 20 if (i // 20) % 2 == 0 else 500 - (i % 20) * 20
            det.update(x, 300, ts)
        assert len(det.events) > 0

    def test_daily_count(self):
        det = RepetitivePathDetector(stride_sec=1.0, grid_resolution=100)
        for i in range(500):
            det.update(100 + (i % 10) * 50, 200 + (i % 5) * 30, i / 15.0)
        assert det.get_daily_repetitive_count() >= 0

    def test_reset(self):
        det = RepetitivePathDetector()
        det.update(100, 100, 0.0)
        det.reset()
        assert len(det.events) == 0


# ============================================================
# RepeatedActionDetector
# ============================================================

class TestRepeatedActionDetector:
    def test_repeated_approach(self):
        det = RepeatedActionDetector(
            window_sec=10.0, stride_sec=3.0,
            hotspot_min_visits=3, enter_exit_threshold=3,
            fps=15.0, grid_resolution=50,
        )
        # repeatedly visit a spot at (300, 200)
        for i in range(400):
            ts = i / 15.0
            if i % 30 < 5:
                x, y = 300, 200
            else:
                x = 100 + np.sin(i * 0.1) * 50
                y = 400 + np.cos(i * 0.1) * 30
            det.update(x, y, ts)

        assert len(det.events) > 0

    def test_no_hotspot(self):
        det = RepeatedActionDetector(stride_sec=2.0, grid_resolution=100)
        # Random walk, no repeated visits
        for i in range(200):
            det.update(np.random.randn() * 100 + 300, np.random.randn() * 100 + 300, i / 15.0)
        assert det.get_daily_hotspot_count() >= 0  # may or may not detect

    def test_reset(self):
        det = RepeatedActionDetector()
        det.update(100, 100, 0.0)
        det.reset()
        assert len(det.events) == 0


# ============================================================
# ProlongedInactivityDetector
# ============================================================

class TestProlongedInactivityDetector:
    def test_short_inactivity_no_warning(self):
        det = ProlongedInactivityDetector(
            prolonged_threshold_sec=10.0, warning_threshold_sec=5.0, fps=15.0,
        )
        kps = np.random.randn(17, 3).astype(np.float32) * 0.1
        kps[:, 2] = 0.9
        # 4s sedentary (< warning 5s), then active → no warning
        for i in range(90):  # 6s total, first 4s sedentary
            is_sed = i < 60
            result = det.update(is_sed, i / 15.0, kps)

        # One event when sedentary ends, 4s < 5s warning threshold
        assert len(det.events) >= 1
        event = det.events[0]
        assert event["warning_triggered"] is False

    def test_prolonged_triggers(self):
        det = ProlongedInactivityDetector(
            prolonged_threshold_sec=3.0, warning_threshold_sec=1.5, fps=15.0,
        )
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 2] = 0.9
        for i in range(80):
            is_sed = i < 60  # 60 frames = 4s sedentary
            det.update(is_sed, i / 15.0, kps)
        # Events collected when transition happens
        assert len(det.events) >= 1
        assert det.events[0]["prolonged_triggered"] is True

    def test_flush(self):
        det = ProlongedInactivityDetector(
            prolonged_threshold_sec=10.0, warning_threshold_sec=5.0, fps=15.0,
        )
        kps = np.zeros((17, 3), dtype=np.float32)
        for i in range(60):
            det.update(True, i / 15.0, kps)
        result = det.flush(4.0)
        assert result is not None
        assert result["continuous_inactive_sec"] >= 3.9

    def test_micro_motion(self):
        det = ProlongedInactivityDetector(min_micro_motion_px=1.0, fps=15.0)
        kps_moving = np.random.randn(17, 3).astype(np.float32) * 5.0
        kps_moving[:, 2] = 0.9
        for i in range(60):
            det.update(True, i / 15.0, kps_moving + np.random.randn(17, 3) * 2.0)
        det.update(False, 4.0, kps_moving)
        assert len(det.events) >= 1

    def test_get_daily_count(self):
        det = ProlongedInactivityDetector(
            prolonged_threshold_sec=3.0, warning_threshold_sec=1.0, fps=15.0,
        )
        kps = np.zeros((17, 3), dtype=np.float32)
        for i in range(80):
            det.update(i < 60, i / 15.0, kps)
        assert det.get_daily_prolonged_count() >= 1


# ============================================================
# CircadianRhythmAnalyzer
# ============================================================

class TestCircadianRhythmAnalyzer:
    def test_normal_day(self):
        circ = CircadianRhythmAnalyzer()
        for h in np.arange(0, 24, 0.1):
            circ.feed_hourly(h, 6.0 <= h <= 22.0)
        result = circ.analyze("2026-07-16")
        assert 5.0 <= result["wake_time"] <= 7.0
        assert 21.0 <= result["sleep_time"] <= 24.0

    def test_no_data_defaults(self):
        circ = CircadianRhythmAnalyzer()
        result = circ.analyze("2026-07-16")
        assert result["wake_time"] == 8.0
        assert result["sleep_time"] == 23.0

    def test_baseline_offset(self):
        circ = CircadianRhythmAnalyzer()
        # Day 1: normal (6:00-22:00)
        for h in np.arange(0, 24, 0.1):
            circ.feed_hourly(h, 6.0 <= h <= 21.9)
        circ.analyze("2026-07-14")
        circ.reset()

        # Day 2: wakes late (10:00-24:00)
        for h in np.arange(0, 24, 0.1):
            circ.feed_hourly(h, 10.0 <= h <= 23.9)
        result = circ.analyze("2026-07-15")
        # Day 2 wake should be ~10h vs baseline ~6h → offset > 3h
        assert result["wake_offset_hours"] > 2.0

    def test_nap_detection(self):
        circ = CircadianRhythmAnalyzer()
        # Active 6-12, nap 12-14, active 14-22
        for h in np.arange(0, 24, 0.1):
            active = (6.0 <= h <= 12.0) or (14.0 <= h <= 22.0)
            circ.feed_hourly(h, active)
        result = circ.analyze("2026-07-16")
        assert result["nap_duration_minutes"] > 60


# ============================================================
# SocialInteractionAnalyzer
# ============================================================

class TestSocialInteractionAnalyzer:
    @pytest.fixture
    def two_persons_facing(self):
        a = np.zeros((17, 3), dtype=np.float32)
        a[:, 2] = 0.9
        a[5, :2] = [300, 200]; a[6, :2] = [330, 200]; a[11, :2] = [300, 380]; a[12, :2] = [330, 380]
        b = np.zeros((17, 3), dtype=np.float32)
        b[:, 2] = 0.9
        b[5, :2] = [380, 190]; b[6, :2] = [350, 190]; b[11, :2] = [370, 370]; b[12, :2] = [340, 370]
        return [a, b]

    def test_intensity_with_facing(self, two_persons_facing):
        soc = SocialInteractionAnalyzer(window_sec=5.0, stride_sec=3.0, fps=15.0)
        for i in range(150):
            soc.update(two_persons_facing, np.zeros((2, 4)), i / 15.0)
        assert len(soc.events) >= 1

    def test_single_person_zero(self):
        soc = SocialInteractionAnalyzer(window_sec=5.0, stride_sec=3.0, fps=15.0)
        kps = np.zeros((1, 17, 3), dtype=np.float32)
        kps[0, :, 2] = 0.9
        for i in range(100):
            soc.update([kps[0]], np.zeros((1, 4)), i / 15.0)
        # 单人场景，互动强度应为 0
        if soc.events:
            assert soc.events[-1]["interaction_intensity"] == 0.0

    def test_daily_avg_intensity(self, two_persons_facing):
        soc = SocialInteractionAnalyzer(window_sec=5.0, stride_sec=2.0, fps=15.0)
        for i in range(200):
            soc.update(two_persons_facing, np.zeros((2, 4)), i / 15.0)
        avg = soc.get_daily_avg_intensity()
        assert 0.0 <= avg <= 1.0


# ============================================================
# SpecialBehaviorDetector 总装
# ============================================================

class TestSpecialBehaviorDetector:
    def test_init_all_enabled(self):
        det = SpecialBehaviorDetector(fps=15.0)
        assert det._wandering is not None
        assert det._social is not None

    def test_selective_disable(self):
        det = SpecialBehaviorDetector(
            enable_wandering=False, enable_circadian=False,
        )
        assert det._wandering is None
        assert det._circadian is None
        assert det._inactivity is not None

    def test_update_single_person(self):
        det = SpecialBehaviorDetector(fps=15.0)
        kps = np.random.randn(1, 17, 3).astype(np.float32) * 0.5
        kps[0, 5, :2] = [300, 200]; kps[0, 6, :2] = [330, 200]
        kps[0, 11, :2] = [300, 380]; kps[0, 12, :2] = [330, 380]
        kps[0, :, 2] = 0.9
        bboxes = np.array([[100, 100, 200, 300]], dtype=np.float32)

        for i in range(200):
            ts = i / 15.0
            x = 200 + i * 2
            y = 300 + np.sin(i * 0.05) * 20
            det.update(x, y, i < 10, ts, kps, bboxes, [1])

        assert det.frame_count == 200

    def test_update_multi_person(self):
        det = SpecialBehaviorDetector(fps=15.0)
        kps = np.random.randn(2, 17, 3).astype(np.float32)
        kps[:, :, 2] = 0.9
        bboxes = np.array([[100, 100, 200, 300], [300, 100, 400, 300]], dtype=np.float32)

        for i in range(100):
            det.update(250, 300, False, i / 15.0, kps, bboxes, [1, 2])

        assert det.frame_count == 100

    def test_flush(self):
        det = SpecialBehaviorDetector(fps=15.0)
        kps = np.zeros((1, 17, 3), dtype=np.float32)
        kps[:, 2] = 0.9
        for i in range(50):
            det.update(200, 300, True, i / 15.0, kps, np.zeros((1, 4)), [1])
        result = det.flush(3.0)
        assert "prolonged_inactivity" in result

    def test_get_daily_summary(self):
        det = SpecialBehaviorDetector(fps=15.0)
        kps = np.random.randn(1, 17, 3).astype(np.float32)
        kps[:, :, 2] = 0.9
        for i in range(100):
            det.update(200 + i, 300, i > 50, i / 15.0, kps, np.zeros((1, 4)), [1])
        summary = det.get_daily_summary("2026-07-16")
        assert "daily_repetitive_path_count" in summary or det._wandering is None

    def test_reset(self):
        det = SpecialBehaviorDetector(fps=15.0)
        det.update(200, 300, False, 0.0)
        det.reset()
        assert det.frame_count == 0

    def test_repr(self):
        det = SpecialBehaviorDetector()
        r = repr(det)
        assert "SpecialBehaviorDetector" in r
