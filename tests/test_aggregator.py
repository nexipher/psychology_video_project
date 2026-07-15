"""Tests for 1.4 — FeatureAggregator (A1 daily/hourly aggregation)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.video_analysis.aggregator import (
    DailyAggregation,
    FeatureAggregator,
    HourlyAggregation,
    SequenceReport,
    batch_process_sequences,
)
from src.video_analysis.feature_extractor import BasicFeatures, FeatureWindow
from src.video_analysis.config import NUM_JOINTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(
    start_frame: int = 0,
    num_frames: int = 30,
    duration_s: float = 2.0,
    activity_minutes: float = 1.0,
    sedentary_ratio: float = 0.5,
    room_transitions: int = 2,
    average_velocity: float = 0.3,
    night_activity_count: int = 0,
    night_activity_duration_seconds: float = 0.0,
    multi_person_duration_seconds: float = 10.0,
) -> FeatureWindow:
    return FeatureWindow(
        window_id=f"w_test_{start_frame:04d}",
        start_frame=start_frame,
        end_frame=start_frame + num_frames - 1,
        duration_s=duration_s,
        num_frames=num_frames,
        basic_features=BasicFeatures(
            activity_minutes=activity_minutes,
            sedentary_ratio=sedentary_ratio,
            room_transitions=room_transitions,
            average_velocity=average_velocity,
            night_activity_count=night_activity_count,
            night_activity_duration_seconds=night_activity_duration_seconds,
            multi_person_duration_seconds=multi_person_duration_seconds,
        ),
    )


def _build_skeleton_zip(tmp_path: Path, n_frames: int = 100) -> Path:
    """Minimal skeleton zip for integration tests."""
    zip_path = tmp_path / "test_skel.zip"
    json_dir = tmp_path / "skel"
    json_dir.mkdir()

    frames = []
    for fi in range(n_frames):
        x = 0.3 * np.sin(fi * 0.2) + np.random.randn() * 0.01
        y = fi * 0.005 + np.random.randn() * 0.005
        pose3d = [0.0] * (NUM_JOINTS * 3)
        pose3d[0] = x
        pose3d[1] = y
        pose3d[2] = 0.0
        pose2d = [0.0] * (NUM_JOINTS * 2)
        pose2d[0] = x * 100 + 320
        pose2d[1] = y * 100 + 240
        frames.append([{"pose2d": pose2d, "pose3d": pose3d}])

    raw = {"njts": NUM_JOINTS, "K": 1, "frames": frames}
    (json_dir / "Test_Walk_p01_pose3d.json").write_text(json.dumps(raw))

    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(json_dir.glob("*.json")):
            zf.write(f, f.name)
    return zip_path


# ---------------------------------------------------------------------------
# HourlyAggregation
# ---------------------------------------------------------------------------


class TestHourlyAggregation:
    def test_defaults(self) -> None:
        ha = HourlyAggregation(hour=8)
        assert ha.hour == 8
        assert ha.num_windows == 0
        assert ha.activity_minutes == 0.0
        assert ha.sedentary_ratio == 0.0
        assert ha.room_transitions == 0

    def test_to_dict(self) -> None:
        ha = HourlyAggregation(
            hour=14,
            num_windows=10,
            activity_minutes=25.5,
            sedentary_ratio=0.68,
            room_transitions=5,
            average_velocity=0.42,
            night_activity_count=0,
            night_activity_duration_seconds=0,
            multi_person_duration_seconds=120.0,
        )
        d = ha.to_dict()
        assert d["hour"] == 14
        assert d["activity_minutes"] == 25.5
        assert d["sedentary_ratio"] == 0.68
        json.dumps(d)  # serializable


# ---------------------------------------------------------------------------
# DailyAggregation
# ---------------------------------------------------------------------------


class TestDailyAggregation:
    def test_empty(self) -> None:
        da = DailyAggregation()
        assert da.date == ""
        assert da.total_windows == 0
        assert da.hourly_breakdown == []

    def test_to_dict(self) -> None:
        ha = HourlyAggregation(hour=10, num_windows=3, activity_minutes=5.0)
        da = DailyAggregation(
            date="2026-07-14",
            basic_features=BasicFeatures(activity_minutes=15.0, sedentary_ratio=0.5),
            hourly_breakdown=[ha],
            total_windows=3,
            total_duration_s=6.0,
            monitoring_quality={"quality_confidence": 0.95},
        )
        d = da.to_dict()
        assert d["date"] == "2026-07-14"
        assert "basic_features" in d
        assert len(d["hourly_breakdown"]) == 1
        json.dumps(d)


# ---------------------------------------------------------------------------
# SequenceReport
# ---------------------------------------------------------------------------


class TestSequenceReport:
    def test_to_dict_matches_schema(self) -> None:
        sr = SequenceReport(
            user_id="ELDER_TEST",
            device_id="CAM_TEST",
            sequence_name="Walk_p01",
            time_window={
                "start_time": "2026-07-14T08:00:00Z",
                "end_time": "2026-07-14T09:00:00Z",
            },
            monitoring_quality={"quality_confidence": 0.92},
            basic_features=BasicFeatures(activity_minutes=10.0).to_dict(),
            hourly_breakdown=[],
        )
        d = sr.to_dict()
        assert "user_id" in d
        assert "device_id" in d
        assert "sequence_name" in d
        assert "time_window" in d
        assert "monitoring_quality" in d
        assert "basic_features" in d
        assert "hourly_breakdown" in d
        json.dumps(d)


# ---------------------------------------------------------------------------
# FeatureAggregator
# ---------------------------------------------------------------------------


class TestFeatureAggregator:
    def test_ingest_single_window(self) -> None:
        agg = FeatureAggregator(fps=15.0, video_start_hour=8.0)
        w = _make_window(start_frame=0)
        agg.ingest(w)
        daily = agg.flush_daily()
        assert daily.total_windows == 1
        assert daily.basic_features.activity_minutes == 1.0

    def test_hour_binning(self) -> None:
        """Two windows at different hours should land in different bins."""
        agg = FeatureAggregator(fps=15.0, video_start_hour=8.0)
        # Frame 0 at 8:00 → hour 8
        w1 = _make_window(start_frame=0, activity_minutes=1.0)
        # Frame 54000 at 8:00 + 54000/(15*3600) = 8:00 + 1h = 9:00 → hour 9
        w2 = _make_window(start_frame=54000, activity_minutes=2.0)
        agg.ingest(w1)
        agg.ingest(w2)
        daily = agg.flush_daily()
        hours = {h.hour: h for h in daily.hourly_breakdown}
        assert hours[8].activity_minutes == 1.0
        assert hours[9].activity_minutes == 2.0

    def test_flush_daily_sums(self) -> None:
        """Daily total should be the sum across all ingested windows."""
        agg = FeatureAggregator(fps=15.0, video_start_hour=12.0)
        for _ in range(5):
            agg.ingest(_make_window(
                start_frame=0,
                activity_minutes=2.0,
                room_transitions=3,
                multi_person_duration_seconds=30.0,
            ))
        daily = agg.flush_daily()
        assert daily.basic_features.activity_minutes == pytest.approx(10.0)
        assert daily.basic_features.room_transitions == 15
        assert daily.basic_features.multi_person_duration_seconds == pytest.approx(150.0)

    def test_weighted_averages(self) -> None:
        """Sedentary ratio and velocity should be weighted by window duration."""
        agg = FeatureAggregator(fps=15.0, video_start_hour=8.0)
        # Window with duration 2s, sed=0.8
        w1 = _make_window(start_frame=0, duration_s=2.0, sedentary_ratio=0.8,
                          average_velocity=0.1)
        # Window with duration 8s, sed=0.2
        w2 = _make_window(start_frame=0, duration_s=8.0, sedentary_ratio=0.2,
                          average_velocity=0.5)
        agg.ingest(w1)
        agg.ingest(w2)
        daily = agg.flush_daily()
        # Weighted sed = (0.8*2 + 0.2*8) / (2+8) = (1.6 + 1.6) / 10 = 0.32
        assert daily.basic_features.sedentary_ratio == pytest.approx(0.32)
        # Weighted vel = (0.1*2 + 0.5*8) / 10 = (0.2 + 4.0) / 10 = 0.42
        assert daily.basic_features.average_velocity == pytest.approx(0.42)

    def test_night_activity_aggregation(self) -> None:
        """Night activity should accumulate for night-hour windows."""
        agg = FeatureAggregator(fps=15.0, video_start_hour=23.0)  # start at 23:00
        # Frame 0 at 23:00 → night
        w = _make_window(
            start_frame=0,
            night_activity_count=2,
            night_activity_duration_seconds=60.0,
        )
        agg.ingest(w)
        daily = agg.flush_daily()
        assert daily.basic_features.night_activity_count == 2
        assert daily.basic_features.night_activity_duration_seconds == 60.0

    def test_hourly_breakdown_populated_only(self) -> None:
        """flush_daily should only return hours that have windows."""
        agg = FeatureAggregator(fps=15.0, video_start_hour=10.0)
        agg.ingest(_make_window(start_frame=0))
        daily = agg.flush_daily()
        assert len(daily.hourly_breakdown) == 1
        assert daily.hourly_breakdown[0].hour == 10

    def test_reset(self) -> None:
        agg = FeatureAggregator(fps=15.0, video_start_hour=8.0)
        agg.ingest(_make_window(start_frame=0))
        assert agg._total_windows == 1
        agg.reset()
        assert agg._total_windows == 0
        daily = agg.flush_daily()
        assert daily.total_windows == 0
        assert daily.basic_features.activity_minutes == 0.0

    def test_empty_flush(self) -> None:
        """Flushing without any ingest should return zeros, not crash."""
        agg = FeatureAggregator()
        daily = agg.flush_daily()
        assert daily.total_windows == 0
        assert daily.basic_features.activity_minutes == 0.0

    def test_flush_sequence_report(self) -> None:
        agg = FeatureAggregator(
            fps=15.0, video_start_hour=8.0,
            user_id="ELDER_01", device_id="CAM_01",
        )
        agg.ingest(_make_window(start_frame=0, activity_minutes=5.0))
        report = agg.flush_sequence_report(
            sequence_name="test_seq.json",
            date="2026-07-14",
        )
        assert report.user_id == "ELDER_01"
        assert report.device_id == "CAM_01"
        assert report.sequence_name == "test_seq.json"
        assert report.basic_features["activity_minutes"] == 5.0
        assert "start_time" in report.time_window
        json.dumps(report.to_dict())

    def test_flush_sequence_report_override_ids(self) -> None:
        agg = FeatureAggregator(user_id="DEFAULT")
        agg.ingest(_make_window(start_frame=0))
        report = agg.flush_sequence_report(user_id="OVERRIDE")
        assert report.user_id == "OVERRIDE"

    def test_monitoring_quality_aggregation(self) -> None:
        agg = FeatureAggregator()
        w = FeatureWindow(
            window_id="w",
            start_frame=0, end_frame=29,
            duration_s=2.0, num_frames=30,
            basic_features=BasicFeatures(),
            monitoring_quality={"quality_confidence": 0.8, "missing_frames": 2},
        )
        agg.ingest(w)
        daily = agg.flush_daily()
        assert "quality_confidence" in daily.monitoring_quality
        assert daily.monitoring_quality["quality_confidence"] == pytest.approx(0.8)

    def test_ingest_all_chain(self) -> None:
        """ingest_all should accept an iterator and return self."""
        windows = [_make_window(start_frame=0) for _ in range(3)]
        agg = FeatureAggregator().ingest_all(iter(windows))
        daily = agg.flush_daily()
        assert daily.total_windows == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_midnight_wraparound(self) -> None:
        """Window starting at 23:30 should map to hour 23, not 0."""
        # video_start_hour=23.5 = 23:30
        agg = FeatureAggregator(fps=15.0, video_start_hour=23.5)
        w = _make_window(start_frame=0)
        agg.ingest(w)
        daily = agg.flush_daily()
        assert daily.hourly_breakdown[0].hour == 23

    def test_hour_rollover(self) -> None:
        """After 24+ hours, hour should wrap correctly."""
        # video_start_hour=0, frame at 25 hours → hour 1
        frames_for_25h = int(25 * 3600 * 15)  # = 1,350,000
        agg = FeatureAggregator(fps=15.0, video_start_hour=0.0)
        w = _make_window(start_frame=frames_for_25h)
        agg.ingest(w)
        daily = agg.flush_daily()
        assert daily.hourly_breakdown[0].hour == 1

    def test_zero_fps_fallback(self) -> None:
        """With fps=0, _window_to_hour should not divide by zero."""
        agg = FeatureAggregator(fps=0.0, video_start_hour=12.0)
        w = _make_window(start_frame=9999)
        agg.ingest(w)
        daily = agg.flush_daily()
        assert daily.total_windows == 1
        # All windows fall into hour 12 (start_hour)
        assert daily.hourly_breakdown[0].hour == 12

    def test_multiple_hours_in_one_day(self) -> None:
        """Across a full day, all 24-hour bins should be addressable."""
        agg = FeatureAggregator(fps=15.0, video_start_hour=0.0)
        # One window per hour for 12 hours
        for h in range(12):
            start_frame = int(h * 3600 * 15)
            agg.ingest(_make_window(start_frame=start_frame,
                                    activity_minutes=float(h)))
        daily = agg.flush_daily()
        assert len(daily.hourly_breakdown) == 12
        hours_seen = {h.hour for h in daily.hourly_breakdown}
        assert hours_seen == set(range(12))
        # Daily total = sum(0..11) = 66
        assert daily.basic_features.activity_minutes == pytest.approx(66.0)

    def test_sequence_report_includes_all_schema_fields(self) -> None:
        """Verify every required field from the project JSON Schema is present."""
        agg = FeatureAggregator(
            fps=15.0, video_start_hour=8.0,
            user_id="ELDER_603211", device_id="CAMERA_LIVING_01",
        )
        for _ in range(5):
            agg.ingest(_make_window(start_frame=0))
        report = agg.flush_sequence_report(
            sequence_name="test.json", date="2026-07-14",
        )
        d = report.to_dict()

        required_toplevel = [
            "user_id", "device_id", "sequence_name",
            "time_window", "monitoring_quality", "basic_features",
            "hourly_breakdown",
        ]
        for key in required_toplevel:
            assert key in d, f"Missing key: {key}"

        required_basic = [
            "activity_minutes", "sedentary_ratio", "room_transitions",
            "average_velocity", "night_activity_count",
            "night_activity_duration_seconds", "multi_person_duration_seconds",
        ]
        for key in required_basic:
            assert key in d["basic_features"], f"Missing basic_feature: {key}"

        assert "start_time" in d["time_window"]
        assert "end_time" in d["time_window"]


# ---------------------------------------------------------------------------
# Integration — batch_process_sequences
# ---------------------------------------------------------------------------


class TestBatchProcessSequences:
    def test_batch_with_synthetic_data(self, tmp_path: Path) -> None:
        zip_path = _build_skeleton_zip(tmp_path, n_frames=90)
        reports = list(batch_process_sequences(
            str(zip_path),
            user_id="U1", device_id="D1",
            window_size=30, stride=30, fps=15.0,
            video_start_hour=8.0,
        ))
        assert len(reports) == 1  # one file
        report = reports[0]
        assert report.user_id == "U1"
        assert report.device_id == "D1"
        assert "Test_Walk_p01_pose3d" in report.sequence_name
        assert report.basic_features["activity_minutes"] >= 0
        assert 0.0 <= report.basic_features["sedentary_ratio"] <= 1.0
        json.dumps(report.to_dict())

    def test_batch_with_real_data(self) -> None:
        """Smoke test with one real skeleton sequence."""
        reports = list(batch_process_sequences(
            "dataset/toyota_smarthome_skeleton_v1.2.zip",
            user_id="ELDER_REAL", device_id="CAM_REAL",
            window_size=30, stride=15, fps=15.0,
            max_sequences=1,
        ))
        assert len(reports) == 1
        r = reports[0]
        assert r.user_id == "ELDER_REAL"
        assert r.basic_features["activity_minutes"] >= 0
        assert 0.0 <= r.basic_features["sedentary_ratio"] <= 1.0
        json.dumps(r.to_dict())
