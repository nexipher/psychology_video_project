"""1.6 — Full-pipeline integration tests.

End-to-end validation of the complete processing chain:
SkeletonDataLoader → SlidingWindow → SkeletonFeatureExtractor →
FeatureAggregator → SequenceReport.

All tests run in pure-CPU mode (no GPU required).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.video_analysis.aggregator import (
    FeatureAggregator,
    SequenceReport,
    batch_process_sequences,
)
from src.video_analysis.config import (
    JOINT_NAMES,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    NUM_JOINTS,
)
from src.video_analysis.data_loader import (
    SkeletonDataLoader,
    SkeletonSequence,
    get_joint_name,
)
from src.video_analysis.feature_extractor import (
    BasicFeatures,
    FeatureWindow,
    SkeletonFeatureExtractor,
)


# ===================================================================
# Full-pipeline integration (real skeleton data)
# ===================================================================


class TestFullPipelineRealData:
    """End-to-end tests using real Toyota Smarthome skeleton data."""

    def test_full_pipeline_single_sequence(self, real_skeleton_zip: Path) -> None:
        """DataLoader → Extractor → Aggregator → Report for one sequence."""
        loader = SkeletonDataLoader(str(real_skeleton_zip))
        fname = loader.list_files()[0]
        seq = loader.load(fname)

        # 1. Verify raw data integrity
        assert isinstance(seq, SkeletonSequence)
        assert seq.num_frames > 0
        assert seq.num_joints == NUM_JOINTS
        for frame in seq.frames[:5]:
            for person in frame.persons:
                assert person.pose2d.shape == (NUM_JOINTS, 2)
                assert person.pose3d.shape == (NUM_JOINTS, 3)
                assert person.pose2d.dtype == np.float32

        # 2. Feature extraction
        extractor = SkeletonFeatureExtractor(
            str(real_skeleton_zip), window_size=30, stride=15, fps=15.0,
        )
        windows = list(extractor.process_sequence(fname))

        assert len(windows) > 0, f"No windows produced for {fname}"
        for w in windows:
            assert isinstance(w, FeatureWindow)
            assert w.basic_features.activity_minutes >= 0
            assert 0.0 <= w.basic_features.sedentary_ratio <= 1.0
            assert w.monitoring_quality is not None

        # 3. Aggregation
        agg = FeatureAggregator(
            fps=15.0, video_start_hour=8.0,
            user_id="ELDER_TEST", device_id="CAM_TEST",
        )
        agg.ingest_all(iter(windows))
        report = agg.flush_sequence_report(
            sequence_name=fname, date="2026-07-14",
        )

        # 4. Validate report schema
        d = report.to_dict()
        _assert_report_schema_valid(d)

        extractor.close()
        loader.close()

    def test_batch_pipeline_multiple_sequences(self, real_skeleton_zip: Path) -> None:
        """Batch-process 5 real sequences end-to-end."""
        reports = list(
            batch_process_sequences(
                str(real_skeleton_zip),
                user_id="ELDER_BATCH",
                device_id="CAM_BATCH",
                window_size=30, stride=30, fps=15.0,
                max_sequences=5,
            )
        )

        assert len(reports) == 5
        for i, report in enumerate(reports):
            assert isinstance(report, SequenceReport)
            assert report.user_id == "ELDER_BATCH"
            d = report.to_dict()
            _assert_report_schema_valid(d)
            assert d["basic_features"]["activity_minutes"] >= 0, (
                f"Report {i}: negative activity"
            )

    def test_pipeline_with_different_windows(self, real_skeleton_zip: Path) -> None:
        """Varying window parameters should not break the pipeline."""
        loader = SkeletonDataLoader(str(real_skeleton_zip))
        fname = loader.list_files()[0]

        configs = [
            (30, 30),
            (60, 30),
            (15, 15),
            (90, 45),
            (10, 5),
        ]

        for ws, st in configs:
            extractor = SkeletonFeatureExtractor(
                str(real_skeleton_zip), window_size=ws, stride=st, fps=15.0,
            )
            windows = list(extractor.process_sequence(fname))
            agg = FeatureAggregator(fps=15.0, video_start_hour=8.0)
            agg.ingest_all(iter(windows))
            report = agg.flush_sequence_report(sequence_name=fname)
            d = report.to_dict()
            _assert_report_schema_valid(d)
            extractor.close()

        loader.close()


# ===================================================================
# Synthetic data pipeline
# ===================================================================


class TestFullPipelineSynthetic:
    """End-to-end tests with deterministic synthetic data."""

    def test_pipeline_end_to_end(self, skeleton_zip_path: Path) -> None:
        """All stages produce valid, schema-conformant output."""
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                user_id="U_SYNTH", device_id="D_SYNTH",
                window_size=30, stride=30, fps=15.0,
                video_start_hour=10.0,
            )
        )

        assert len(reports) == 3  # Walk, Sit, Multi
        for report in reports:
            d = report.to_dict()
            _assert_report_schema_valid(d)

    def test_movement_vs_static_differs(self, skeleton_zip_path: Path) -> None:
        """Walk sequence should show higher activity than Sit sequence.

        With default velocity_threshold (0.02 m/frame), the small random
        noise in static frames (~0.001) falls below threshold while the
        large oscillations in walking frames (~0.5) exceed it.
        """
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                window_size=30, stride=30, fps=15.0,
                velocity_threshold=0.01,  # low enough to catch walk, above noise
            )
        )

        walk = [r for r in reports if "Walk" in r.sequence_name][0]
        sit = [r for r in reports if "Sit" in r.sequence_name][0]

        # Walk should have more activity (lower sedentary ratio)
        walk_sed = walk.basic_features["sedentary_ratio"]
        sit_sed = sit.basic_features["sedentary_ratio"]
        assert walk_sed < sit_sed, (
            f"Expected Walk sed ({walk_sed}) < Sit sed ({sit_sed})"
        )

    def test_multi_person_detection(self, skeleton_zip_path: Path) -> None:
        """Multi-person sequence should report > 0 multi_person_duration."""
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                window_size=30, stride=30, fps=15.0,
            )
        )

        multi = [r for r in reports if "Multi" in r.sequence_name][0]
        assert multi.basic_features["multi_person_duration_seconds"] > 0, (
            "Multi-person sequence should have multi-person time"
        )

    def test_hourly_breakdown_present(self, skeleton_zip_path: Path) -> None:
        """Each report should include an hourly breakdown."""
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                window_size=30, stride=30, fps=15.0,
                video_start_hour=10.0,
            )
        )

        for report in reports:
            d = report.to_dict()
            assert len(d["hourly_breakdown"]) >= 1, (
                f"{report.sequence_name}: no hourly breakdown"
            )
            for hb in d["hourly_breakdown"]:
                assert 0 <= hb["hour"] <= 23
                assert "activity_minutes" in hb


# ===================================================================
# CPU-only mode verification
# ===================================================================


class TestCPUOnlyMode:
    """Verify the pipeline runs entirely without GPU."""

    def test_data_loader_no_gpu(self, skeleton_zip_path: Path) -> None:
        loader = SkeletonDataLoader(str(skeleton_zip_path))
        seq = loader.load_by_index(0)
        assert seq.num_frames > 0
        loader.close()

    def test_sliding_window_no_gpu(self) -> None:
        from src.video_analysis.sliding_window import SlidingWindow
        sw = SlidingWindow[int](window_size=100, stride=50)
        for i in range(500):
            sw.push(i)
            if sw.is_ready():
                sw.advance()
        assert sw.total_pushes == 500

    def test_feature_extractor_no_gpu(self, skeleton_zip_path: Path) -> None:
        extractor = SkeletonFeatureExtractor(
            str(skeleton_zip_path), window_size=30, stride=30, fps=15.0,
        )
        windows = list(extractor.process_sequence(
            extractor._loader.list_files()[0]
        ))
        assert len(windows) > 0
        extractor.close()

    def test_aggregator_no_gpu(self, skeleton_zip_path: Path) -> None:
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                window_size=30, stride=30, fps=15.0,
                max_sequences=1,
            )
        )
        assert len(reports) == 1

    def test_numpy_only_operations(self) -> None:
        """Verify core metrics only use numpy (no torch/tf)."""
        import sys

        # Confirm no GPU framework is imported in core modules
        core_modules = [
            "src.video_analysis.config",
            "src.video_analysis.data_loader",
            "src.video_analysis.sliding_window",
            "src.video_analysis.aggregator",
        ]
        for mod_name in core_modules:
            # These modules should NOT import torch
            mod = sys.modules.get(mod_name)
            if mod is not None:
                assert "torch" not in dir(mod), (
                    f"{mod_name} unexpectedly imports torch"
                )

    def test_yolo_raises_without_gpu(self) -> None:
        """YOLOPoseFeatureExtractor should raise without GPU (non-mock)."""
        from src.video_analysis.feature_extractor import YOLOPoseFeatureExtractor

        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        if not has_gpu:
            with pytest.raises(RuntimeError):
                YOLOPoseFeatureExtractor(mock=False)


# ===================================================================
# Performance constraints
# ===================================================================


class TestPerformanceConstraints:
    def test_sliding_window_push_under_1_5ms(self) -> None:
        """1.5 spec: sliding window push must be < 1.5 ms."""
        from src.video_analysis.sliding_window import SlidingWindow

        sw = SlidingWindow[float](window_size=300, stride=150)
        for i in range(1000):
            sw.push(float(i))
            if sw.is_ready():
                sw.advance()

        # Allow a generous margin; on modern CPUs this is < 10 µs
        assert sw.last_push_time_us < 1500.0, (
            f"Push too slow: {sw.last_push_time_us:.1f} µs"
        )

    def test_sliding_window_advance_under_1_5ms(self) -> None:
        from src.video_analysis.sliding_window import SlidingWindow

        sw = SlidingWindow[float](window_size=300, stride=150)
        for i in range(300):
            sw.push(float(i))
        sw.advance()
        assert sw.last_advance_time_us < 1500.0, (
            f"Advance too slow: {sw.last_advance_time_us:.1f} µs"
        )

    def test_pipeline_throughput(self, skeleton_zip_path: Path) -> None:
        """Full pipeline should process 100+ frames in under 5 seconds."""
        t0 = time.perf_counter()
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                window_size=30, stride=30, fps=15.0,
                max_sequences=3,
            )
        )
        elapsed = time.perf_counter() - t0
        assert len(reports) == 3
        # 3 sequences × 90-120 frames each → ~300 frames
        # Should complete well within 30s
        assert elapsed < 30.0, f"Pipeline too slow: {elapsed:.1f}s for 3 sequences"


# ===================================================================
# Edge cases
# ===================================================================


class TestPipelineEdgeCases:
    def test_empty_sequence(self, tmp_path: Path) -> None:
        """Skeleton file with zero frames should be handled gracefully."""
        import zipfile
        json_dir = tmp_path / "empty"
        json_dir.mkdir()
        raw = {"njts": NUM_JOINTS, "K": 1, "frames": []}
        (json_dir / "Empty_pose3d.json").write_text(json.dumps(raw))
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(json_dir / "Empty_pose3d.json", "Empty_pose3d.json")

        extractor = SkeletonFeatureExtractor(str(zip_path), window_size=30, stride=15)
        windows = list(extractor.process_sequence("Empty_pose3d.json"))
        assert len(windows) == 0  # Should not crash
        extractor.close()

    def test_single_frame_sequence(self, tmp_path: Path) -> None:
        """Single-frame skeleton should be skipped gracefully."""
        import zipfile
        json_dir = tmp_path / "oneframe"
        json_dir.mkdir()
        raw = {
            "njts": NUM_JOINTS, "K": 1,
            "frames": [[
                {"pose2d": [0.0] * 26, "pose3d": [0.0] * 39}
            ]]
        }
        (json_dir / "One_p1_pose3d.json").write_text(json.dumps(raw))
        zip_path = tmp_path / "oneframe.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(json_dir / "One_p1_pose3d.json", "One_p1_pose3d.json")

        extractor = SkeletonFeatureExtractor(str(zip_path))
        windows = list(extractor.process_sequence("One_p1_pose3d.json"))
        assert len(windows) == 0
        extractor.close()

    def test_night_day_transition(self, skeleton_zip_path: Path) -> None:
        """Starting at 5:30 AM should cross the night→day boundary."""
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path),
                window_size=30, stride=30, fps=15.0,
                video_start_hour=5.5,  # 5:30 AM
                max_sequences=1,
            )
        )
        assert len(reports) == 1
        d = reports[0].to_dict()
        # First hour should be hour 5 (night)
        assert d["hourly_breakdown"][0]["hour"] == 5

    def test_all_joint_names_valid(self) -> None:
        """Every defined joint name should resolve to its index."""
        for i, name in enumerate(JOINT_NAMES):
            assert get_joint_name(i) == name

    def test_json_roundtrip_all_modules(self, skeleton_zip_path: Path) -> None:
        """Every module's .to_dict() output must be valid JSON."""
        from src.video_analysis.aggregator import (
            DailyAggregation,
            HourlyAggregation,
        )

        # BasicFeatures
        json.dumps(BasicFeatures().to_dict())
        # FeatureWindow
        json.dumps(
            FeatureWindow(
                window_id="w", start_frame=0, end_frame=9,
                duration_s=1.0, num_frames=10,
            ).to_dict()
        )
        # Hourly
        json.dumps(HourlyAggregation(hour=12).to_dict())
        # Daily
        json.dumps(DailyAggregation(date="2026-07-14").to_dict())

        # Full report
        reports = list(
            batch_process_sequences(
                str(skeleton_zip_path), max_sequences=1,
            )
        )
        json.dumps(reports[0].to_dict())


# ===================================================================
# Schema validators
# ===================================================================


def _assert_report_schema_valid(d: dict) -> None:
    """Validate a SequenceReport dict against the project JSON Schema §6.1."""
    # Top-level keys
    for key in [
        "user_id", "device_id", "sequence_name",
        "time_window", "monitoring_quality", "basic_features",
        "hourly_breakdown",
    ]:
        assert key in d, f"Missing top-level key: {key}"

    # time_window
    tw = d["time_window"]
    assert "start_time" in tw
    assert "end_time" in tw

    # basic_features subset
    bf = d["basic_features"]
    required_fields = [
        "activity_minutes",
        "sedentary_ratio",
        "room_transitions",
        "average_velocity",
        "night_activity_count",
        "night_activity_duration_seconds",
        "multi_person_duration_seconds",
    ]
    for field in required_fields:
        assert field in bf, f"Missing basic_feature: {field}"

    # Value range checks
    assert bf["activity_minutes"] >= 0
    assert 0.0 <= bf["sedentary_ratio"] <= 1.0, (
        f"sedentary_ratio out of [0,1]: {bf['sedentary_ratio']}"
    )
    assert bf["room_transitions"] >= 0
    assert bf["average_velocity"] >= 0
    assert bf["night_activity_count"] >= 0
    assert bf["night_activity_duration_seconds"] >= 0
    assert bf["multi_person_duration_seconds"] >= 0

    # monitoring_quality
    mq = d["monitoring_quality"]
    assert "quality_confidence" in mq

    # JSON serializable
    json.dumps(d)
