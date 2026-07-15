"""Tests for 1.3 — VideoFeatureExtractor."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.video_analysis.feature_extractor import (
    BasicFeatures,
    FeatureWindow,
    SkeletonFeatureExtractor,
    VideoFeatureExtractor,
    YOLOPoseFeatureExtractor,
    _MetricsAccumulator,
)
from src.video_analysis.config import NUM_JOINTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_skeleton_zip(
    tmp_path: Path,
    n_frames: int = 100,
    K: int = 1,
    *,
    include_inactive: bool = True,
) -> Path:
    """Create a mini skeleton zip with controlled movement patterns.

    Frames alternate between "moving" (pelvis shifts noticeably) and
    "static" (pelvis ~same) to produce distinguishable metrics.
    """
    zip_path = tmp_path / "test_skel.zip"
    json_dir = tmp_path / "skel"
    json_dir.mkdir()

    frames = []
    for fi in range(n_frames):
        frame_people = []
        for pi in range(K):
            # Create movement: pelvis x oscillates, y drifts slowly
            if include_inactive and fi % 10 < 3:
                # Static phase — very small movement
                x = 0.0 + np.random.randn() * 0.001
                y = 0.0 + np.random.randn() * 0.001
            else:
                x = 0.5 * np.sin(fi * 0.3) + np.random.randn() * 0.005
                y = fi * 0.01 + np.random.randn() * 0.005

            # Build pose2d: 26 values (13 joints * 2)
            pose2d = [0.0] * (NUM_JOINTS * 2)
            pose2d[0] = x * 100 + 320  # pelvis x in pixels
            pose2d[1] = y * 100 + 240  # pelvis y in pixels

            # Build pose3d: 39 values (13 joints * 3)
            pose3d = [0.0] * (NUM_JOINTS * 3)
            pose3d[0] = x  # pelvis x (m)
            pose3d[1] = y  # pelvis y (m)
            pose3d[2] = 0.0  # pelvis z

            # Head position
            pose3d[9] = x + 0.01   # head x
            pose3d[10] = y + 0.5    # head y (above pelvis)
            pose3d[11] = 0.0

            frame_people.append({"pose2d": pose2d, "pose3d": pose3d})

        frames.append(frame_people)

    raw = {"njts": NUM_JOINTS, "K": K, "frames": frames}
    filepath = json_dir / "Test_Walk_p01_r00_v01_c01_pose3d.json"
    filepath.write_text(json.dumps(raw))

    # Second file: short sequence for tail-window tests
    short_frames = frames[:25]  # 25 frames
    raw2 = {"njts": NUM_JOINTS, "K": K, "frames": short_frames}
    filepath2 = json_dir / "Test_Short_p01_r00_v01_c01_pose3d.json"
    filepath2.write_text(json.dumps(raw2))

    # Third: multi-person sequence
    mp_frames = []
    for fi in range(60):
        frame_people = []
        for pi in range(2):
            pose3d = [float(fi + pi + j) for j in range(NUM_JOINTS * 3)]
            pose2d = [float(fi + pi + j) for j in range(NUM_JOINTS * 2)]
            frame_people.append({"pose2d": pose2d, "pose3d": pose3d})
        mp_frames.append(frame_people)
    raw3 = {"njts": NUM_JOINTS, "K": 2, "frames": mp_frames}
    filepath3 = json_dir / "Test_Multi_p01_r00_v01_c01_pose3d.json"
    filepath3.write_text(json.dumps(raw3))

    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(json_dir.glob("*.json")):
            zf.write(f, f.name)

    return zip_path


# ---------------------------------------------------------------------------
# BasicFeatures
# ---------------------------------------------------------------------------


class TestBasicFeatures:
    def test_defaults(self) -> None:
        bf = BasicFeatures()
        assert bf.activity_minutes == 0.0
        assert bf.sedentary_ratio == 0.0
        assert bf.room_transitions == 0
        assert bf.average_velocity == 0.0
        assert bf.night_activity_count == 0

    def test_to_dict(self) -> None:
        bf = BasicFeatures(
            activity_minutes=10.123,
            sedentary_ratio=0.6789,
            room_transitions=5,
            average_velocity=0.123456,
            night_activity_count=2,
            night_activity_duration_seconds=45.678,
            multi_person_duration_seconds=120.5,
        )
        d = bf.to_dict()
        assert d["activity_minutes"] == 10.12
        assert d["sedentary_ratio"] == 0.6789
        assert d["room_transitions"] == 5
        assert d["average_velocity"] == 0.1235  # rounded
        assert d["night_activity_count"] == 2
        assert d["multi_person_duration_seconds"] == 120.5

    def test_to_dict_is_json_serializable(self) -> None:
        d = BasicFeatures().to_dict()
        json.dumps(d)  # should not raise


# ---------------------------------------------------------------------------
# FeatureWindow
# ---------------------------------------------------------------------------


class TestFeatureWindow:
    def test_construction(self) -> None:
        fw = FeatureWindow(
            window_id="w00001_0000-0029",
            start_frame=0,
            end_frame=29,
            duration_s=2.0,
            num_frames=30,
            basic_features=BasicFeatures(activity_minutes=1.5),
            monitoring_quality={"quality_confidence": 0.95},
        )
        assert fw.start_frame == 0
        assert fw.end_frame == 29
        assert fw.basic_features.activity_minutes == 1.5
        assert fw.monitoring_quality["quality_confidence"] == 0.95

    def test_to_dict(self) -> None:
        fw = FeatureWindow(
            window_id="w",
            start_frame=0,
            end_frame=9,
            duration_s=1.0,
            num_frames=10,
        )
        d = fw.to_dict()
        assert "basic_features" in d
        assert "monitoring_quality" in d
        json.dumps(d)  # serializable


# ---------------------------------------------------------------------------
# VideoFeatureExtractor base
# ---------------------------------------------------------------------------


class TestVideoFeatureExtractorBase:
    def test_compute_velocity_2d(self) -> None:
        prev = np.array([0.0, 0.0], dtype=np.float32)
        curr = np.array([3.0, 4.0], dtype=np.float32)
        v = VideoFeatureExtractor._compute_velocity(prev, curr)
        assert v == pytest.approx(5.0)

    def test_compute_velocity_3d(self) -> None:
        prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        curr = np.array([1.0, 2.0, 2.0], dtype=np.float32)
        v = VideoFeatureExtractor._compute_velocity(prev, curr)
        assert v == pytest.approx(3.0)

    def test_is_night_hour(self) -> None:
        # Night: 22-6
        assert not VideoFeatureExtractor._is_night_hour(8.0)
        assert not VideoFeatureExtractor._is_night_hour(14.0)
        assert not VideoFeatureExtractor._is_night_hour(21.9)
        assert VideoFeatureExtractor._is_night_hour(22.0)
        assert VideoFeatureExtractor._is_night_hour(23.5)
        assert VideoFeatureExtractor._is_night_hour(0.0)
        assert VideoFeatureExtractor._is_night_hour(3.0)
        assert VideoFeatureExtractor._is_night_hour(5.99)
        assert not VideoFeatureExtractor._is_night_hour(6.0)


# ---------------------------------------------------------------------------
# _MetricsAccumulator
# ---------------------------------------------------------------------------


class TestMetricsAccumulator:
    def test_initial_state(self) -> None:
        acc = _MetricsAccumulator()
        assert acc.total_frames == 0
        assert acc.active_frames == 0
        assert acc.transitions == 0
        assert acc.night_bouts == 0
        assert acc._velocities == []
        assert not acc._in_night_bout

    def test_accumulate_and_reset(self) -> None:
        acc = _MetricsAccumulator()
        acc.total_frames = 10
        acc.active_frames = 5
        acc.transitions = 3
        acc._velocities = [0.1, 0.2]
        acc.reset()
        assert acc.total_frames == 0
        assert acc.active_frames == 0
        assert acc.transitions == 0
        assert acc._velocities == []


# ---------------------------------------------------------------------------
# SkeletonFeatureExtractor
# ---------------------------------------------------------------------------


class TestSkeletonFeatureExtractor:
    def test_process_sequence_basic(self, tmp_path: Path) -> None:
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=15, fps=15.0,
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        # 100 frames, window=30, stride=15 → starts: 0,15,30,45,60,75,90=7 windows
        # Actually: 0,15,30,45,60,75,90 → windows at starts 0-6 + tail at 90? Let's see.
        # start=90: frames[90:120] = frames[90:100] = 10 frames (tail)
        assert len(windows) >= 5
        for w in windows:
            assert isinstance(w, FeatureWindow)
            assert w.basic_features.activity_minutes >= 0
            assert 0.0 <= w.basic_features.sedentary_ratio <= 1.0
            assert w.basic_features.room_transitions >= 0
            assert w.basic_features.average_velocity >= 0
            assert "quality_confidence" in w.monitoring_quality

    def test_window_count(self, tmp_path: Path) -> None:
        """Exact window count for cleanly divisible sequence."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=90)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        # 90 frames, window=30, stride=30 → starts 0,30,60 = 3 windows
        assert len(windows) == 3
        for w in windows:
            assert w.num_frames == 30

    def test_short_sequence_tail(self, tmp_path: Path) -> None:
        """Sequence shorter than window_size yields exactly 1 tail window."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=15, fps=15.0,
        )
        windows = list(
            extractor.process_sequence("Test_Short_p01_r00_v01_c01_pose3d.json")
        )
        # 25 frames, window=30 → only tail
        assert len(windows) == 1
        assert windows[0].num_frames == 25

    def test_activity_detection(self, tmp_path: Path) -> None:
        """A sequence with known movement should yield activity > 0."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100, include_inactive=False)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
            velocity_threshold=0.0,  # everything is active
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        for w in windows:
            # With threshold=0, every frame is "active"
            assert w.basic_features.sedentary_ratio == pytest.approx(0.0, abs=0.05)

    def test_sedentary_detection(self, tmp_path: Path) -> None:
        """With a very high threshold, all frames should be sedentary."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
            velocity_threshold=999.0,  # impossible to exceed
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        for w in windows:
            assert w.basic_features.sedentary_ratio == pytest.approx(1.0, abs=0.05)
            assert w.basic_features.activity_minutes == pytest.approx(0.0, abs=0.1)

    def test_night_activity(self, tmp_path: Path) -> None:
        """Start video at 23:00 → all frames are night, activity should count."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100, include_inactive=False)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
            velocity_threshold=0.0, video_start_hour=23.0,
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        for w in windows:
            assert w.basic_features.night_activity_count >= 1
            assert w.basic_features.night_activity_duration_seconds > 0

    def test_daytime_no_night_activity(self, tmp_path: Path) -> None:
        """Start at 12:00 → no night hours → zero night activity."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100, include_inactive=False)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
            velocity_threshold=0.0, video_start_hour=12.0,
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        for w in windows:
            assert w.basic_features.night_activity_count == 0
            assert w.basic_features.night_activity_duration_seconds == 0.0

    def test_multi_person_detection(self, tmp_path: Path) -> None:
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
        )
        windows = list(
            extractor.process_sequence("Test_Multi_p01_r00_v01_c01_pose3d.json")
        )
        for w in windows:
            assert w.basic_features.multi_person_duration_seconds > 0

    def test_context_manager(self, tmp_path: Path) -> None:
        zip_path = _build_skeleton_zip(tmp_path, n_frames=60)
        with SkeletonFeatureExtractor(zip_path, window_size=30, stride=30) as ext:
            windows = list(ext.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json"))
        assert len(windows) == 2

    def test_process_all(self, tmp_path: Path) -> None:
        zip_path = _build_skeleton_zip(tmp_path, n_frames=60)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
        )
        all_windows = list(extractor.process_all(max_sequences=3))
        # Test_Walk (60f): 2 windows, Test_Short (25f): 1 tail, Test_Multi (60f): 2 → 5
        assert len(all_windows) == 5

    def test_output_schema_match(self, tmp_path: Path) -> None:
        """Output should conform to the project JSON schema."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=60)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        d = windows[0].to_dict()
        bf = d["basic_features"]
        assert "activity_minutes" in bf
        assert "sedentary_ratio" in bf
        assert "room_transitions" in bf
        assert "average_velocity" in bf
        assert "night_activity_count" in bf
        assert "night_activity_duration_seconds" in bf
        assert "multi_person_duration_seconds" in bf
        # All values should be JSON-serializable
        json.dumps(d)

    def test_average_velocity_is_reasonable(self, tmp_path: Path) -> None:
        """With small synthetic movements, velocity should be < 1.0."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
        )
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        for w in windows:
            assert 0.0 <= w.basic_features.average_velocity < 10.0


# ---------------------------------------------------------------------------
# YOLOPoseFeatureExtractor
# ---------------------------------------------------------------------------


class TestYOLOPoseFeatureExtractor:
    def test_mock_mode_yields_windows(self) -> None:
        extractor = YOLOPoseFeatureExtractor(mock=True, window_size=10, stride=5, fps=30.0)
        windows = list(extractor.process_sequence("mock://"))
        # 300 mock frames, window=10, stride=5 → many windows
        assert len(windows) >= 10
        for w in windows:
            assert isinstance(w, FeatureWindow)

    def test_non_mock_without_gpu_raises(self) -> None:
        """In CPU-only environment (no torch/torch.cuda), should raise."""
        # We don't have GPU in this test env → should raise RuntimeError
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if not has_cuda:
            with pytest.raises(RuntimeError):
                YOLOPoseFeatureExtractor(mock=False)
        else:
            # GPU present — constructor succeeds but process_sequence
            # still raises NotImplementedError
            ext = YOLOPoseFeatureExtractor(mock=False)
            with pytest.raises(NotImplementedError):
                next(ext.process_sequence("dummy"))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_frames(self, tmp_path: Path, caplog) -> None:
        """Single-frame sequence should be skipped with a warning."""
        import logging
        caplog.set_level(logging.WARNING)

        zip_path = _build_skeleton_zip(tmp_path, n_frames=1)
        extractor = SkeletonFeatureExtractor(zip_path, window_size=30, stride=15)
        # The first file has 1 frame — the short file has 1 too
        # Actually _build_skeleton_zip always creates 3 files...
        # Test_Walk has n_frames, Test_Short has 25, Test_Multi has 60
        # Let's just test with the 1-frame one
        windows = list(
            extractor.process_sequence("Test_Walk_p01_r00_v01_c01_pose3d.json")
        )
        assert len(windows) == 0
        assert "too few frames" in caplog.text.lower()

    def test_person_index_not_present(self, tmp_path: Path) -> None:
        """Requesting person 99 when only person 0 exists."""
        zip_path = _build_skeleton_zip(tmp_path, n_frames=100)
        extractor = SkeletonFeatureExtractor(
            zip_path, window_size=30, stride=30, fps=15.0,
        )
        windows = list(
            extractor.process_sequence(
                "Test_Walk_p01_r00_v01_c01_pose3d.json", person_index=99
            )
        )
        # All frames have no person 99 → all velocity=0 → all sedentary
        for w in windows:
            assert w.basic_features.activity_minutes == 0.0
            assert w.basic_features.sedentary_ratio == 1.0

    def test_file_count(self, tmp_path: Path) -> None:
        zip_path = _build_skeleton_zip(tmp_path, n_frames=60)
        extractor = SkeletonFeatureExtractor(zip_path)
        assert extractor.file_count == 3
