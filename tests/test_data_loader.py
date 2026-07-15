"""Tests for 1.1 — SkeletonDataLoader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.video_analysis.config import JOINT_NAMES, NUM_JOINTS
from src.video_analysis.data_loader import (
    PersonPose,
    SkeletonDataLoader,
    SkeletonFrame,
    SkeletonSequence,
    get_joint_index,
    get_joint_name,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic skeleton data without touching the real dataset
# ---------------------------------------------------------------------------

_NJ = NUM_JOINTS  # 13


def _make_person(pi: int, n_frames: int = 1) -> list[dict]:
    """Return a list of *n_frames* person dicts (each a single-person frame)."""
    frames = []
    for _ in range(n_frames):
        frames.append(
            [
                {
                    "pose2d": [float(i) for i in range(_NJ * 2)],
                    "pose3d": [float(i) for i in range(_NJ * 3)],
                }
            ]
        )
    return frames


def _make_raw_seq(n_frames: int = 100, K: int = 1) -> dict:
    """Build a minimal valid skeleton JSON dict."""
    frames = []
    for fi in range(n_frames):
        frame_people = []
        for pi in range(K):
            frame_people.append(
                {
                    "pose2d": [float(fi * 100 + pi + j) for j in range(_NJ * 2)],
                    "pose3d": [float(fi * 100 + pi + j) for j in range(_NJ * 3)],
                }
            )
        frames.append(frame_people)
    return {"njts": _NJ, "K": K, "frames": frames}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkeletonDataLoader:
    """Tests against a synthetic in-memory zip archive."""

    @staticmethod
    def _build_zip(tmp_path: Path) -> Path:
        """Create a minimal skeleton zip with 3 synthetic sequences."""
        zip_path = tmp_path / "test_skeletons.zip"
        # We'll create a directory of JSONs, then zip it
        import zipfile

        json_dir = tmp_path / "skeletons"
        json_dir.mkdir()

        for name, n_frames, K in [
            ("Cook.Cleandishes_p02_r00_v02_c03_pose3d", 50, 1),
            ("WatchTV_p05_r01_v03_c01_pose3d", 80, 2),
            ("Eat_p10_r02_v04_c02_pose3d", 30, 1),
        ]:
            raw = _make_raw_seq(n_frames, K)
            filepath = json_dir / f"{name}.json"
            filepath.write_text(json.dumps(raw))

        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in sorted(json_dir.glob("*.json")):
                zf.write(f, f.name)

        return zip_path

    def test_list_files(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        files = loader.list_files()
        assert len(files) == 3
        assert all(f.endswith("_pose3d.json") for f in files)
        assert loader.file_count == 3

    def test_load_single_person(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        seq = loader.load("Cook.Cleandishes_p02_r00_v02_c03_pose3d.json")

        assert isinstance(seq, SkeletonSequence)
        assert seq.num_frames == 50
        assert seq.num_joints == NUM_JOINTS
        assert seq.max_people == 1
        assert len(seq.frames) == 50

        # Spot-check first frame
        f0 = seq.frames[0]
        assert f0.frame_index == 0
        assert len(f0.persons) == 1
        p0 = f0.persons[0]
        assert p0.person_index == 0
        assert p0.pose2d.shape == (NUM_JOINTS, 2)
        assert p0.pose3d.shape == (NUM_JOINTS, 3)
        assert p0.pose2d.dtype == np.float32

    def test_load_multi_person(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        seq = loader.load("WatchTV_p05_r01_v03_c01_pose3d.json")

        assert seq.max_people == 2
        f0 = seq.frames[0]
        assert len(f0.persons) == 2
        assert f0.persons[0].person_index == 0
        assert f0.persons[1].person_index == 1

    def test_load_by_index(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        seq = loader.load_by_index(0)
        assert seq.num_frames == 50

    def test_iter_frames_person_filter(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        frames = list(
            loader.iter_frames(
                "WatchTV_p05_r01_v03_c01_pose3d.json",
                person_indices=[0],
            )
        )
        assert len(frames) == 80
        for f in frames:
            assert len(f.persons) == 1
            assert f.persons[0].person_index == 0

    def test_iter_sliding_windows(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        windows = list(
            loader.iter_sliding_windows(
                "Cook.Cleandishes_p02_r00_v02_c03_pose3d.json",
                window_size=10,
                stride=5,
            )
        )
        # 50 frames, window=10, stride=5 → starts: 0,5,10,15,20,25,30,35,40,45 = 10 windows
        # The last (start=45) is a tail of 5 frames
        assert len(windows) == 10
        assert all(len(w) == 10 for w in windows[:-1])
        assert len(windows[-1]) == 5  # tail

    def test_iter_sliding_windows_tail(self, tmp_path: Path) -> None:
        """A sequence not evenly divisible by stride should yield a tail."""
        zip_path = self._build_zip(tmp_path)
        loader = SkeletonDataLoader(str(zip_path))
        windows = list(
            loader.iter_sliding_windows(
                "Cook.Cleandishes_p02_r00_v02_c03_pose3d.json",
                window_size=10,
                stride=7,
            )
        )
        # 50 frames, window=10, stride=7.
        # starts: 0,7,14,21,28,35,42,49 → 8 windows
        # window at 49 has only 1 frame (tail)
        assert len(windows) == 8
        assert len(windows[-1]) == 1  # tail window

    def test_context_manager(self, tmp_path: Path) -> None:
        zip_path = self._build_zip(tmp_path)
        with SkeletonDataLoader(str(zip_path)) as loader:
            seq = loader.load_by_index(0)
            assert seq.num_frames == 50

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            SkeletonDataLoader("/nonexistent/path.zip")

    def test_directory_mode(self, tmp_path: Path) -> None:
        """Loader should also work with an extracted directory."""
        json_dir = tmp_path / "skeletons"
        json_dir.mkdir()
        raw = _make_raw_seq(n_frames=20, K=1)
        (json_dir / "test_pose3d.json").write_text(json.dumps(raw))

        loader = SkeletonDataLoader(str(json_dir))
        assert not loader._is_zip
        assert loader.file_count == 1
        seq = loader.load("test_pose3d.json")
        assert seq.num_frames == 20


class TestJointUtilities:
    def test_get_joint_name_valid(self) -> None:
        assert get_joint_name(0) == "pelvis"
        assert get_joint_name(3) == "head"
        assert get_joint_name(12) == "left_ankle"

    def test_get_joint_name_invalid(self) -> None:
        with pytest.raises(IndexError):
            get_joint_name(-1)
        with pytest.raises(IndexError):
            get_joint_name(13)

    def test_get_joint_index_valid(self) -> None:
        assert get_joint_index("pelvis") == 0
        assert get_joint_index("left_ankle") == 12

    def test_get_joint_index_invalid(self) -> None:
        with pytest.raises(KeyError):
            get_joint_index("nonexistent")

    def test_roundtrip(self) -> None:
        for i, name in enumerate(JOINT_NAMES):
            assert get_joint_index(name) == i
            assert get_joint_name(i) == name
