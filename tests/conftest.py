"""Shared test fixtures for the video_analysis test suite."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.video_analysis.config import NUM_JOINTS
from src.video_analysis.feature_extractor import BasicFeatures, FeatureWindow


# ---------------------------------------------------------------------------
# Skeleton zip builder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def skeleton_zip_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a reusable synthetic skeleton .zip with multiple sequences."""
    tmp_path = tmp_path_factory.mktemp("skel_data")
    zip_path = tmp_path / "test_skeletons.zip"
    json_dir = tmp_path / "skel"
    json_dir.mkdir()

    def _make_seq(name: str, n_frames: int, K: int, movement: bool = True) -> None:
        frames = []
        for fi in range(n_frames):
            frame_people = []
            for pi in range(K):
                if movement:
                    x = 0.5 * np.sin(fi * 0.3) + np.random.randn() * 0.01
                    y = fi * 0.01 + np.random.randn() * 0.005
                else:
                    x = np.random.randn() * 0.001
                    y = np.random.randn() * 0.001

                pose3d = [0.0] * (NUM_JOINTS * 3)
                pose3d[0] = x
                pose3d[1] = y
                pose3d[2] = 0.0
                pose2d = [0.0] * (NUM_JOINTS * 2)
                pose2d[0] = x * 100 + 320
                pose2d[1] = y * 100 + 240
                frame_people.append({"pose2d": pose2d, "pose3d": pose3d})

            frames.append(frame_people)

        raw = {"njts": NUM_JOINTS, "K": K, "frames": frames}
        (json_dir / f"{name}.json").write_text(json.dumps(raw))

    _make_seq("Test_Walk_p01_r00_v01_c01_pose3d", n_frames=120, K=1, movement=True)
    _make_seq("Test_Sit_p02_r00_v01_c01_pose3d", n_frames=90, K=1, movement=False)
    _make_seq("Test_Multi_p03_r00_v01_c01_pose3d", n_frames=60, K=2, movement=True)

    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(json_dir.glob("*.json")):
            zf.write(f, f.name)

    return zip_path


@pytest.fixture(scope="session")
def real_skeleton_zip() -> Path:
    """Path to the real Toyota Smarthome skeleton zip."""
    p = Path("dataset/toyota_smarthome_skeleton_v1.2.zip")
    if not p.exists():
        pytest.skip("Real skeleton dataset not available")
    return p


# ---------------------------------------------------------------------------
# Feature window builders
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_window() -> FeatureWindow:
    """A single FeatureWindow with known metric values."""
    return FeatureWindow(
        window_id="w_test_0000",
        start_frame=0,
        end_frame=29,
        duration_s=2.0,
        num_frames=30,
        basic_features=BasicFeatures(
            activity_minutes=1.5,
            sedentary_ratio=0.5,
            room_transitions=2,
            average_velocity=0.3,
            night_activity_count=0,
            night_activity_duration_seconds=0.0,
            multi_person_duration_seconds=15.0,
        ),
        monitoring_quality={"quality_confidence": 0.95},
    )


@pytest.fixture
def window_stream() -> list[FeatureWindow]:
    """10 identical synthetic windows at 8:00 (frame 0)."""
    return [
        FeatureWindow(
            window_id=f"w_stream_{i:04d}",
            start_frame=0,
            end_frame=29,
            duration_s=2.0,
            num_frames=30,
            basic_features=BasicFeatures(
                activity_minutes=2.0,
                sedentary_ratio=0.4,
                room_transitions=1,
                average_velocity=0.25,
                night_activity_count=0,
                night_activity_duration_seconds=0.0,
                multi_person_duration_seconds=20.0,
            ),
        )
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# GPU-check helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """True if a CUDA-capable GPU is present."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def cpu_only() -> bool:
    """True if we are running in CPU-only mode (the default)."""
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True
