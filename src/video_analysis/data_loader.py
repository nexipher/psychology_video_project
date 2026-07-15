"""1.1 — Toyota Smarthome Skeleton V1.2 DataLoader.

Reads skeleton JSON files from the compressed zip archive or an extracted
directory.  Parses per-frame 2D / 3D pose data and exposes them through
a clean, typed interface suitable for downstream feature extraction.

Supports:
- Direct reading from .zip (default) or extracted directory
- Lazy file listing & per-file loading
- Frame-level iteration with person filtering
- Sliding-window generator built on top of the raw frame stream
- Pure CPU — no GPU dependency
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from .config import JOINT_NAMES, NUM_JOINTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed data containers
# ---------------------------------------------------------------------------


@dataclass
class PersonPose:
    """Pose data for a single person in a single frame."""

    person_index: int
    pose2d: np.ndarray  # shape (NUM_JOINTS, 2)  — pixel coordinates
    pose3d: np.ndarray  # shape (NUM_JOINTS, 3)  — camera-relative (m)


@dataclass
class SkeletonFrame:
    """All tracked persons in a single frame."""

    frame_index: int
    persons: list[PersonPose]  # length ≤ K (may be empty if no detection)
    num_joints: int = NUM_JOINTS


@dataclass
class SkeletonSequence:
    """Full skeleton sequence loaded from a single JSON file.

    Attributes:
        file_stem:  Original filename without extension, e.g.
                    ``Cook.Cleandishes_p02_r00_v02_c03_pose3d``.
        num_joints: Number of skeleton joints (always 13 for V1.2).
        max_people: Maximum people tracked simultaneously (K).
        frames:     List of SkeletonFrame ordered by frame index.
    """

    file_stem: str
    num_joints: int
    max_people: int
    frames: list[SkeletonFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# SkeletonDataLoader
# ---------------------------------------------------------------------------


class SkeletonDataLoader:
    """Loader for Toyota Smarthome Refined Skeleton V1.2 data.

    Can read directly from the official ``toyota_smarthome_skeleton_v1.2.zip``
    or from an extracted directory of JSON files.

    Usage::

        loader = SkeletonDataLoader("/data/toyota_smarthome_skeleton_v1.2.zip")
        files = loader.list_files()               # discover available sequences
        seq = loader.load("Cook.Cleandishes_p02_r00_v02_c03_pose3d")

        for frame in seq.frames:
            for person in frame.persons:
                pelvis_xy = person.pose2d[0]      # (x, y) in pixels
                pelvis_3d = person.pose3d[0]      # (x, y, z) in metres
    """

    def __init__(self, source_path: str | Path) -> None:
        """
        Args:
            source_path: Path to the skeleton .zip archive **or** an
                         extracted directory containing ``*_pose3d.json`` files.
        """
        self._source = Path(source_path)
        if not self._source.exists():
            raise FileNotFoundError(f"Skeleton source not found: {self._source}")

        self._is_zip = self._source.is_file() and self._source.suffix == ".zip"
        self._zf: zipfile.ZipFile | None = None

        if self._is_zip:
            self._zf = zipfile.ZipFile(self._source, "r")
            self._file_list: list[str] = sorted(
                name
                for name in self._zf.namelist()
                if name.endswith("_pose3d.json")
            )
        else:
            self._file_list = sorted(
                p.name
                for p in self._source.glob("*_pose3d.json")
            )

        logger.info(
            "SkeletonDataLoader ready: %d sequences found (%s mode).",
            len(self._file_list),
            "zip" if self._is_zip else "directory",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_files(self) -> list[str]:
        """Return the sorted list of available skeleton file names (full paths
        inside the zip, or bare filenames for a directory)."""
        return list(self._file_list)

    @property
    def file_count(self) -> int:
        return len(self._file_list)

    def load(self, filename: str) -> SkeletonSequence:
        """Load and parse a single skeleton JSON file.

        Args:
            filename: Either a full zip-internal path or a bare filename,
                      e.g. ``"Cook.Cleandishes_p02_r00_v02_c03_pose3d.json"``.

        Returns:
            Fully parsed SkeletonSequence.
        """
        raw = self._read_json(filename)
        return self._parse(raw, filename)

    def load_by_index(self, index: int) -> SkeletonSequence:
        """Load the *index*-th skeleton file from the file list."""
        return self.load(self._file_list[index])

    def iter_frames(
        self,
        filename: str,
        person_indices: Optional[Sequence[int]] = None,
    ) -> Iterator[SkeletonFrame]:
        """Lazy frame iterator for a single sequence.

        Args:
            filename: Skeleton file to load.
            person_indices: If given, only return poses for these person
                            indices (0-based).  ``None`` means all persons.
        """
        seq = self.load(filename)
        for frame in seq.frames:
            if person_indices is not None:
                frame = SkeletonFrame(
                    frame_index=frame.frame_index,
                    persons=[p for p in frame.persons if p.person_index in person_indices],
                    num_joints=frame.num_joints,
                )
            yield frame

    def iter_sliding_windows(
        self,
        filename: str,
        window_size: int = 30,
        stride: int = 15,
        person_indices: Optional[Sequence[int]] = None,
    ) -> Iterator[list[SkeletonFrame]]:
        """Generate sliding windows over a skeleton sequence.

        Each window is a list of ``SkeletonFrame`` objects covering
        ``window_size`` consecutive frames.  Windows advance by ``stride``
        frames.  If the remaining frames are fewer than ``window_size`` the
        final window is **shorter** (tail window).

        Args:
            filename: Skeleton file to load.
            window_size: Number of frames per window.
            stride: Frame offset between successive windows.
            person_indices: Optional person filter.

        Yields:
            List of SkeletonFrame, one window at a time.
        """
        frames = list(self.iter_frames(filename, person_indices))
        total = len(frames)
        start = 0
        while start < total:
            yield frames[start : start + window_size]
            start += stride

    def close(self) -> None:
        """Release the underlying zip file handle (no-op for directory mode)."""
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self) -> "SkeletonDataLoader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SkeletonDataLoader(source={self._source!r}, "
            f"files={len(self._file_list)}, is_zip={self._is_zip})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_json(self, filename: str) -> dict:
        """Read the raw JSON dict for *filename*."""
        if self._is_zip:
            assert self._zf is not None
            # Accept both bare name and full zip path
            if filename not in self._zf.namelist():
                # Try with .json suffix
                candidate = filename if filename.endswith(".json") else filename + ".json"
                if candidate in self._zf.namelist():
                    filename = candidate
            with self._zf.open(filename) as fh:
                return json.load(fh)  # type: ignore[no-any-return]
        else:
            filepath = self._source / filename
            with open(filepath, encoding="utf-8") as fh:
                return json.load(fh)  # type: ignore[no-any-return]

    @staticmethod
    def _parse(raw: dict, file_stem: str) -> SkeletonSequence:
        """Convert a raw dict into a typed SkeletonSequence."""
        njts = raw.get("njts", NUM_JOINTS)
        K = raw.get("K", 1)
        raw_frames: list = raw.get("frames", [])

        if njts != NUM_JOINTS:
            logger.warning(
                "%s: expected %d joints, got %d — using declared count.",
                file_stem,
                NUM_JOINTS,
                njts,
            )

        frames: list[SkeletonFrame] = []
        for fi, frame_data in enumerate(raw_frames):
            persons: list[PersonPose] = []
            for pi, person_data in enumerate(frame_data):
                pose2d_raw = person_data.get("pose2d", [])
                pose3d_raw = person_data.get("pose3d", [])

                if not pose2d_raw and not pose3d_raw:
                    continue  # empty detection slot

                pose2d = np.array(pose2d_raw, dtype=np.float32).reshape(njts, 2)
                pose3d = np.array(pose3d_raw, dtype=np.float32).reshape(njts, 3)

                persons.append(
                    PersonPose(person_index=pi, pose2d=pose2d, pose3d=pose3d)
                )

            frames.append(
                SkeletonFrame(frame_index=fi, persons=persons, num_joints=njts)
            )

        stem = Path(file_stem).stem  # strip .json if present
        return SkeletonSequence(
            file_stem=stem,
            num_joints=njts,
            max_people=K,
            frames=frames,
        )


# ---------------------------------------------------------------------------
# Utility – joint-name accessor
# ---------------------------------------------------------------------------


def get_joint_name(index: int) -> str:
    """Return the human-readable name for joint *index* (0–12)."""
    if 0 <= index < NUM_JOINTS:
        return JOINT_NAMES[index]
    raise IndexError(f"Joint index {index} out of range [0, {NUM_JOINTS - 1}]")


def get_joint_index(name: str) -> int:
    """Return the index for a named joint."""
    try:
        return JOINT_NAMES.index(name)
    except ValueError:
        raise KeyError(f"Unknown joint name: {name!r}") from None
