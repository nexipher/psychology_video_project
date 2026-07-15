"""1.3 — VideoFeatureExtractor base class + Skeleton-based implementation.

Provides:
- :class:`BasicFeatures` — structured output for A1-level 6 indicators
- :class:`FeatureWindow` — a single time-window result with metadata
- :class:`VideoFeatureExtractor` — abstract base (pluggable backends)
- :class:`SkeletonFeatureExtractor` — pure-CPU, reads Toyota Smarthome
  V1.2 skeleton data and computes A1 metrics per sliding window
- :class:`YOLOPoseFeatureExtractor` — stub for GPU-backed YOLOv8-Pose
  pipeline (raises NotImplementedError in CPU mode; mock-friendly)

Usage::

    from src.video_analysis.feature_extractor import SkeletonFeatureExtractor

    extractor = SkeletonFeatureExtractor(
        skeleton_source="dataset/toyota_smarthome_skeleton_v1.2.zip",
        window_size=30, stride=15, fps=15.0,
    )
    for fw in extractor.process_sequence("Cook.Cleandishes_p02_r00_v02_c03_pose3d"):
        print(fw.basic_features)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .config import (
    HEAD_JOINT,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    NUM_JOINTS,
    PELVIS_JOINT,
)
from .data_loader import SkeletonDataLoader, SkeletonFrame
from .sliding_window import SlidingWindow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output data types
# ---------------------------------------------------------------------------


@dataclass
class BasicFeatures:
    """A1-level 6 basic behavioural indicators for one time window.

    All values are aggregated over the window duration unless noted
    otherwise.
    """

    # 1. Cumulative active (non-static) time in minutes
    activity_minutes: float = 0.0

    # 2. Fraction of time spent sedentary / nearly motionless  [0, 1]
    sedentary_ratio: float = 0.0

    # 3. Number of room / spatial-zone transitions detected
    room_transitions: int = 0

    # 4. Mean pelvis movement speed (metres / second for 3D,
    #    pixels / second for 2D-only data)
    average_velocity: float = 0.0

    # 5a. Number of discrete activity bouts during night window
    night_activity_count: int = 0

    # 5b. Total seconds spent active during night window
    night_activity_duration_seconds: float = 0.0

    # 6. Total seconds where ≥ 2 people appear in the same frame
    multi_person_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serialisable dict matching the project schema."""
        return {
            "activity_minutes": round(self.activity_minutes, 2),
            "sedentary_ratio": round(self.sedentary_ratio, 4),
            "room_transitions": self.room_transitions,
            "average_velocity": round(self.average_velocity, 4),
            "night_activity_count": self.night_activity_count,
            "night_activity_duration_seconds": round(
                self.night_activity_duration_seconds, 2
            ),
            "multi_person_duration_seconds": round(
                self.multi_person_duration_seconds, 2
            ),
        }


@dataclass
class FeatureWindow:
    """One sliding-window result bundle.

    Attributes:
        window_id: Unique identifier for this window (e.g. ``"seq_0000-0029"``).
        start_frame: First frame index (inclusive).
        end_frame: Last frame index (inclusive).
        duration_s: Wall-clock duration of the window in seconds.
        num_frames: Number of frames actually used.
        basic_features: Computed A1 metrics.
        monitoring_quality: Metadata about data quality within the window.
    """

    window_id: str
    start_frame: int
    end_frame: int
    duration_s: float
    num_frames: int
    basic_features: BasicFeatures = field(default_factory=BasicFeatures)
    monitoring_quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_s": round(self.duration_s, 3),
            "num_frames": self.num_frames,
            "basic_features": self.basic_features.to_dict(),
            "monitoring_quality": self.monitoring_quality,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VideoFeatureExtractor(ABC):
    """Abstract feature extractor — platform-agnostic interface.

    Subclasses must implement :meth:`extract_from_frames` (video / YOLO
    path) **or** :meth:`extract_from_skeleton` (pre-computed skeleton
    path).  At least one of the two must be functional.
    """

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 15,
        fps: float = 15.0,
    ) -> None:
        self.window_size = window_size
        self.stride = stride
        self.fps = fps
        self._window_counter: int = 0

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process_sequence(self, source: str, **kwargs) -> Iterator[FeatureWindow]:
        """Process a single video / skeleton sequence and yield one
        FeatureWindow per sliding-window step."""
        ...

    # ------------------------------------------------------------------
    # Shared metric computation (usable by all subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_velocity(
        prev_pelvis: np.ndarray,
        curr_pelvis: np.ndarray,
    ) -> float:
        """Euclidean distance between two pelvis positions.

        Works with both 2-D (pixels) and 3-D (metres) coordinates.
        """
        delta = curr_pelvis - prev_pelvis
        return float(np.sqrt((delta * delta).sum()))

    @staticmethod
    def _is_night_hour(hour: float) -> bool:
        """Return True if *hour* falls within the night window.

        Handles wrap-around (e.g. 22:00 – 06:00).
        """
        if NIGHT_START_HOUR <= NIGHT_END_HOUR:
            return NIGHT_START_HOUR <= hour < NIGHT_END_HOUR
        # Wrap: 22-24 OR 0-6
        return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR

    def _make_window(
        self,
        start_frame: int,
        end_frame: int,
        features: BasicFeatures,
        num_frames: int,
        quality: dict | None = None,
    ) -> FeatureWindow:
        self._window_counter += 1
        duration = num_frames / self.fps if self.fps > 0 else 0.0
        return FeatureWindow(
            window_id=f"w{self._window_counter:05d}_{start_frame:05d}-{end_frame:05d}",
            start_frame=start_frame,
            end_frame=end_frame,
            duration_s=duration,
            num_frames=num_frames,
            basic_features=features,
            monitoring_quality=quality or {},
        )


# ---------------------------------------------------------------------------
# Skeleton-based extractor  (pure CPU)
# ---------------------------------------------------------------------------


class SkeletonFeatureExtractor(VideoFeatureExtractor):
    """Compute A1 features directly from pre-extracted skeleton data.

    No GPU required.  Reads Toyota Smarthome V1.2 skeleton JSON files
    (via :class:`SkeletonDataLoader`) and aggregates the six A1
    indicators inside a sliding window.

    Parameters:
        skeleton_source: Path to skeleton .zip or extracted directory.
        window_size: Number of frames per sliding window.
        stride: Frame step between successive windows.
        fps: Nominal frame rate (used to convert frame counts ↔ seconds).
        velocity_threshold: Minimum pelvis speed (m/frame or px/frame) to
            classify a frame as "active".
        room_transition_threshold: Minimum pelvis displacement (m or px)
            between consecutive frames to count as a spatial transition.
        video_start_hour: Clock hour (0–23) for frame 0.  Used for
            night-activity classification.
    """

    def __init__(
        self,
        skeleton_source: str | Path,
        window_size: int = 30,
        stride: int = 15,
        fps: float = 15.0,
        velocity_threshold: float = 0.02,  # m/frame ≈ 0.3 m/s at 15 fps
        room_transition_threshold: float = 0.5,  # m
        video_start_hour: float = 8.0,
    ) -> None:
        super().__init__(window_size=window_size, stride=stride, fps=fps)
        self._loader = SkeletonDataLoader(str(skeleton_source))
        self.velocity_threshold = velocity_threshold
        self.room_transition_threshold = room_transition_threshold
        self.video_start_hour = video_start_hour

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_sequence(
        self,
        filename: str,
        person_index: int = 0,
    ) -> Iterator[FeatureWindow]:
        """Extract features from one skeleton file, window by window.

        Args:
            filename: Skeleton JSON filename (as listed by the loader).
            person_index: Which tracked person to follow (default 0).

        Yields:
            FeatureWindow for each complete sliding window + optional tail.
        """
        seq = self._loader.load(filename)
        frames = seq.frames
        total = len(frames)

        if total < 2:
            logger.warning("%s: too few frames (%d); skipping.", filename, total)
            return

        sw = SlidingWindow[SkeletonFrame](window_size=self.window_size, stride=self.stride)

        # Per-frame state used for accumulation
        # We accumulate metrics incrementally and flush on each window.
        acc = _MetricsAccumulator()

        prev_pelvis: np.ndarray | None = None
        prev_is_active: bool = False

        for fi, frame in enumerate(frames):
            # --- Per-frame computation ---
            hour = (self.video_start_hour + fi / (self.fps * 3600.0)) % 24

            # Select target person
            person = _get_person(frame, person_index)
            if person is None:
                # No detection — carry forward previous state but mark as
                # "uncertain" quality.  We still push the frame so window
                # sizes are consistent.
                sw.push(frame)
                if sw.is_ready():
                    yield self._flush_window(sw, acc, start=fi - self.window_size + 1, end=fi)
                    sw.advance()
                    acc.reset()
                continue

            # Always use 2-D (x, y) pelvis for velocity so shapes are
            # consistent across windows.  3-D data: drop z; 2-D: as-is.
            if person.pose3d.size > 0:
                pelvis_pos = person.pose3d[PELVIS_JOINT, :2].astype(np.float64)
            else:
                pelvis_pos = person.pose2d[PELVIS_JOINT].astype(np.float64)

            # Velocity vs previous frame
            velocity = 0.0
            if prev_pelvis is not None:
                velocity = self._compute_velocity(prev_pelvis, pelvis_pos)

            is_active = velocity > self.velocity_threshold
            is_night = self._is_night_hour(hour)
            n_people = len(frame.persons) if frame.persons else 0

            # Room transition detection
            is_transition = velocity > self.room_transition_threshold

            # --- Accumulate ---
            acc.total_frames += 1
            if is_active:
                acc.active_frames += 1
            if is_transition:
                acc.transitions += 1
            if is_night and is_active:
                acc.night_active_frames += 1
                if not acc._in_night_bout:
                    acc._in_night_bout = True
                    acc.night_bouts += 1
            else:
                acc._in_night_bout = False
            if n_people >= 2:
                acc.multi_person_frames += 1

            acc._velocities.append(velocity)

            prev_pelvis = pelvis_pos
            prev_is_active = is_active

            # --- Sliding window ---
            sw.push(frame)
            if sw.is_ready():
                start = fi - self.window_size + 1
                yield self._flush_window(sw, acc, start=start, end=fi)
                sw.advance()
                acc.reset()
                # Re-seed prev_pelvis from the last frame kept in buffer
                # so velocity continuity is maintained across windows.
                buf = list(sw._buffer)
                if buf:
                    last_person = _get_person(buf[-1], person_index)
                    if last_person is not None:
                        if last_person.pose3d.size > 0:
                            prev_pelvis = last_person.pose3d[PELVIS_JOINT, :2].astype(np.float64)
                        else:
                            prev_pelvis = last_person.pose2d[PELVIS_JOINT].astype(np.float64)

        # Tail window
        if len(sw._buffer) > 0:
            tail_start = total - len(sw._buffer)
            yield self._flush_window(sw, acc, start=tail_start, end=total - 1)

    def process_all(
        self,
        person_index: int = 0,
        max_sequences: int | None = None,
    ) -> Iterator[FeatureWindow]:
        """Convenience: process every skeleton file in the data source.

        Args:
            person_index: Person to track.
            max_sequences: Limit number of files (None = all).
        """
        files = self._loader.list_files()
        if max_sequences is not None:
            files = files[:max_sequences]
        for fname in files:
            yield from self.process_sequence(fname, person_index=person_index)

    @property
    def file_count(self) -> int:
        return self._loader.file_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_window(
        self,
        sw: SlidingWindow,
        acc: "_MetricsAccumulator",
        start: int,
        end: int,
    ) -> FeatureWindow:
        """Build a FeatureWindow from the accumulator state."""
        n = acc.total_frames
        if n == 0:
            n = 1  # avoid div-by-zero

        frame_duration_s = 1.0 / self.fps if self.fps > 0 else 0.0

        features = BasicFeatures(
            activity_minutes=(acc.active_frames * frame_duration_s) / 60.0,
            sedentary_ratio=1.0 - (acc.active_frames / n),
            room_transitions=acc.transitions,
            average_velocity=float(np.mean(acc._velocities)) if acc._velocities else 0.0,
            night_activity_count=acc.night_bouts,
            night_activity_duration_seconds=acc.night_active_frames * frame_duration_s,
            multi_person_duration_seconds=acc.multi_person_frames * frame_duration_s,
        )

        quality = {
            "effective_duration_seconds": n * frame_duration_s,
            "missing_frames": self.window_size - n if n < self.window_size else 0,
            "occlusion_ratio": (
                0.0  # skeleton data does not carry occlusion info
            ),
            "quality_confidence": 1.0 if n >= self.window_size * 0.8 else 0.5,
        }

        return self._make_window(start, end, features, num_frames=n, quality=quality)

    def close(self) -> None:
        self._loader.close()

    def __enter__(self) -> "SkeletonFeatureExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# YOLOv8-Pose extractor  (GPU-backed stub)
# ---------------------------------------------------------------------------


class YOLOPoseFeatureExtractor(VideoFeatureExtractor):
    """GPU-backed feature extractor using YOLOv8-Pose + ByteTrack.

    **Currently a stub** — raises :exc:`NotImplementedError` in CPU mode.
    When a GPU is available this class will:

    1. Load frames via :class:`~.video_stream.VideoStreamReader`.
    2. Run YOLOv8-Pose inference to obtain 2D keypoints per person.
    3. Link detections across frames with ByteTrack.
    4. Feed the resulting per-person skeleton stream into the same
       sliding-window metric pipeline used by
       :class:`SkeletonFeatureExtractor`.

    For testing without a GPU use ``mock=True``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        window_size: int = 30,
        stride: int = 15,
        fps: float = 15.0,
        mock: bool = False,
    ) -> None:
        super().__init__(window_size=window_size, stride=stride, fps=fps)
        self.model_path = model_path
        self._mock = mock

        if not mock:
            self._check_gpu()

    @staticmethod
    def _check_gpu() -> None:
        """Verify GPU is available; raise otherwise."""
        try:
            import torch  # type: ignore[import-untyped]

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU not available. Use SkeletonFeatureExtractor for CPU mode, "
                    "or pass mock=True for testing."
                )
        except ImportError:
            raise RuntimeError(
                "PyTorch not installed. Use SkeletonFeatureExtractor for CPU mode."
            )

    def process_sequence(
        self, source: str, **kwargs
    ) -> Iterator[FeatureWindow]:
        if not self._mock:
            raise NotImplementedError(
                "YOLOPoseFeatureExtractor.process_sequence is not yet "
                "implemented for real GPU inference.  Use mock=True for "
                "a synthetic test stream."
            )

        # Mock path — generate synthetic feature windows
        from .video_stream import MockVideoReader

        reader = MockVideoReader(total_frames=300, fps=self.fps)
        sw = SlidingWindow[BasicFeatures](window_size=self.window_size, stride=self.stride)

        for frame in reader:
            # Synthesise dummy features (random walk around baseline)
            bf = BasicFeatures(
                activity_minutes=np.random.uniform(0, 5),
                sedentary_ratio=np.random.uniform(0.5, 1.0),
                room_transitions=np.random.randint(0, 3),
                average_velocity=np.random.uniform(0, 0.5),
                night_activity_count=np.random.randint(0, 2),
                night_activity_duration_seconds=np.random.uniform(0, 60),
                multi_person_duration_seconds=np.random.uniform(0, 30),
            )
            sw.push(bf)
            if sw.is_ready():
                yield self._make_window(
                    start_frame=frame.frame_index - self.window_size + 1,
                    end_frame=frame.frame_index,
                    features=bf,
                    num_frames=self.window_size,
                )
                sw.advance()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _MetricsAccumulator:
    """Mutable bag for per-window metric accumulation."""

    __slots__ = (
        "total_frames",
        "active_frames",
        "transitions",
        "night_active_frames",
        "night_bouts",
        "multi_person_frames",
        "_velocities",
        "_in_night_bout",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_frames = 0
        self.active_frames = 0
        self.transitions = 0
        self.night_active_frames = 0
        self.night_bouts = 0
        self.multi_person_frames = 0
        self._velocities: list[float] = []
        self._in_night_bout = False


def _get_person(frame: SkeletonFrame, index: int):
    """Return the PersonPose at *index* or None if not present."""
    for p in frame.persons:
        if p.person_index == index:
            return p
    return None
