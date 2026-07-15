"""1.4 — Basic behaviour feature aggregation (daily & hourly).

Consumes a stream of :class:`~.feature_extractor.FeatureWindow` objects
and produces:

- :class:`HourlyAggregation` — A1 metrics binned by clock hour
- :class:`DailyAggregation` — full-day report with hourly breakdown
- :class:`SequenceReport`  — per-sequence summary (matches project JSON Schema)

All aggregation is pure-CPU and works in streaming fashion (no need to
hold every window in memory at once).

Usage::

    from src.video_analysis.aggregator import FeatureAggregator
    from src.video_analysis.feature_extractor import SkeletonFeatureExtractor

    extractor = SkeletonFeatureExtractor("dataset/skeletons.zip")
    agg = FeatureAggregator(fps=15.0, video_start_hour=8.0, user_id="ELDER_01")

    for fw in extractor.process_sequence("some_file.json"):
        agg.ingest(fw)

    report = agg.flush_sequence_report(device_id="CAM_LIVING_01")
    print(report.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .config import NIGHT_END_HOUR, NIGHT_START_HOUR
from .feature_extractor import BasicFeatures, FeatureWindow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class HourlyAggregation:
    """A1 metrics aggregated over one clock hour (0–23).

    Attributes:
        hour: Clock hour [0, 23].
        num_windows: How many sliding windows contributed to this hour.
        activity_minutes: Sum of active time (minutes).
        sedentary_ratio: Weighted-average sedentary fraction across windows.
        room_transitions: Total room / spatial transitions.
        average_velocity: Weighted-average pelvis speed.
        night_activity_count: Night-time activity bouts (nonzero only for
                               night hours 22–5).
        night_activity_duration_seconds: Total active seconds during night.
        multi_person_duration_seconds: Total seconds with ≥ 2 people.
    """

    hour: int = 0
    num_windows: int = 0
    activity_minutes: float = 0.0
    sedentary_ratio: float = 0.0
    room_transitions: int = 0
    average_velocity: float = 0.0
    night_activity_count: int = 0
    night_activity_duration_seconds: float = 0.0
    multi_person_duration_seconds: float = 0.0

    # Internal accumulators (not serialised)
    _weight_sum: float = field(default=0.0, repr=False)
    _sedentary_weighted: float = field(default=0.0, repr=False)
    _velocity_weighted: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict:
        return {
            "hour": self.hour,
            "num_windows": self.num_windows,
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
class DailyAggregation:
    """Full-day (24-hour) A1 report.

    The ``basic_features`` field carries the daily totals; the
    ``hourly_breakdown`` list provides drill-down by hour.
    """

    date: str = ""  # ISO date e.g. "2026-07-14"
    basic_features: BasicFeatures = field(default_factory=BasicFeatures)
    hourly_breakdown: list[HourlyAggregation] = field(default_factory=list)
    monitoring_quality: dict = field(default_factory=dict)
    total_windows: int = 0
    total_duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "total_windows": self.total_windows,
            "total_duration_seconds": round(self.total_duration_s, 2),
            "basic_features": self.basic_features.to_dict(),
            "hourly_breakdown": [h.to_dict() for h in self.hourly_breakdown],
            "monitoring_quality": self.monitoring_quality,
        }


@dataclass
class SequenceReport:
    """Per-sequence report matching the project JSON Schema (§6.1).

    This is the primary output contract for downstream modules
    (voice, questionnaire, fusion).
    """

    user_id: str = ""
    device_id: str = ""
    sequence_name: str = ""
    time_window: dict = field(default_factory=dict)
    monitoring_quality: dict = field(default_factory=dict)
    basic_features: dict = field(default_factory=dict)
    hourly_breakdown: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "device_id": self.device_id,
            "sequence_name": self.sequence_name,
            "time_window": self.time_window,
            "monitoring_quality": self.monitoring_quality,
            "basic_features": self.basic_features,
            "hourly_breakdown": self.hourly_breakdown,
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class FeatureAggregator:
    """Ingest :class:`FeatureWindow` items and build hourly / daily reports.

    Designed for streaming use: call :meth:`ingest` for each window as it
    is produced by the extractor, then call :meth:`flush_daily` or
    :meth:`flush_sequence_report` to obtain the final aggregation.

    Parameters:
        fps: Frames per second (used to map frame indices → wall-clock time).
        video_start_hour: Clock hour (0–23) assigned to frame 0.  This
            anchors the window → hour-of-day mapping.
        user_id: Default user identifier embedded in sequence reports.
        device_id: Default device identifier.
    """

    def __init__(
        self,
        fps: float = 15.0,
        video_start_hour: float = 8.0,
        user_id: str = "",
        device_id: str = "",
    ) -> None:
        self.fps = fps
        self.video_start_hour = video_start_hour
        self.user_id = user_id
        self.device_id = device_id

        # Hourly bins: index 0..23
        self._hours: list[HourlyAggregation] = [
            HourlyAggregation(hour=h) for h in range(24)
        ]

        self._total_windows: int = 0
        self._total_duration_s: float = 0.0
        self._quality_sum: dict = {}  # accumulated quality fields
        self._quality_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, window: FeatureWindow) -> None:
        """Ingest one feature window.

        The window is assigned to its clock hour based on
        ``video_start_hour + start_frame / (fps * 3600)``, and its
        metrics are accumulated into the appropriate hourly bin.
        """
        hour = self._window_to_hour(window)
        bin_ = self._hours[hour]

        # Weight for this window = its duration in seconds
        w = window.duration_s if window.duration_s > 0 else 1.0

        bf = window.basic_features

        # --- Accumulate into hourly bin ---
        bin_.num_windows += 1
        bin_.activity_minutes += bf.activity_minutes
        bin_.room_transitions += bf.room_transitions
        bin_.night_activity_count += bf.night_activity_count
        bin_.night_activity_duration_seconds += bf.night_activity_duration_seconds
        bin_.multi_person_duration_seconds += bf.multi_person_duration_seconds

        # Weighted averages
        bin_._weight_sum += w
        bin_._sedentary_weighted += bf.sedentary_ratio * w
        bin_._velocity_weighted += bf.average_velocity * w

        # --- Daily accumulators ---
        self._total_windows += 1
        self._total_duration_s += w

        # Merge quality
        for k, v in window.monitoring_quality.items():
            if isinstance(v, (int, float)):
                self._quality_sum[k] = self._quality_sum.get(k, 0.0) + float(v)
        self._quality_count += 1

    def ingest_all(self, windows: Iterator[FeatureWindow]) -> "FeatureAggregator":
        """Ingest a stream of windows.  Returns self for chaining."""
        for w in windows:
            self.ingest(w)
        return self

    def flush_daily(self, date: str = "") -> DailyAggregation:
        """Finalise weighted averages and return a :class:`DailyAggregation`.

        Args:
            date: ISO date string (e.g. ``"2026-07-14"``).  If empty the
                  current date is **not** filled in.
        """
        # Finalise weighted averages for each hour
        for bin_ in self._hours:
            if bin_._weight_sum > 0:
                bin_.sedentary_ratio = bin_._sedentary_weighted / bin_._weight_sum
                bin_.average_velocity = bin_._velocity_weighted / bin_._weight_sum

        # Build daily BasicFeatures by summing across all hours
        daily = BasicFeatures()
        for bin_ in self._hours:
            daily.activity_minutes += bin_.activity_minutes
            daily.room_transitions += bin_.room_transitions
            daily.night_activity_count += bin_.night_activity_count
            daily.night_activity_duration_seconds += bin_.night_activity_duration_seconds
            daily.multi_person_duration_seconds += bin_.multi_person_duration_seconds

        # Daily weighted averages (weight = total weighted sum per hour)
        total_weight = sum(b._weight_sum for b in self._hours)
        if total_weight > 0:
            daily.sedentary_ratio = (
                sum(b._sedentary_weighted for b in self._hours) / total_weight
            )
            daily.average_velocity = (
                sum(b._velocity_weighted for b in self._hours) / total_weight
            )

        # Monitoring quality (averaged across all ingested windows)
        quality: dict = {}
        if self._quality_count > 0:
            for k, v in self._quality_sum.items():
                quality[k] = round(v / self._quality_count, 4)
        quality.setdefault("quality_confidence", 1.0)

        # Populated hours only
        populated = [h for h in self._hours if h.num_windows > 0]

        return DailyAggregation(
            date=date,
            basic_features=daily,
            hourly_breakdown=populated,
            monitoring_quality=quality,
            total_windows=self._total_windows,
            total_duration_s=self._total_duration_s,
        )

    def flush_sequence_report(
        self,
        sequence_name: str = "",
        user_id: str | None = None,
        device_id: str | None = None,
        date: str = "",
    ) -> SequenceReport:
        """Produce a :class:`SequenceReport` matching the project JSON Schema.

        This is the canonical output that downstream modules consume.
        """
        daily = self.flush_daily(date=date)
        populated_hours = daily.hourly_breakdown

        # Build time window from the first & last populated hour
        if populated_hours:
            start_h = min(h.hour for h in populated_hours)
            end_h = max(h.hour for h in populated_hours) + 1
        else:
            start_h, end_h = 0, 24

        # Format time_window as in the project schema
        date_prefix = (date + "T") if date else ""
        time_window = {
            "start_time": f"{date_prefix}{start_h:02d}:00:00Z",
            "end_time": f"{date_prefix}{end_h:02d}:00:00Z" if end_h <= 24
            else f"{date_prefix}23:59:59Z",
        }

        return SequenceReport(
            user_id=user_id or self.user_id,
            device_id=device_id or self.device_id,
            sequence_name=sequence_name,
            time_window=time_window,
            monitoring_quality=daily.monitoring_quality,
            basic_features=daily.basic_features.to_dict(),
            hourly_breakdown=[h.to_dict() for h in populated_hours],
        )

    def reset(self) -> None:
        """Clear all accumulators for a fresh sequence."""
        self._hours = [HourlyAggregation(hour=h) for h in range(24)]
        self._total_windows = 0
        self._total_duration_s = 0.0
        self._quality_sum = {}
        self._quality_count = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _window_to_hour(self, window: FeatureWindow) -> int:
        """Map a window's start frame to its clock hour [0, 23]."""
        seconds_from_start = window.start_frame / self.fps if self.fps > 0 else 0.0
        hour_fraction = (self.video_start_hour + seconds_from_start / 3600.0) % 24
        return int(hour_fraction) % 24


# ---------------------------------------------------------------------------
# Convenience — batch processing
# ---------------------------------------------------------------------------


def batch_process_sequences(
    skeleton_source: str,
    user_id: str = "ELDER_001",
    device_id: str = "CAMERA_MAIN",
    *,
    window_size: int = 30,
    stride: int = 15,
    fps: float = 15.0,
    video_start_hour: float = 8.0,
    velocity_threshold: float = 0.02,
    room_transition_threshold: float = 0.5,
    max_sequences: int | None = None,
) -> Iterator[SequenceReport]:
    """End-to-end batch pipeline: skeleton files → aggregated reports.

    Chains :class:`SkeletonFeatureExtractor` → :class:`FeatureAggregator`
    for every skeleton file in *skeleton_source*, yielding one
    :class:`SequenceReport` per file.

    Args:
        skeleton_source: Path to skeleton .zip or directory.
        user_id: Embedded in each report.
        device_id: Embedded in each report.
        window_size: Frames per sliding window.
        stride: Window stride.
        fps: Frame rate.
        video_start_hour: Clock hour for frame 0.
        velocity_threshold: Minimum pelvis speed (m/frame) for "active".
        room_transition_threshold: Minimum displacement (m) for transition.
        max_sequences: Limit number of files (None = all).
    """
    from .feature_extractor import SkeletonFeatureExtractor

    extractor = SkeletonFeatureExtractor(
        skeleton_source=skeleton_source,
        window_size=window_size,
        stride=stride,
        fps=fps,
        video_start_hour=video_start_hour,
        velocity_threshold=velocity_threshold,
        room_transition_threshold=room_transition_threshold,
    )

    files = extractor._loader.list_files()
    if max_sequences is not None:
        files = files[:max_sequences]

    for fname in files:
        agg = FeatureAggregator(
            fps=fps,
            video_start_hour=video_start_hour,
            user_id=user_id,
            device_id=device_id,
        )
        agg.ingest_all(extractor.process_sequence(fname))
        yield agg.flush_sequence_report(
            sequence_name=fname,
            user_id=user_id,
            device_id=device_id,
        )
        agg.reset()

    extractor.close()
