"""1.2 — Camera video stream access module.

Provides a unified interface for reading video frames from:
- **Local video files** (.mp4, .avi, etc.) — offline / batch mode
- **RTSP / HTTP streams** — real-time camera ingestion
- **Mock / synthetic source** — for testing without physical hardware

All readers implement the :class:`VideoStreamReader` abstract interface and
are pure-CPU (OpenCV backend, no GPU required).  Frame timestamps are
normalised to seconds for downstream synchronisation.

Usage::

    # Local file
    reader = LocalVideoReader("/data/mp4/Walk_p25_r09_v10_c02.mp4")
    for frame in reader:
        process(frame.image)       # np.ndarray (H, W, 3) BGR
        print(frame.timestamp_s)   # float seconds from start

    # RTSP camera
    reader = RTSPStreamReader("rtsp://192.168.1.100:554/stream1",
                              reconnect=True, timeout_s=10)
    for frame in reader:
        ...

    # Mock (testing)
    reader = MockVideoReader(total_frames=300, fps=30, resolution=(1920, 1080))
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame container
# ---------------------------------------------------------------------------


@dataclass
class VideoFrame:
    """A single decoded video frame with metadata.

    Attributes:
        image:        Frame pixel data as ``np.ndarray`` with shape (H, W, 3),
                      dtype ``uint8``, channel order **BGR** (OpenCV default).
        frame_index:  Zero-based frame number since stream start.
        timestamp_s:  Elapsed seconds from the first frame (monotonic).
        width:        Frame width in pixels.
        height:       Frame height in pixels.
    """

    image: np.ndarray
    frame_index: int
    timestamp_s: float
    width: int
    height: int

    @property
    def rgb(self) -> np.ndarray:
        """Return a **copy** of the frame in RGB order."""
        return self.image[:, :, ::-1].copy()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(height, width, channels)."""
        return self.image.shape  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"VideoFrame(idx={self.frame_index}, "
            f"ts={self.timestamp_s:.3f}s, "
            f"shape={self.shape})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VideoStreamReader(ABC):
    """Abstract interface for video frame sources.

    All concrete readers must implement:
    - :meth:`_open` — initialise the underlying capture backend
    - :meth:`_read_next_frame` — grab + decode the next frame
    - :meth:`close` — release resources
    - Properties: ``fps``, ``total_frames``, ``resolution``, ``duration_s``,
      ``is_live``
    """

    def __init__(self) -> None:
        self._frame_index: int = 0
        self._start_time: float | None = None
        self._closed: bool = False

    # -- Abstract ----------------------------------------------------------

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _read_next_frame(self) -> np.ndarray | None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def total_frames(self) -> int: ...

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]: ...

    @property
    @abstractmethod
    def duration_s(self) -> float: ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True for RTSP/webcam streams that have no finite end."""
        ...

    # -- Iterator protocol -------------------------------------------------

    def __iter__(self) -> "VideoStreamReader":
        return self

    def __next__(self) -> VideoFrame:
        if self._closed:
            raise StopIteration("Video source is closed.")

        if self._start_time is None:
            self._start_time = time.perf_counter()

        raw = self._read_next_frame()
        if raw is None:
            raise StopIteration("End of video stream.")

        h, w = raw.shape[:2]
        elapsed = time.perf_counter() - self._start_time

        frame = VideoFrame(
            image=raw,
            frame_index=self._frame_index,
            timestamp_s=elapsed,
            width=w,
            height=h,
        )
        self._frame_index += 1
        return frame

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> "VideoStreamReader":
        self._open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        fps_str = f"{self.fps:.1f}" if self.fps > 0 else "?"
        return (
            f"{type(self).__name__}(fps={fps_str}, "
            f"frames={self._frame_index}/{self.total_frames or '∞'}, "
            f"live={self.is_live})"
        )


# ---------------------------------------------------------------------------
# Local video file reader
# ---------------------------------------------------------------------------


class LocalVideoReader(VideoStreamReader):
    """Read frames from a local video file (mp4 / avi / mkv / …).

    Parameters:
        file_path: Path to the video file.
        start_frame: Skip to this frame index before yielding (0 = start).
        max_frames: Stop after this many frames (``None`` = entire file).
        frame_skip: Only yield every *N*-th frame.  1 = every frame.
    """

    def __init__(
        self,
        file_path: str | Path,
        start_frame: int = 0,
        max_frames: int | None = None,
        frame_skip: int = 1,
    ) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.frame_skip = max(1, frame_skip)

        self._cap: "cv2.VideoCapture | None" = None  # noqa: F821
        self._cached_fps: float = 0.0
        self._cached_total: int = -1
        self._cached_resolution: Tuple[int, int] = (0, 0)
        self._cached_duration: float = 0.0

        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.file_path}")

        self._open()

    # ------------------------------------------------------------------
    # Abstract implementation
    # ------------------------------------------------------------------

    def _open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.file_path}")

        self._cached_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._cached_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cached_resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if self._cached_fps > 0 and self._cached_total > 0:
            self._cached_duration = self._cached_total / self._cached_fps

        # Seek to start_frame
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._frame_index = self.start_frame

        logger.info(
            "LocalVideoReader: %s — %dx%d @ %.2f fps, %d frames",
            self.file_path.name,
            self._cached_resolution[0],
            self._cached_resolution[1],
            self._cached_fps,
            self._cached_total,
        )

    def _read_next_frame(self) -> np.ndarray | None:
        assert self._cap is not None

        if self.max_frames is not None and self._frame_index >= self.start_frame + self.max_frames:
            return None

        while True:
            ok, frame = self._cap.read()
            if not ok:
                return None

            # Apply frame-skip
            if (self._frame_index - self.start_frame) % self.frame_skip == 0:
                return frame

            self._frame_index += 1
            # continue skipping silently

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._closed = True

    # -- Properties --------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._cached_fps

    @property
    def total_frames(self) -> int:
        effective = self._cached_total - self.start_frame
        if self.max_frames is not None:
            effective = min(effective, self.max_frames)
        return effective // self.frame_skip

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._cached_resolution

    @property
    def duration_s(self) -> float:
        if self._cached_fps > 0:
            return self.total_frames / self._cached_fps
        return 0.0

    @property
    def is_live(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# RTSP / HTTP stream reader
# ---------------------------------------------------------------------------


class RTSPStreamReader(VideoStreamReader):
    """Read frames from an RTSP or HTTP network camera stream.

    Features:
    - Automatic reconnection on connection drop with exponential back-off.
    - Configurable read timeout.
    - Frame counter (total_frames is infinite for live streams).

    Parameters:
        url: RTSP / HTTP stream URL (e.g. ``rtsp://...`` or ``http://.../video``).
        reconnect: If True, attempt to reconnect when the stream drops.
        timeout_s: Seconds to wait for a frame before treating as dropped.
        max_frames: Stop after this many frames (``None`` = unlimited).
        frame_skip: Only yield every *N*-th frame.
    """

    _MAX_RECONNECT_DELAY = 30.0  # seconds
    _RECONNECT_BASE_DELAY = 1.0

    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        timeout_s: float = 10.0,
        max_frames: int | None = None,
        frame_skip: int = 1,
    ) -> None:
        super().__init__()
        self.url = url
        self._reconnect = reconnect
        self._timeout_s = timeout_s
        self.max_frames = max_frames
        self.frame_skip = max(1, frame_skip)

        self._cap: "cv2.VideoCapture | None" = None  # noqa: F821
        self._cached_fps: float = 0.0
        self._cached_resolution: Tuple[int, int] = (0, 0)
        self._reconnect_attempts: int = 0

        self._open()

    # ------------------------------------------------------------------
    # Abstract implementation
    # ------------------------------------------------------------------

    def _open(self) -> None:
        import cv2

        logger.info("RTSPStreamReader: connecting to %s …", self._url_redacted)

        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP stream: {self._url_redacted}")

        self._cached_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._cached_resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        if self._cached_fps <= 0:
            self._cached_fps = 15.0  # sensible default for IP cameras

        logger.info(
            "RTSPStreamReader: connected — %dx%d @ %.1f fps",
            self._cached_resolution[0],
            self._cached_resolution[1],
            self._cached_fps,
        )

    def _read_next_frame(self) -> np.ndarray | None:
        assert self._cap is not None

        if self.max_frames is not None and self._frame_index >= self.max_frames:
            return None

        while True:
            ok, frame = self._cap.read()
            if ok:
                self._reconnect_attempts = 0
                # Apply frame-skip
                if self._frame_index % self.frame_skip == 0:
                    return frame
                self._frame_index += 1
                continue

            # --- Frame read failed — handle reconnection ---
            if not self._reconnect:
                logger.warning("RTSP stream dropped; reconnect disabled.")
                return None

            delay = min(
                self._RECONNECT_BASE_DELAY * (2 ** self._reconnect_attempts),
                self._MAX_RECONNECT_DELAY,
            )
            self._reconnect_attempts += 1
            logger.warning(
                "RTSP stream %s dropped (attempt %d); reconnecting in %.1f s …",
                self._url_redacted,
                self._reconnect_attempts,
                delay,
            )
            time.sleep(delay)

            self._cap.release()
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if self._cap.isOpened():
                logger.info("RTSP stream reconnected.")
            else:
                logger.error("RTSP reconnection failed.")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._closed = True

    # -- Properties --------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._cached_fps

    @property
    def total_frames(self) -> int:
        return -1  # live stream = unknown duration

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._cached_resolution

    @property
    def duration_s(self) -> float:
        return -1.0

    @property
    def is_live(self) -> bool:
        return True

    @property
    def reconnect_attempts(self) -> int:
        """Cumulative reconnection attempts across the session."""
        return self._reconnect_attempts

    @property
    def _url_redacted(self) -> str:
        """Log-safe URL (strip credentials)."""
        # Quick redact: hide anything after @ in rtsp://user:pass@host
        if "@" in self.url:
            parts = self.url.split("@", 1)
            return f"...@{parts[1]}"
        return self.url


# ---------------------------------------------------------------------------
# Mock / synthetic video source (for testing without hardware)
# ---------------------------------------------------------------------------


class MockVideoReader(VideoStreamReader):
    """Generate synthetic video frames — no camera or file needed.

    Each frame contains a simple moving shape (circle) for visual
    differentiation, plus a frame-counter overlay.  Useful for CI,
    unit tests, and pipeline smoke-tests.

    Parameters:
        total_frames: Number of frames to generate before StopIteration.
        fps: Nominal frame rate.
        resolution: (width, height) of generated frames.
        seed: Random seed for reproducible patterns.
    """

    def __init__(
        self,
        total_frames: int = 300,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (640, 480),
        seed: int = 42,
        frame_skip: int = 1,
    ) -> None:
        super().__init__()
        self._total_frames = total_frames
        self._fps = fps
        self._resolution = resolution
        self.frame_skip = max(1, frame_skip)
        self._rng = np.random.RandomState(seed)

        self._circle_x: float = resolution[0] / 2
        self._circle_y: float = resolution[1] / 2
        self._dx: float = 3.0
        self._dy: float = 2.0
        self._radius: int = 30

        self._opened = False

    def _open(self) -> None:
        self._opened = True
        logger.info(
            "MockVideoReader: %dx%d, %d frames @ %.1f fps",
            self._resolution[0],
            self._resolution[1],
            self._total_frames,
            self._fps,
        )

    def _read_next_frame(self) -> np.ndarray | None:
        import cv2

        while True:
            if self._frame_index >= self._total_frames:
                return None

            # Apply frame-skip
            if self._frame_index % self.frame_skip != 0:
                self._frame_index += 1
                continue

            break

        w, h = self._resolution
        # Dark grey background
        frame = np.full((h, w, 3), 48, dtype=np.uint8)

        # Bouncing circle
        self._circle_x += self._dx
        self._circle_y += self._dy

        if self._circle_x - self._radius < 0 or self._circle_x + self._radius > w:
            self._dx *= -1
        if self._circle_y - self._radius < 0 or self._circle_y + self._radius > h:
            self._dy *= -1

        cv2.circle(
            frame,
            (int(self._circle_x), int(self._circle_y)),
            self._radius,
            (0, 220, 0),
            -1,
        )

        # Frame counter text
        cv2.putText(
            frame,
            f"Mock Frame {self._frame_index:05d}",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Grid lines for visual reference
        for i in range(1, 4):
            cv2.line(frame, (w * i // 4, 0), (w * i // 4, h), (64, 64, 64), 1)
            cv2.line(frame, (0, h * i // 4), (w, h * i // 4), (64, 64, 64), 1)

        return frame

    def close(self) -> None:
        self._opened = False
        self._closed = True

    # -- Properties --------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames // self.frame_skip

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def duration_s(self) -> float:
        return self.total_frames / self._fps if self._fps > 0 else 0.0

    @property
    def is_live(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_reader(
    source: str,
    **kwargs,
) -> VideoStreamReader:
    """Factory: auto-detect source type and return an appropriate reader.

    Detection rules:
    - ``"mock://"`` prefix → :class:`MockVideoReader`
    - ``"rtsp://"`` or ``"http://"`` prefix → :class:`RTSPStreamReader`
    - Otherwise → :class:`LocalVideoReader` (file path)

    Args:
        source: URL, file path, or ``"mock://"``.
        **kwargs: Forwarded to the concrete reader constructor.

    Returns:
        A ready-to-use VideoStreamReader.
    """
    if source.startswith("mock://"):
        return MockVideoReader(**kwargs)

    if source.startswith("rtsp://") or source.startswith("http://"):
        return RTSPStreamReader(source, **kwargs)

    return LocalVideoReader(source, **kwargs)
