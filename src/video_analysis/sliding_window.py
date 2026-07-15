"""1.5 — High-performance temporal sliding window.

Built on ``collections.deque`` for O(1) push/pop at both ends.  Designed to
run on CPU with amortised per-push cost well under 1.5 ms so that it never
becomes the bottleneck in a real-time 30 fps pipeline.

Typical usage::

    sw = SlidingWindow(window_size=30, stride=15)
    for frame in frame_stream:
        sw.push(frame)
        if sw.is_ready():
            window = sw.get_window()      # list of last ≤30 frames
            features = compute(window)    # your feature-extraction logic
            sw.advance()                  # drop *stride* oldest frames
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Generic, Iterator, Optional, TypeVar

from .config import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STRIDE

T = TypeVar("T")

# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------


class SlidingWindow(Generic[T]):
    """Fixed-capacity FIFO sliding window with stride-based advancement.

    Frames are pushed one at a time via :meth:`push`.  Once the window reaches
    *window_size* items (i.e. :meth:`is_ready` returns `True`), callers can
    read :meth:`get_window` to obtain the current slice and then
    :meth:`advance` to drop *stride* oldest items — the window immediately
    starts collecting toward the next output.

    Parameters:
        window_size: Maximum number of items held in the window.
        stride:      How many items to drop on each :meth:`advance`.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_WINDOW_STRIDE,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.window_size: int = window_size
        self.stride: int = stride
        self._buffer: Deque[T] = deque(maxlen=window_size)

        # Internal frame counter (total pushes)
        self._total_pushes: int = 0

        # Performance tracking
        self._last_push_time_ns: int = 0
        self._last_advance_time_ns: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(self, item: T) -> None:
        """Push a single item into the window.

        If the buffer is already full (len == window_size), the oldest item
        is silently dropped — in that scenario :meth:`advance` should have
        been called first, but we handle it gracefully.

        Complexity: O(1) amortised.
        """
        t0 = time.perf_counter_ns()
        self._buffer.append(item)
        self._total_pushes += 1
        self._last_push_time_ns = time.perf_counter_ns() - t0

    def is_ready(self) -> bool:
        """Return True once exactly ``window_size`` items have accumulated."""
        return len(self._buffer) == self.window_size

    @property
    def is_full(self) -> bool:
        """Alias for :meth:`is_ready`."""
        return self.is_ready()

    def get_window(self) -> list[T]:
        """Return a **copy** of the current window contents (oldest → newest).

        Returns a plain ``list`` so callers are free to mutate it without
        affecting the internal buffer.
        """
        return list(self._buffer)

    def get_window_as_deque(self) -> Deque[T]:
        """Return a **shallow copy** of the internal deque.

        Faster than ``get_window()`` when you only need to iterate.
        """
        return self._buffer.copy()

    def advance(self) -> None:
        """Drop the oldest ``stride`` items, making room for new data.

        After this call :meth:`is_ready` returns False until the buffer
        fills up again.

        Complexity: O(stride).
        """
        t0 = time.perf_counter_ns()
        drop = min(self.stride, len(self._buffer))
        for _ in range(drop):
            self._buffer.popleft()
        self._last_advance_time_ns = time.perf_counter_ns() - t0

    def reset(self) -> None:
        """Empty the buffer and reset counters."""
        self._buffer.clear()
        self._total_pushes = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> Deque[T]:
        """Direct access to the internal deque (read-only intent)."""
        return self._buffer

    @property
    def total_pushes(self) -> int:
        """Total number of items ever pushed (monotonic)."""
        return self._total_pushes

    @property
    def last_push_time_us(self) -> float:
        """Duration of the most recent :meth:`push` in microseconds."""
        return self._last_push_time_ns / 1000.0

    @property
    def last_advance_time_us(self) -> float:
        """Duration of the most recent :meth:`advance` in microseconds."""
        return self._last_advance_time_ns / 1000.0

    # ------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"SlidingWindow(size={self.window_size}, stride={self.stride}, "
            f"len={len(self._buffer)}, ready={self.is_ready()})"
        )

    def __iter__(self) -> Iterator[T]:
        """Iterate over items in the window (oldest → newest)."""
        return iter(self._buffer)

    def __contains__(self, item: T) -> bool:
        return item in self._buffer


# ---------------------------------------------------------------------------
# Streaming adapter — process a full sequence through the window
# ---------------------------------------------------------------------------


def stream_windows(
    items: list[T],
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
    *,
    drop_partial: bool = False,
) -> Iterator[list[T]]:
    """Convenience generator: push an entire sequence through SlidingWindow.

    Args:
        items: Full list of items to process.
        window_size: Frames per window.
        stride: Frame advance between windows.
        drop_partial: If True, skip the final window when it is shorter than
                      *window_size*.  If False (default), yield all windows
                      including a potentially shorter tail.

    Yields:
        List[T] for each complete window (and optionally the tail).
    """
    sw = SlidingWindow[T](window_size=window_size, stride=stride)
    for item in items:
        sw.push(item)
        if sw.is_ready():
            yield sw.get_window()
            sw.advance()

    # Tail window
    if not drop_partial and len(sw._buffer) > 0:
        yield sw.get_window()
