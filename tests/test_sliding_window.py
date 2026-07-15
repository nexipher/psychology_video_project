"""Tests for 1.5 — SlidingWindow."""

from __future__ import annotations

import pytest

from src.video_analysis.sliding_window import SlidingWindow, stream_windows


class TestSlidingWindowBasics:
    def test_initial_state(self) -> None:
        sw = SlidingWindow[int](window_size=10, stride=5)
        assert len(sw) == 0
        assert not sw.is_ready()
        assert sw.total_pushes == 0

    def test_push_and_ready(self) -> None:
        sw = SlidingWindow[int](window_size=5, stride=2)
        for i in range(4):
            sw.push(i)
            assert not sw.is_ready()
        sw.push(4)  # 5th item
        assert sw.is_ready()
        assert len(sw) == 5

    def test_get_window_returns_copy(self) -> None:
        sw = SlidingWindow[int](window_size=3, stride=1)
        for i in range(3):
            sw.push(i)
        window = sw.get_window()
        assert window == [0, 1, 2]
        # Mutating the returned list should not affect internal buffer
        window.append(99)
        assert len(sw) == 3

    def test_advance(self) -> None:
        sw = SlidingWindow[int](window_size=5, stride=2)
        for i in range(5):
            sw.push(i)
        assert sw.is_ready()
        sw.advance()
        assert len(sw) == 3
        assert list(sw) == [2, 3, 4]
        assert not sw.is_ready()

    def test_full_cycle(self) -> None:
        """Simulate a real streaming scenario."""
        sw = SlidingWindow[int](window_size=4, stride=2)
        results = []
        for i in range(10):
            sw.push(i)
            if sw.is_ready():
                results.append(sw.get_window())
                sw.advance()
        # window_size=4, stride=2
        # push 0,1,2,3 → ready → [0,1,2,3] → advance → keep [2,3]
        # push 4,5       → ready → [2,3,4,5] → advance → keep [4,5]
        # push 6,7       → ready → [4,5,6,7] → advance → keep [6,7]
        # push 8,9       → ready → [6,7,8,9]
        assert results == [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5, 6, 7],
            [6, 7, 8, 9],
        ]

    def test_reset(self) -> None:
        sw = SlidingWindow[int](window_size=5, stride=2)
        for i in range(5):
            sw.push(i)
        assert sw.is_ready()
        sw.reset()
        assert len(sw) == 0
        assert not sw.is_ready()
        assert sw.total_pushes == 0

    def test_contains(self) -> None:
        sw = SlidingWindow[int](window_size=5, stride=2)
        for i in range(5):
            sw.push(i)
        assert 3 in sw
        assert 99 not in sw


class TestSlidingWindowEdgeCases:
    def test_window_size_one(self) -> None:
        sw = SlidingWindow[int](window_size=1, stride=1)
        sw.push(42)
        assert sw.is_ready()
        assert sw.get_window() == [42]

    def test_stride_larger_than_window(self) -> None:
        sw = SlidingWindow[int](window_size=3, stride=5)
        for i in range(3):
            sw.push(i)
        sw.advance()
        assert len(sw) == 0  # stride 5 > len 3 → clears everything

    def test_stride_equals_window(self) -> None:
        sw = SlidingWindow[int](window_size=4, stride=4)
        for i in range(4):
            sw.push(i)
        sw.advance()
        assert len(sw) == 0  # non-overlapping windows

    def test_push_beyond_capacity(self) -> None:
        """If caller keeps pushing without advancing, oldest items are silently evicted."""
        sw = SlidingWindow[int](window_size=3, stride=1)
        sw.push(1)
        sw.push(2)
        sw.push(3)
        sw.push(4)  # deque maxlen=3 → 1 is evicted
        assert sw.is_ready()
        assert list(sw) == [2, 3, 4]

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(window_size=0, stride=1)
        with pytest.raises(ValueError):
            SlidingWindow(window_size=-1, stride=1)

    def test_invalid_stride(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindow(window_size=5, stride=0)
        with pytest.raises(ValueError):
            SlidingWindow(window_size=5, stride=-1)


class TestSlidingWindowPerformance:
    def test_push_latency(self) -> None:
        """Push should complete in well under 1.5 ms (target << 1500 µs)."""
        sw = SlidingWindow[int](window_size=300, stride=150)
        for i in range(1000):
            sw.push(i)
            if sw.is_ready():
                sw.advance()
        # After 1000 pushes, last push should be sub-1.5ms
        assert sw.last_push_time_us < 1500.0
        # In practice on a modern CPU this is < 5 µs

    def test_advance_latency(self) -> None:
        sw = SlidingWindow[int](window_size=300, stride=150)
        for i in range(300):
            sw.push(i)
        assert sw.is_ready()
        sw.advance()
        assert sw.last_advance_time_us < 1500.0

    def test_many_pushes(self) -> None:
        """Stress-test: 100 000 pushes should complete quickly."""
        sw = SlidingWindow[int](window_size=300, stride=150)
        for i in range(100_000):
            sw.push(i)
            if sw.is_ready():
                sw.advance()
        assert sw.total_pushes == 100_000


class TestStreamWindows:
    def test_basic(self) -> None:
        items = list(range(10))
        windows = list(stream_windows(items, window_size=3, stride=3))
        assert windows == [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9],  # tail
        ]

    def test_drop_partial(self) -> None:
        items = list(range(10))
        windows = list(
            stream_windows(items, window_size=3, stride=3, drop_partial=True)
        )
        assert windows == [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]

    def test_empty_input(self) -> None:
        assert list(stream_windows([], window_size=5, stride=2)) == []

    def test_fewer_items_than_window(self) -> None:
        windows = list(stream_windows([1, 2], window_size=5, stride=2))
        assert windows == [[1, 2]]  # tail only

    def test_large_sequence(self) -> None:
        items = list(range(1000))
        windows = list(stream_windows(items, window_size=100, stride=50))
        # 1000 items, window=100, stride=50 → 19 full + 1 tail = 20 windows
        assert len(windows) == 20
        assert len(windows[-1]) == 50  # tail window (950..999)
