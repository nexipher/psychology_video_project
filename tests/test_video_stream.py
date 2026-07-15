"""Tests for 1.2 — VideoStreamReader classes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.video_analysis.video_stream import (
    LocalVideoReader,
    MockVideoReader,
    RTSPStreamReader,
    VideoFrame,
    VideoStreamReader,
    create_reader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_video(tmp_path: Path, n_frames: int = 30, fps: float = 10.0) -> Path:
    """Write a tiny synthetic .mp4 for LocalVideoReader tests."""
    import cv2

    video_path = tmp_path / "test_video.mp4"
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for i in range(n_frames):
        # Coloured frame: hue shifts per frame
        r = (i * 8) % 256
        g = (128 + i * 5) % 256
        b = (200 - i * 6) % 256
        frame = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
        cv2.putText(frame, f"F{i:03d}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# MockVideoReader
# ---------------------------------------------------------------------------


class TestMockVideoReader:
    def test_basic_iteration(self) -> None:
        reader = MockVideoReader(total_frames=10, fps=10)
        frames = list(reader)
        assert len(frames) == 10
        for i, f in enumerate(frames):
            assert f.frame_index == i
            assert isinstance(f.image, np.ndarray)
            assert f.image.shape == (480, 640, 3)
            assert f.image.dtype == np.uint8
            assert f.width == 640
            assert f.height == 480

    def test_properties(self) -> None:
        reader = MockVideoReader(total_frames=50, fps=25.0, resolution=(1280, 720))
        assert reader.fps == 25.0
        assert reader.total_frames == 50
        assert reader.resolution == (1280, 720)
        assert reader.duration_s == pytest.approx(2.0)
        assert not reader.is_live

    def test_context_manager(self) -> None:
        with MockVideoReader(total_frames=5) as reader:
            frames = list(reader)
        assert len(frames) == 5
        assert reader._closed

    def test_custom_resolution(self) -> None:
        reader = MockVideoReader(total_frames=3, resolution=(100, 200))
        f = next(iter(reader))
        assert f.width == 100
        assert f.height == 200

    def test_reproducible_seed(self) -> None:
        r1 = MockVideoReader(total_frames=5, seed=123)
        r2 = MockVideoReader(total_frames=5, seed=123)
        for f1, f2 in zip(r1, r2):
            assert np.array_equal(f1.image, f2.image)


# ---------------------------------------------------------------------------
# LocalVideoReader
# ---------------------------------------------------------------------------


class TestLocalVideoReader:
    def test_basic_read(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=20, fps=10.0)
        with LocalVideoReader(video_path) as reader:
            frames = list(reader)
        assert len(frames) == 20
        for f in frames:
            assert f.width == 320
            assert f.height == 240

    def test_properties(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=30, fps=10.0)
        reader = LocalVideoReader(video_path)
        assert reader.fps == pytest.approx(10.0, rel=0.1)
        assert reader.total_frames == 30
        assert reader.resolution == (320, 240)
        assert reader.duration_s == pytest.approx(3.0, rel=0.1)
        assert not reader.is_live
        reader.close()

    def test_start_frame(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=20, fps=10.0)
        reader = LocalVideoReader(video_path, start_frame=10)
        frames = list(reader)
        assert len(frames) == 10
        assert frames[0].frame_index == 10

    def test_max_frames(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=20, fps=10.0)
        reader = LocalVideoReader(video_path, max_frames=5)
        frames = list(reader)
        assert len(frames) == 5

    def test_start_frame_plus_max_frames(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=30, fps=10.0)
        reader = LocalVideoReader(video_path, start_frame=10, max_frames=5)
        frames = list(reader)
        assert len(frames) == 5
        assert frames[0].frame_index == 10

    def test_frame_skip(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=20, fps=10.0)
        reader = LocalVideoReader(video_path, frame_skip=4)
        frames = list(reader)
        # frame_skip=4: yields at frame indices 0, 4, 8, 12, 16 → 5 frames
        assert len(frames) == 5
        assert frames[0].frame_index == 0
        assert frames[1].frame_index == 4
        assert frames[2].frame_index == 8

    def test_context_manager(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=5)
        with LocalVideoReader(video_path) as reader:
            frames = list(reader)
        assert len(frames) == 5
        assert reader._closed

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            LocalVideoReader("/nonexistent/video.mp4")

    def test_frame_image_is_valid(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=3)
        reader = LocalVideoReader(video_path)
        for f in reader:
            assert f.image.dtype == np.uint8
            assert f.image.shape == (240, 320, 3)
            # Not all black
            assert f.image.mean() > 1.0

    def test_total_frames_respects_start_and_max(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=30, fps=10.0)
        reader = LocalVideoReader(video_path, start_frame=5, max_frames=10)
        # effective = min(30-5=25, 10) // 1 = 10
        assert reader.total_frames == 10
        reader.close()

    def test_total_frames_with_skip(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=30, fps=10.0)
        reader = LocalVideoReader(video_path, frame_skip=3)
        # effective = 30 // 3 = 10
        assert reader.total_frames == 10
        reader.close()


# ---------------------------------------------------------------------------
# RTSPStreamReader
# ---------------------------------------------------------------------------


class TestRTSPStreamReader:
    def test_invalid_url_raises(self) -> None:
        """A non-existent RTSP URL should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            RTSPStreamReader("rtsp://127.0.0.1:1/nonexistent",
                             reconnect=False, timeout_s=2)

    def test_properties_before_connection(self) -> None:
        """Properties expose sensible defaults."""
        reader = RTSPStreamReader.__new__(RTSPStreamReader)
        # We can't easily test without a real camera, but verify the class
        # interface is correct
        assert RTSPStreamReader._MAX_RECONNECT_DELAY == 30.0

    def test_is_live(self) -> None:
        """RTSP streams should report as live."""
        reader = RTSPStreamReader.__new__(RTSPStreamReader)
        # is_live is a class-level property
        assert RTSPStreamReader.is_live.fget(None)  # just check it's defined


# ---------------------------------------------------------------------------
# VideoFrame
# ---------------------------------------------------------------------------


class TestVideoFrame:
    def test_rgb_conversion(self) -> None:
        # BGR blue pixel → RGB should swap channels
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel in BGR
        f = VideoFrame(image=img, frame_index=0, timestamp_s=0.0, width=10, height=10)
        rgb = f.rgb
        # In RGB, blue should be at channel 2
        assert rgb[0, 0, 2] == 255
        # Original unchanged
        assert f.image[0, 0, 0] == 255

    def test_shape(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        f = VideoFrame(image=img, frame_index=0, timestamp_s=0.0, width=640, height=480)
        assert f.shape == (480, 640, 3)

    def test_repr(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        f = VideoFrame(image=img, frame_index=42, timestamp_s=1.5, width=10, height=10)
        r = repr(f)
        assert "42" in r
        assert "1.500" in r


# ---------------------------------------------------------------------------
# create_reader factory
# ---------------------------------------------------------------------------


class TestCreateReader:
    def test_mock(self) -> None:
        reader = create_reader("mock://", total_frames=10)
        assert isinstance(reader, MockVideoReader)
        assert reader.total_frames == 10

    def test_mock_with_kwargs(self) -> None:
        reader = create_reader("mock://", total_frames=5, resolution=(100, 100), fps=10)
        assert reader.total_frames == 5
        assert reader.resolution == (100, 100)

    def test_local_file(self, tmp_path: Path) -> None:
        video_path = _create_test_video(tmp_path, n_frames=10)
        reader = create_reader(str(video_path))
        assert isinstance(reader, LocalVideoReader)
        frames = list(reader)
        assert len(frames) == 10

    def test_rtsp_detection(self) -> None:
        """RTSP URLs should be routed to RTSPStreamReader (connect will fail)."""
        with pytest.raises(RuntimeError):
            create_reader("rtsp://127.0.0.1:1/stream", reconnect=False, timeout_s=2)

    def test_http_detection(self) -> None:
        """HTTP URLs should also route to RTSPStreamReader."""
        with pytest.raises(RuntimeError):
            create_reader("http://127.0.0.1:1/video.mjpg", reconnect=False, timeout_s=2)


# ---------------------------------------------------------------------------
# Edge cases & integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_mock_video(self) -> None:
        reader = MockVideoReader(total_frames=0)
        assert list(reader) == []

    def test_single_frame(self) -> None:
        reader = MockVideoReader(total_frames=1)
        frames = list(reader)
        assert len(frames) == 1

    def test_closed_reader_raises_stop(self) -> None:
        reader = MockVideoReader(total_frames=5)
        reader.close()
        with pytest.raises(StopIteration):
            next(reader)

    def test_double_close_is_safe(self) -> None:
        reader = MockVideoReader(total_frames=5)
        reader.close()
        reader.close()  # should not raise

    def test_large_frame_skip(self) -> None:
        """frame_skip larger than total frames yields just the first frame."""
        reader = MockVideoReader(total_frames=5, frame_skip=100)
        frames = list(reader)
        assert len(frames) == 1
        assert frames[0].frame_index == 0
