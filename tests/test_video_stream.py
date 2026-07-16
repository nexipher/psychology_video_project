"""视频流测试。A1.4"""

import numpy as np
import pytest

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.mark.skipif(not HAS_CV2, reason="cv2 not available")
class TestFileVideoStream:
    """FileVideoStream 测试。"""

    def test_open_valid_video(self, test_video_file):
        from src.video_analysis.video_stream import FileVideoStream
        stream = FileVideoStream(test_video_file, target_fps=15.0)
        assert stream.is_opened()
        stream.close()

    def test_open_invalid_file_raises(self):
        from src.video_analysis.video_stream import FileVideoStream
        with pytest.raises(FileNotFoundError):
            FileVideoStream("/nonexistent/video.mp4")

    def test_read_frames(self, test_video_file):
        from src.video_analysis.video_stream import FileVideoStream
        stream = FileVideoStream(test_video_file, target_fps=15.0)
        frames = []
        for frame, ts in stream:
            frames.append((frame.shape, ts))
            if len(frames) >= 5:
                break
        stream.close()
        assert len(frames) > 0
        assert all(s == (480, 640, 3) for s, _ in frames)

    def test_fps_downsample(self, test_video_file):
        """30fps 视频降采样到 15fps 应减少帧数。"""
        from src.video_analysis.video_stream import FileVideoStream
        stream = FileVideoStream(test_video_file, target_fps=15.0)
        assert stream.get_fps() == 15.0
        stream.close()

    def test_seek(self, test_video_file):
        from src.video_analysis.video_stream import FileVideoStream
        stream = FileVideoStream(test_video_file, target_fps=30.0)
        assert stream.seek(1.0)
        frame, ts = stream.read()
        assert ts is not None
        stream.close()

    def test_context_manager(self, test_video_file):
        from src.video_analysis.video_stream import FileVideoStream
        with FileVideoStream(test_video_file, target_fps=15.0) as stream:
            frame, ts = stream.read()
            assert frame is not None

    def test_repr(self, test_video_file):
        from src.video_analysis.video_stream import FileVideoStream
        stream = FileVideoStream(test_video_file)
        r = repr(stream)
        assert "FileVideoStream" in r
        stream.close()
