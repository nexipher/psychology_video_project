"""数据加载器测试。A1.3"""

import numpy as np
import pytest
from src.video_analysis.data_loader import (
    SkeletonLoader,
    DataLoaderFactory,
    PerFrameData,
    RGBVideoLoader,
    DataLoader,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class TestSkeletonLoader:
    """SkeletonLoader 测试。"""

    def test_load_skeleton_file(self, skeleton_json_file):
        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)
        assert loader.get_total_frames() == 30
        assert loader.get_fps() == 15.0

    def test_iterate_frames(self, skeleton_json_file):
        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)
        frames = list(loader.frames())
        assert len(frames) == 30
        for f in frames:
            assert isinstance(f, PerFrameData)
            assert f.source_type == "skeleton"
            assert f.keypoints is not None
            assert f.keypoints.shape == (17, 3)

    def test_get_slice(self, skeleton_json_file):
        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)
        kps, start_f, end_f = loader.get_slice(0.2, 0.8)
        assert kps.ndim == 3
        assert kps.shape[1] == 17
        assert kps.shape[2] == 3

    def test_keypoints_array_property(self, skeleton_json_file):
        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)
        arr = loader.keypoints_array
        assert arr.shape == (30, 17, 3)
        assert arr.dtype == np.float32

    def test_duration(self, skeleton_json_file):
        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)
        assert abs(loader.duration_sec - 2.0) < 0.1

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            SkeletonLoader(skeleton_path="/nonexistent/skel.json")

    def test_repr(self, skeleton_json_file):
        loader = SkeletonLoader(skeleton_path=skeleton_json_file, fps=15.0)
        r = repr(loader)
        assert "SkeletonLoader" in r


class TestDataLoaderFactory:
    """DataLoaderFactory 策略模式测试。"""

    def test_create_skeleton_loader(self, skeleton_json_file):
        loader = DataLoaderFactory.create(
            source=skeleton_json_file,
            source_type="skeleton",
            fps=15.0,
        )
        assert isinstance(loader, SkeletonLoader)
        assert loader.get_total_frames() == 30

    def test_create_rgb_loader(self, test_video_file):
        if not HAS_CV2:
            pytest.skip("cv2 not available")
        loader = DataLoaderFactory.create(
            source=test_video_file,
            source_type="file",
        )
        assert isinstance(loader, RGBVideoLoader)

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError):
            DataLoaderFactory.create(source="test", source_type="unknown")


class TestPerFrameData:
    """PerFrameData dataclass 测试。"""

    def test_default_values(self):
        pfd = PerFrameData(frame_index=0, timestamp=0.0)
        assert pfd.image is None
        assert pfd.keypoints is None
        assert pfd.source_type == "unknown"

    def test_full_fields(self):
        kps = np.zeros((1, 17, 3), dtype=np.float32)
        bboxes = np.array([[0, 0, 100, 200]], dtype=np.float32)
        pfd = PerFrameData(
            frame_index=5, timestamp=0.5,
            keypoints=kps, bboxes=bboxes,
            track_ids=[1], source_type="video",
        )
        assert pfd.frame_index == 5
        assert pfd.keypoints.shape == (1, 17, 3)
        assert pfd.bboxes.shape == (1, 4)
