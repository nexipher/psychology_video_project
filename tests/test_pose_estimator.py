"""姿态估计器测试。A1.5"""

import numpy as np
import pytest
from src.video_analysis.pose_estimator import (
    PoseEstimator,
    check_gpu_available,
    HAS_TORCH,
    COCO_KP,
    NUM_KEYPOINTS,
)


class TestPoseEstimatorMock:
    """Mock 模式测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.estimator = PoseEstimator(mode="mock")
        self.frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_mock_mode_default(self):
        est = PoseEstimator()
        assert est.mode == "mock"
        assert not est.is_real

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            PoseEstimator(mode="invalid")

    def test_estimate_returns_correct_structure(self):
        result = self.estimator.estimate(self.frame)
        assert "keypoints" in result
        assert "bboxes" in result
        assert "confidences" in result
        assert result["keypoints"].ndim == 3
        assert result["keypoints"].shape[1] == NUM_KEYPOINTS
        assert result["keypoints"].shape[2] == 3

    def test_estimate_returns_at_least_one_person(self):
        result = self.estimator.estimate(self.frame)
        assert result["keypoints"].shape[0] >= 1

    def test_estimate_keypoints_in_bounds(self):
        result = self.estimator.estimate(self.frame)
        kps = result["keypoints"]
        H, W = self.frame.shape[:2]
        assert (kps[:, :, 0] >= 0).all()
        assert (kps[:, :, 0] <= W).all()
        assert (kps[:, :, 1] >= 0).all()
        assert (kps[:, :, 1] <= H).all()

    def test_estimate_batch(self):
        results = self.estimator.estimate_batch([self.frame, self.frame])
        assert len(results) == 2
        for r in results:
            assert "keypoints" in r

    def test_mock_produces_varying_results(self):
        """连续帧应产生变化的关键点（模拟运动）。"""
        r1 = self.estimator.estimate(self.frame)
        r2 = self.estimator.estimate(self.frame)
        # 质心应略有不同
        c1 = r1["keypoints"][0, :, :2].mean(axis=0)
        c2 = r2["keypoints"][0, :, :2].mean(axis=0)
        assert not np.allclose(c1, c2)

    def test_empty_frame_still_works(self):
        """全黑帧也应正常处理。"""
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.estimator.estimate(black)
        assert result["keypoints"].shape[0] >= 1


class TestGPUCheck:
    """GPU 检查测试。"""

    def test_check_gpu_available(self):
        result = check_gpu_available()
        assert isinstance(result, bool)

    def test_torch_import_status(self):
        assert isinstance(HAS_TORCH, bool)


class TestMockTemplate:
    """关键点模板测试。"""

    def test_template_size(self):
        template = PoseEstimator._get_mock_keypoint_template()
        assert len(template) == NUM_KEYPOINTS
        for x, y in template:
            assert isinstance(x, float)
            assert isinstance(y, float)
