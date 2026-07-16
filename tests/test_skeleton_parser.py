"""骨骼解析器测试。A1.2"""

import numpy as np
import pytest
from src.utils.skeleton_parser import (
    SkeletonParser,
    normalize_keypoints,
    filter_low_confidence,
    COCO_KEYPOINT_NAMES,
    COCO_NUM_KEYPOINTS,
)


class TestSkeletonParser:
    """SkeletonParser 核心功能测试。"""

    def test_parse_standard_format(self, skeleton_json_file, sample_keypoints_sequence):
        parser = SkeletonParser()
        result = parser.parse_file(skeleton_json_file)
        assert result.shape == sample_keypoints_sequence.shape
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, sample_keypoints_sequence, decimal=1)

    def test_parse_empty_data_raises(self):
        parser = SkeletonParser()
        with pytest.raises(ValueError):
            parser.parse({"info": {}, "data": []})

    def test_parse_invalid_raises(self):
        parser = SkeletonParser()
        with pytest.raises(ValueError):
            parser.parse({"info": {}})

    def test_parse_list_format(self):
        """直接列表格式。"""
        parser = SkeletonParser()
        data = [
            {"frame_index": 0, "skeletons": [{"id": 0, "keypoints": [[i, i + 1, 0.8] for i in range(17)]}]},
        ]
        result = parser.parse(data)
        assert result.shape == (1, 17, 3)

    def test_parse_2d_coordinates(self):
        """2D 坐标应自动补充置信度。"""
        parser = SkeletonParser()
        data = [{"keypoints": [[100.0, 200.0] for _ in range(17)]}]
        result = parser.parse(data)
        assert result.shape == (1, 17, 3)
        assert result[0, 0, 2] == 1.0  # 默认置信度

    def test_parse_fewer_keypoints_pads(self):
        """关键点不足 17 个时自动补零。"""
        parser = SkeletonParser()
        data = [{"keypoints": [[1.0, 2.0, 0.9] for _ in range(10)]}]
        result = parser.parse(data)
        assert result.shape == (1, 17, 3)
        # 后 7 个关键点应为零
        assert np.all(result[0, 10:, :] == 0.0)

    def test_get_centroid_sequence(self):
        parser = SkeletonParser()
        kps = np.zeros((5, 17, 3), dtype=np.float32)
        # 设置髋关节位置
        kps[:, 11, :] = [100, 200, 0.9]  # left_hip
        kps[:, 12, :] = [120, 200, 0.9]  # right_hip
        centroid = parser.get_centroid_sequence(kps)
        assert centroid.shape == (5, 2)
        np.testing.assert_array_almost_equal(centroid[0], [110, 200])

    def test_get_velocity_sequence(self):
        parser = SkeletonParser()
        kps = np.zeros((4, 17, 3), dtype=np.float32)
        kps[0, 11, :] = [100, 200, 0.9]
        kps[0, 12, :] = [120, 200, 0.9]
        kps[1, 11, :] = [110, 200, 0.9]
        kps[1, 12, :] = [130, 200, 0.9]
        kps[2, 11, :] = [120, 200, 0.9]
        kps[2, 12, :] = [140, 200, 0.9]
        kps[3, 11, :] = [130, 200, 0.9]
        kps[3, 12, :] = [150, 200, 0.9]
        vel = parser.get_velocity_sequence(kps, fps=15.0)
        assert vel.shape == (3,)
        # 每帧质心移动 10px，fps=15 → 150 px/s
        assert abs(vel[0] - 150.0) < 1.0


class TestUtilityFunctions:
    """辅助函数测试。"""

    def test_normalize_keypoints(self):
        kps = np.array([[[320, 240, 0.9]]], dtype=np.float32)
        norm = normalize_keypoints(kps, image_width=640, image_height=480)
        assert norm[0, 0, 0] == 0.5
        assert norm[0, 0, 1] == 0.5

    def test_filter_low_confidence(self):
        kps = np.array([[[100, 200, 0.3], [100, 200, 0.8]]], dtype=np.float32)
        filtered = filter_low_confidence(kps, threshold=0.5)
        assert filtered[0, 0, 0] == 0.0  # 低置信度归零
        assert filtered[0, 1, 0] == 100.0  # 高置信度保留

    def test_coco_keypoints(self):
        assert len(COCO_KEYPOINT_NAMES) == COCO_NUM_KEYPOINTS
        assert COCO_KEYPOINT_NAMES[0] == "nose"
        assert COCO_KEYPOINT_NAMES[16] == "right_ankle"
