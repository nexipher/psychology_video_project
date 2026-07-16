"""Pytest 共享 fixtures。

提供所有 A1 测试所需的 Mock 对象和合成数据。
全部在 CPU 模式下运行，不依赖 GPU。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# Config fixtures
# ============================================================

@pytest.fixture
def test_config_path() -> str:
    """返回测试配置文件路径。"""
    return str(_PROJECT_ROOT / "configs" / "default.yaml")


@pytest.fixture
def test_config(test_config_path: str):
    """加载测试配置。"""
    from src.video_analysis.config import load_config, reset_config
    reset_config()
    cfg = load_config(test_config_path)
    yield cfg
    reset_config()


# ============================================================
# Synthetic skeleton data
# ============================================================

@pytest.fixture
def sample_keypoints_2d() -> np.ndarray:
    """生成 (1, 17, 3) 单人单帧关键点。"""
    kps = np.zeros((1, 17, 3), dtype=np.float32)
    # 近似站立姿势
    template = [
        (320, 100), (310, 90), (330, 90), (300, 95), (340, 95),
        (280, 200), (360, 200), (250, 320), (390, 320),
        (230, 440), (410, 440), (290, 380), (350, 380),
        (280, 500), (360, 500), (270, 620), (370, 620),
    ]
    for i, (x, y) in enumerate(template):
        kps[0, i] = [x, y, 0.9]
    return kps


@pytest.fixture
def sample_keypoints_sequence() -> np.ndarray:
    """生成 (T, 17, 3) 多人多帧关键点序列 (30 帧, 1 人, 15fps ≈ 2s)。"""
    T = 30
    kps = np.zeros((T, 17, 3), dtype=np.float32)
    for t in range(T):
        x_offset = t * 2  # 向右移动
        for i in range(17):
            kps[t, i] = [200 + i * 10 + x_offset, 150 + i * 15, 0.85 + 0.05 * np.random.random()]
    return kps


@pytest.fixture
def skeleton_json_file(sample_keypoints_sequence: np.ndarray) -> str:
    """创建符合 V1.2 格式的临时骨骼 JSON 文件。"""
    data = {
        "info": {"format": "skeleton_v1.2", "fps": 15},
        "data": [],
    }
    for t, frame_kps in enumerate(sample_keypoints_sequence):
        data["data"].append({
            "frame_index": t,
            "skeletons": [{
                "id": 0,
                "keypoints": frame_kps.tolist(),
            }],
        })

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def multi_person_skeleton_json() -> str:
    """创建多人骨骼 JSON (2 人, 10 帧)。"""
    data = {"info": {"format": "skeleton_v1.2"}, "data": []}
    for t in range(10):
        skeletons = []
        for pid in range(2):
            kps = [[100 + pid * 150 + i * 8 + t, 150 + i * 12, 0.9] for i in range(17)]
            skeletons.append({"id": pid, "keypoints": kps})
        data["data"].append({"frame_index": t, "skeletons": skeletons})

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


# ============================================================
# Synthetic video
# ============================================================

@pytest.fixture
def test_video_file() -> str:
    """创建合成测试视频 (60 帧, 30fps, 640x480)。"""
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))
    for i in range(60):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    yield tmp.name
    os.unlink(tmp.name)


# ============================================================
# Mock PoseEstimator
# ============================================================

@pytest.fixture
def mock_pose_estimator():
    """Mock PoseEstimator 返回固定关键点。"""
    from src.video_analysis.pose_estimator import PoseEstimator
    estimator = PoseEstimator(mode="mock")
    return estimator


# ============================================================
# PerFrameData generator
# ============================================================

@pytest.fixture
def per_frame_generator():
    """生成模拟 PerFrameData 序列的工厂函数。"""
    from src.video_analysis.data_loader import PerFrameData

    def _generate(num_frames: int = 100, fps: float = 15.0, num_persons: int = 1):
        for i in range(num_frames):
            ts = i / fps
            kps = np.zeros((num_persons, 17, 3), dtype=np.float32)
            for p in range(num_persons):
                x_base = 200 + p * 100 + i * 0.3
                for k in range(17):
                    kps[p, k] = [x_base + k * 2, 300 + k * 3 - 50, 0.9]

            bboxes = np.array([
                [x_base - 30, 150, x_base + 30, 450] for x_base in
                [200 + p * 100 + i * 0.3 for p in range(num_persons)]
            ], dtype=np.float32)

            yield PerFrameData(
                frame_index=i,
                timestamp=ts,
                image=None,
                keypoints=kps,
                track_ids=list(range(num_persons)),
                bboxes=bboxes,
                source_type="skeleton",
            )
    return _generate


# ============================================================
# Tracker fixtures
# ============================================================

@pytest.fixture
def tracker():
    from src.video_analysis.tracker import MultiObjectTracker
    return MultiObjectTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        min_hits=2,  # 降低确认阈值以便测试
        max_lost=10,
    )


# ============================================================
# Extractor fixture
# ============================================================

@pytest.fixture
def extractor():
    from src.video_analysis.feature_extractor import VideoFeatureExtractor
    return VideoFeatureExtractor(
        window_size_sec=30.0,
        window_stride_sec=10.0,
        fps=15.0,
    )


# ============================================================
# Aggregator fixture
# ============================================================

@pytest.fixture
def aggregator():
    from src.video_analysis.aggregator import DailyAggregator
    return DailyAggregator(fps=15.0)
