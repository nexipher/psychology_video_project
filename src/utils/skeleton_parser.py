"""Toyota Smarthome Skeleton V1.2 格式解析器。

将 Toyota Smarthome 的骨骼 JSON 数据解析为标准化关键点张量格式，
输出 shape 为 (T, K, 3) 的 numpy 数组，其中 T=帧数, K=关键点数, 3=(x, y, confidence)。

纯 CPU 实现，不依赖 GPU。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO 17 关键点名称
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 关键点数量
COCO_NUM_KEYPOINTS = 17

# 每个关键点的坐标维度 (x, y, confidence)
COORD_DIM = 3


def normalize_keypoints(
    keypoints: np.ndarray,
    image_width: float = 640.0,
    image_height: float = 480.0,
) -> np.ndarray:
    """将像素坐标归一化到 [0, 1]。

    Args:
        keypoints: (T, K, 3) 或 (K, 3) 的关键点数组。
        image_width: 图像宽度（像素）。
        image_height: 图像高度（像素）。

    Returns:
        归一化后的关键点数组。
    """
    result = keypoints.copy().astype(np.float32)
    result[..., 0] /= image_width
    result[..., 1] /= image_height
    return result


def filter_low_confidence(
    keypoints: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """将低置信度关键点坐标置零。

    Args:
        keypoints: (T, K, 3) 的关键点数组。
        threshold: 置信度阈值。

    Returns:
        过滤后的关键点数组（低置信度坐标归零）。
    """
    mask = keypoints[..., 2:3] < threshold  # (T, K, 1)
    result = keypoints.copy()
    result[mask[..., 0]] = 0.0
    return result


class SkeletonParser:
    """Toyota Smarthome Skeleton V1.2 解析器。

    支持如下 JSON 结构变体：

    1. 标准格式:
       {
         "info": {...},
         "data": [
           {"frame_index": 0, "skeletons": [{"id": 0, "keypoints": [[x,y,c],...]}]},
           ...
         ]
       }

    2. 简化帧列表格式:
       [
         {"frame": 0, "keypoints": [[x,y,c], ...]},
         ...
       ]

    3. 平铺数组格式:
       {
         "keypoints": [[[x,y,c], ...], ...]   # shape: (T, K, 3)
       }
    """

    # 支持的顶层键名
    _DATA_KEYS = ("data", "frames", "skeletons")

    def __init__(self, image_width: float = 640.0, image_height: float = 480.0):
        self.image_width = image_width
        self.image_height = image_height

    def parse_file(self, file_path: str) -> np.ndarray:
        """从文件解析骨骼数据。

        Args:
            file_path: JSON 文件路径。

        Returns:
            (T, K, 3) 的 numpy 数组。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return self.parse(raw)

    def parse(self, raw: dict | list) -> np.ndarray:
        """从已解析的 JSON 对象中提取关键点序列。

        Args:
            raw: 已解析的 JSON dict 或 list。

        Returns:
            (T, K, 3) 的 numpy 数组。

        Raises:
            ValueError: 无法识别的格式。
        """
        # 尝试各数据键
        data = None
        if isinstance(raw, dict):
            for key in self._DATA_KEYS:
                if key in raw:
                    data = raw[key]
                    break
            if data is None:
                # 顶层可能就是直接的关键点数据
                if "keypoints" in raw:
                    data = raw["keypoints"]
                else:
                    raise ValueError(
                        f"Cannot find skeleton data. Expected one of keys: {self._DATA_KEYS} or 'keypoints'"
                    )

        if isinstance(raw, list):
            data = raw

        if data is None:
            raise ValueError("Empty skeleton data")

        return self._extract_keypoints_sequence(data)

    def _extract_keypoints_sequence(self, data: list) -> np.ndarray:
        """从数据列表中提取 (T, K, 3) 序列。

        Handles per-frame entries that each contain one or more skeleton dicts.
        """
        frames: List[np.ndarray] = []

        for entry in data:
            frame_kps = self._extract_frame_keypoints(entry)
            if frame_kps is not None:
                frames.append(frame_kps)

        if not frames:
            raise ValueError("No keypoints found in data")

        return np.stack(frames, axis=0)  # (T, K, 3)

    def _extract_frame_keypoints(self, entry) -> Optional[np.ndarray]:
        """从单帧条目中提取关键点数组 (K, 3)。

        始终返回第一个检测到的骨架。
        """
        if isinstance(entry, dict):
            # 情况1: {"skeletons": [{"keypoints": ...}, ...]}
            if "skeletons" in entry:
                skels = entry["skeletons"]
                if skels and isinstance(skels, list):
                    return self._parse_single_skeleton(skels[0])

            # 情况2: {"keypoints": [[x,y,c],...]}
            if "keypoints" in entry:
                return self._parse_single_skeleton(entry)

            # 情况3: entry 本身就是一个骨架字典 {"id": 0, "keypoints": ...}
            if "keypoints" not in entry and any(
                isinstance(v, (list, tuple)) for v in entry.values()
            ):
                # 尝试直接当作坐标列表
                pass

        if isinstance(entry, list):
            # 情况4: entry 直接是 [[x,y,c],...]
            return self._parse_keypoint_array(entry)

        return None

    def _parse_single_skeleton(self, skeleton: dict) -> np.ndarray:
        """从骨架字典中提取 (K, 3) 数组。"""
        kps = skeleton.get("keypoints")
        if kps is None:
            # 尝试 "joints" / "pose" / "points" 等常见别名
            for alias in ("joints", "pose", "points", "landmarks"):
                kps = skeleton.get(alias)
                if kps is not None:
                    break
        if kps is None:
            raise ValueError("Skeleton dict has no 'keypoints' field")
        return self._parse_keypoint_array(kps)

    def _parse_keypoint_array(self, arr: list) -> np.ndarray:
        """将二维列表转换为 (K, 3) numpy 数组，自动处理 2D/3D 坐标。"""
        kps = np.array(arr, dtype=np.float32)
        if kps.ndim != 2:
            raise ValueError(f"Expected 2D keypoint array, got shape {kps.shape}")

        K, C = kps.shape
        if C == 2:
            # 2D 坐标，补充默认置信度=1.0
            conf = np.ones((K, 1), dtype=np.float32)
            kps = np.concatenate([kps, conf], axis=1)
        elif C == 3:
            pass  # (x, y, c) 已经是目标格式
        elif C == 4:
            # (x, y, z, c) 3D 坐标，去除 z
            kps = kps[:, [0, 1, 3]]
        else:
            raise ValueError(
                f"Unexpected keypoint dimension: {C} (expected 2, 3, or 4)"
            )

        # 确保有 17 个关键点，不足则补零
        if K < COCO_NUM_KEYPOINTS:
            padded = np.zeros((COCO_NUM_KEYPOINTS, COORD_DIM), dtype=np.float32)
            padded[:K] = kps
            kps = padded
        elif K > COCO_NUM_KEYPOINTS:
            kps = kps[:COCO_NUM_KEYPOINTS]

        return kps

    def get_centroid_sequence(self, keypoints: np.ndarray) -> np.ndarray:
        """计算质心序列（基于臀部中点）。

        Args:
            keypoints: (T, K, 3) 关键点数组。

        Returns:
            (T, 2) 质心坐标序列 (x, y)。
        """
        # 左髋 (11) 和右髋 (12) 的中点
        left_hip = keypoints[:, 11, :2]
        right_hip = keypoints[:, 12, :2]

        # 仅当两个髋关节置信度均 > 0 时有效
        valid_mask = (keypoints[:, 11, 2] > 0) & (keypoints[:, 12, 2] > 0)
        centroid = (left_hip + right_hip) / 2.0
        centroid[~valid_mask] = np.nan
        return centroid

    def get_velocity_sequence(self, keypoints: np.ndarray, fps: float = 15.0) -> np.ndarray:
        """计算质心位移速度序列。

        Args:
            keypoints: (T, K, 3) 关键点数组。
            fps: 帧率。

        Returns:
            (T-1,) 速度序列（像素/秒或归一化单位/秒）。
        """
        centroid = self.get_centroid_sequence(keypoints)  # (T, 2)
        diff = np.diff(centroid, axis=0)  # (T-1, 2)
        velocity = np.linalg.norm(diff, axis=1) * fps
        return velocity
