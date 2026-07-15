"""双模式数据加载器。

实现策略模式的 DataLoader：
  - 生产模式 (RGBVideoLoader)：读取视频文件/摄像头流 → 逐帧输出 RGB 图像
  - 验证模式 (SkeletonLoader)：读取 Toyota Smarthome Skeleton V1.2 JSON → 直接输出关键点

两种模式均返回统一的标准化的 PerFrameData，FeatureExtractor 不感知数据来源。

纯 CPU 实现（SkeletonLoader）/ 可选的 GPU 推理（RGBVideoLoader 的实际推理在 pose_estimator 中进行）。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.utils.skeleton_parser import SkeletonParser

logger = logging.getLogger(__name__)


# ============================================================
# 标准化帧数据结构
# ============================================================

@dataclass
class PerFrameData:
    """单帧标准化数据。

    所有 DataLoader 实现必须输出此结构。
    FeatureExtractor 仅依赖此结构进行特征计算。
    """

    frame_index: int                              # 帧序号（从 0 开始）
    timestamp: float                               # 时间戳（秒）
    image: Optional[np.ndarray] = None             # RGB 图像 (H, W, 3)，skeleton 模式为 None
    keypoints: Optional[np.ndarray] = None         # 关键点 (K, 3)，video 模式推理前为 None
    track_ids: Optional[List[int]] = None          # 跟踪 ID 列表（每个检测目标一个）
    bboxes: Optional[np.ndarray] = None            # 检测框 (N, 4) xyxy 格式
    source_type: str = "unknown"                   # "video" / "skeleton" / "camera"
    metadata: Dict = field(default_factory=dict)   # 额外元数据


# ============================================================
# DataLoader 抽象基类
# ============================================================

class DataLoader(ABC):
    """数据加载器抽象基类。

    所有具体实现必须提供统一的 frames() 迭代器，
    输出标准化的 PerFrameData。
    """

    @abstractmethod
    def frames(self) -> Iterator[PerFrameData]:
        """返回帧数据迭代器。

        Yields:
            PerFrameData 对象。
        """
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """获取数据流的帧率。"""
        ...

    @abstractmethod
    def get_total_frames(self) -> int:
        """获取总帧数。不可预估时返回 -1。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self.frames()


# ============================================================
# RGB 视频加载器（生产模式）
# ============================================================

class RGBVideoLoader(DataLoader):
    """从视频文件或摄像头流读取 RGB 帧。

    仅负责帧读取和预处理，不执行推理（推理在 PoseEstimator 中）。
    """

    def __init__(
        self,
        source: str,
        source_type: str = "file",
        target_fps: float = 15.0,
        target_width: int = 640,
        target_height: int = 480,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Args:
            source: 视频文件路径 或 摄像头设备 ID 字符串。
            source_type: "file" / "camera" / "rtsp"。
            target_fps: 目标帧率。
            target_width: 输出帧宽度。
            target_height: 输出帧高度。
            max_frames: 最大读取帧数，None 表示不限。
        """
        self._source = source
        self._source_type = source_type
        self._target_fps = target_fps
        self._target_width = target_width
        self._target_height = target_height
        self._max_frames = max_frames

        # 延迟导入以避免循环依赖
        from src.video_analysis.video_stream import (
            CameraStream,
            FileVideoStream,
            RTSPStream,
        )

        if source_type == "file":
            self._stream = FileVideoStream(
                source, target_fps, target_width, target_height
            )
        elif source_type == "camera":
            device_id = int(source) if source.isdigit() else 0
            self._stream = CameraStream(
                device_id, target_fps, target_width, target_height
            )
        elif source_type == "rtsp":
            self._stream = RTSPStream(
                source, target_fps, target_width, target_height
            )
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

    def frames(self) -> Iterator[PerFrameData]:
        frame_idx = 0
        for rgb_frame, ts in self._stream:
            if self._max_frames is not None and frame_idx >= self._max_frames:
                break

            yield PerFrameData(
                frame_index=frame_idx,
                timestamp=ts,
                image=rgb_frame,
                keypoints=None,      # 由 PoseEstimator 后续填充
                track_ids=None,
                bboxes=None,
                source_type=self._source_type,
                metadata={
                    "source": self._source,
                    "original_fps": self._stream.get_fps(),
                },
            )
            frame_idx += 1

    def get_fps(self) -> float:
        return self._target_fps

    def get_total_frames(self) -> int:
        native = self._stream.get_frame_count()
        if self._max_frames is not None and native > 0:
            return min(native, self._max_frames)
        return native

    def close(self) -> None:
        self._stream.close()

    def __repr__(self) -> str:
        return (
            f"RGBVideoLoader(source={self._source[:50]}, "
            f"type={self._source_type}, fps={self._target_fps})"
        )


# ============================================================
# 骨骼数据加载器（验证/测试模式）
# ============================================================

class SkeletonLoader(DataLoader):
    """从 Toyota Smarthome Skeleton V1.2 JSON 文件加载预存关键点。

    绕过神经网络推理，直接输出标准化的关键点序列。
    用于验证特征计算算法的数学正确性。
    """

    def __init__(
        self,
        skeleton_path: str,
        fps: float = 15.0,
        image_width: float = 640.0,
        image_height: float = 480.0,
        load_images: bool = False,
        image_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            skeleton_path: Skeleton JSON 文件路径。
            fps: 骨骼数据的原始帧率。
            image_width: 图像宽度（用于坐标反归一化参考）。
            image_height: 图像高度。
            load_images: 是否同步加载对应的 RGB 图像（默认不加载以节省内存）。
            image_dir: RGB 图像目录（load_images=True 时需要）。
        """
        self._skeleton_path = Path(skeleton_path)
        self._fps = fps
        self._image_width = image_width
        self._image_height = image_height
        self._load_images = load_images
        self._image_dir = Path(image_dir) if image_dir else None

        if not self._skeleton_path.exists():
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")

        self._parser = SkeletonParser(image_width, image_height)

        # 解析骨骼数据
        self._keypoints: np.ndarray = self._parser.parse_file(str(self._skeleton_path))
        # shape: (T, K, 3)
        self._total_frames = self._keypoints.shape[0]

        logger.info(
            f"SkeletonLoader loaded: {self._total_frames} frames, "
            f"{self._keypoints.shape[1]} keypoints, "
            f"fps={fps}"
        )

    @property
    def keypoints_array(self) -> np.ndarray:
        """返回完整的 (T, K, 3) 关键点数组（只读视图）。"""
        return self._keypoints

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_sec(self) -> float:
        return self._total_frames / self._fps

    def frames(self) -> Iterator[PerFrameData]:
        """逐帧迭代，输出标准化的 PerFrameData。"""
        for i in range(self._total_frames):
            ts = i / self._fps
            kps = self._keypoints[i].copy()  # (K, 3)

            # 可选：加载对应 RGB 图像
            image = None
            if self._load_images and self._image_dir:
                image = self._load_frame_image(i)

            yield PerFrameData(
                frame_index=i,
                timestamp=ts,
                image=image,
                keypoints=kps,
                track_ids=[0],  # 骨骼文件通常已绑定单一个体
                bboxes=None,
                source_type="skeleton",
                metadata={
                    "skeleton_file": str(self._skeleton_path),
                    "fps": self._fps,
                },
            )

    def _load_frame_image(self, frame_index: int) -> Optional[np.ndarray]:
        """尝试加载对应帧的 RGB 图像。"""
        if not self._image_dir:
            return None

        try:
            import cv2
        except ImportError:
            return None

        # 尝试多种命名模式
        patterns = [
            self._image_dir / f"frame_{frame_index:06d}.jpg",
            self._image_dir / f"frame_{frame_index:04d}.jpg",
            self._image_dir / f"{frame_index:06d}.jpg",
            self._image_dir / f"img_{frame_index:05d}.jpg",
        ]
        for p in patterns:
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def get_slice(
        self, start_sec: float, end_sec: float
    ) -> Tuple[np.ndarray, int, int]:
        """获取指定时间区间的关键点切片。

        Args:
            start_sec: 起始时间（秒）。
            end_sec: 结束时间（秒）。

        Returns:
            (keypoints_slice, start_frame, end_frame)
        """
        start_frame = max(0, int(start_sec * self._fps))
        end_frame = min(self._total_frames, int(end_sec * self._fps) + 1)
        return (
            self._keypoints[start_frame:end_frame].copy(),
            start_frame,
            end_frame,
        )

    def get_fps(self) -> float:
        return self._fps

    def get_total_frames(self) -> int:
        return self._total_frames

    def close(self) -> None:
        pass  # 无需释放资源

    def __repr__(self) -> str:
        return (
            f"SkeletonLoader(file={self._skeleton_path.name}, "
            f"frames={self._total_frames}, fps={self._fps})"
        )


# ============================================================
# DataLoader 工厂
# ============================================================

class DataLoaderFactory:
    """DataLoader 工厂。

    根据 source_type 自动创建对应的加载器实例。
    """

    @staticmethod
    def create(
        source: str,
        source_type: str = "file",
        **kwargs,
    ) -> DataLoader:
        """创建 DataLoader。

        Args:
            source: 数据源（文件路径 / 设备 ID / RTSP URL）。
            source_type:
              - "file" / "video" → RGBVideoLoader
              - "camera" / "webcam" → RGBVideoLoader
              - "rtsp" → RGBVideoLoader
              - "skeleton" → SkeletonLoader
            **kwargs: 传递给具体加载器的额外参数。

        Returns:
            DataLoader 实例。
        """
        if source_type in ("file", "video", "camera", "webcam", "rtsp"):
            if source_type in ("video",):
                source_type = "file"
            if source_type in ("webcam",):
                source_type = "camera"
            return RGBVideoLoader(source=source, source_type=source_type, **kwargs)

        elif source_type == "skeleton":
            return SkeletonLoader(skeleton_path=source, **kwargs)

        else:
            raise ValueError(
                f"Unknown source_type: {source_type}. "
                f"Available: file, camera, rtsp, skeleton"
            )
