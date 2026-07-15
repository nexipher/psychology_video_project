"""视频帧均匀采样器。

从视频中均匀采样固定数量的帧，用于 MLLM (Qwen2.5-VL) 输入。
纯 CPU 实现（依赖 OpenCV）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FrameSampler:
    """视频帧均匀采样器。

    支持从视频文件路径或 cv2.VideoCapture 对象采样。
    """

    def __init__(self, target_width: int = 640, target_height: int = 480) -> None:
        """
        Args:
            target_width: 输出帧的统一宽度。
            target_height: 输出帧的统一高度。
        """
        self.target_width = target_width
        self.target_height = target_height

    def sample_from_path(
        self,
        video_path: str,
        num_frames: int = 16,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """从视频文件路径采样帧。

        Args:
            video_path: 视频文件路径。
            num_frames: 采样帧数。
            start_sec: 起始时间（秒）。
            end_sec: 结束时间（秒），None 表示视频末尾。

        Returns:
            RGB 帧列表 (H, W, 3)，uint8。
        """
        if not HAS_CV2:
            raise ImportError("cv2 is required for frame sampling")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        try:
            return self.sample_from_capture(cap, num_frames, start_sec, end_sec)
        finally:
            cap.release()

    def sample_from_capture(
        self,
        cap: "cv2.VideoCapture",
        num_frames: int = 16,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> List[np.ndarray]:
        """从 cv2.VideoCapture 对象采样帧。

        Args:
            cap: 已打开的 VideoCapture 对象。
            num_frames: 采样帧数。
            start_sec: 起始时间（秒）。
            end_sec: 结束时间（秒），None 表示视频末尾。

        Returns:
            RGB 帧列表 (H, W, 3)，uint8。
        """
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0  # 默认 30fps
        if end_sec is None:
            end_sec = total_frames / fps if total_frames > 0 else float("inf")

        start_frame = max(0, int(start_sec * fps))
        end_frame = (
            int(end_sec * fps)
            if end_sec < float("inf")
            else total_frames
        )
        if end_frame <= start_frame:
            end_frame = start_frame + num_frames

        # 均匀采样帧索引
        indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(frame_resized)
            else:
                # 读取失败，填充空白帧
                blank = np.zeros(
                    (self.target_height, self.target_width, 3),
                    dtype=np.uint8,
                )
                frames.append(blank)

        return frames

    def sample_time_window(
        self,
        video_path: str,
        trigger_ts: float,
        pre_sec: float = 5.0,
        post_sec: float = 10.0,
        num_frames: int = 16,
    ) -> Tuple[List[np.ndarray], float, float]:
        """围绕触发时间戳采样事件窗口。

        Args:
            video_path: 视频文件路径。
            trigger_ts: 触发时间戳（视频内秒数）。
            pre_sec: 触发前采样秒数。
            post_sec: 触发后采样秒数。
            num_frames: 均匀采样帧数。

        Returns:
            (frames, actual_start_sec, actual_end_sec)
        """
        start_sec = max(0.0, trigger_ts - pre_sec)
        end_sec = trigger_ts + post_sec
        frames = self.sample_from_path(video_path, num_frames, start_sec, end_sec)
        return frames, start_sec, end_sec
