"""视频流抽象层。

提供统一的视频输入接口，支持文件、摄像头、RTSP 流三种来源。
纯 CPU 实现（依赖 OpenCV），不依赖 GPU。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VideoStream(ABC):
    """视频流抽象基类。

    所有具体实现必须提供统一的 `read()` 接口，
    返回 (frame, timestamp) 元组。
    """

    @abstractmethod
    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """读取下一帧。

        Returns:
            (frame, timestamp) 或 None（流结束）。
            frame: (H, W, 3) RGB uint8 numpy 数组。
            timestamp: 浮点秒数，从流开始计时。
        """
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """流是否处于打开状态。"""
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """获取流的帧率。"""
        ...

    @abstractmethod
    def get_frame_count(self) -> int:
        """获取总帧数。实时流返回 -1。"""
        ...

    @abstractmethod
    def seek(self, position_sec: float) -> bool:
        """跳转到指定时间位置（秒）。实时流不支持。

        Returns:
            是否跳转成功。
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭流，释放资源。"""
        ...

    # ---- 便捷方法 ----

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        return self

    def __next__(self) -> Tuple[np.ndarray, float]:
        result = self.read()
        if result is None:
            raise StopIteration
        return result

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        """返回帧迭代器（与 __iter__ 相同）。"""
        return self


class FileVideoStream(VideoStream):
    """从视频文件读取的流。

    封装 cv2.VideoCapture，提供统一的 RGB 帧输出。
    """

    def __init__(
        self,
        file_path: str,
        target_fps: Optional[float] = None,
        target_width: int = 640,
        target_height: int = 480,
    ) -> None:
        """
        Args:
            file_path: 视频文件路径。
            target_fps: 目标帧率，None 表示使用原始帧率。
            target_width: 输出帧宽度。
            target_height: 输出帧高度。
        """
        if not HAS_CV2:
            raise ImportError("cv2 is required for FileVideoStream")

        self._path = Path(file_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {file_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0

        self._target_fps = target_fps if target_fps is not None else self._fps
        self._target_width = target_width
        self._target_height = target_height
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_index = 0
        self._start_time: Optional[float] = None

        # 帧步长（用于降采样）
        if self._target_fps < self._fps:
            self._frame_step = int(self._fps / self._target_fps)
        else:
            self._frame_step = 1

    # ---- 属性 ----

    @property
    def file_path(self) -> str:
        return str(self._path)

    @property
    def target_fps(self) -> float:
        return self._target_fps

    @property
    def native_fps(self) -> float:
        return self._fps

    # ---- VideoStream 接口 ----

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def get_fps(self) -> float:
        return self._target_fps

    def get_frame_count(self) -> int:
        if self._frame_step == 1:
            return self._total_frames
        return self._total_frames // self._frame_step

    def seek(self, position_sec: float) -> bool:
        """跳转到指定时间位置。

        Args:
            position_sec: 目标时间位置（秒）。

        Returns:
            是否跳转成功。
        """
        frame_idx = int(position_sec * self._fps)
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self._frame_index = frame_idx
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """读取下一帧（自动降采样）。

        Returns:
            (frame_rgb, timestamp) 或 None。
        """
        if not self.is_opened():
            return None

        if self._start_time is None:
            self._start_time = time.time()

        # 跳过不需要的帧（降采样）
        for _ in range(self._frame_step):
            ret = self._cap.grab()
            if not ret:
                return None
            self._frame_index += 1

        ret, frame = self._cap.retrieve()
        if not ret:
            return None

        # BGR → RGB + resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (
            frame_rgb.shape[1] != self._target_width
            or frame_rgb.shape[0] != self._target_height
        ):
            frame_rgb = cv2.resize(
                frame_rgb,
                (self._target_width, self._target_height),
                interpolation=cv2.INTER_AREA,
            )

        # 时间戳：基于帧索引计算（保证可复现）
        timestamp = self._frame_index / self._fps
        return (frame_rgb, timestamp)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._start_time = None

    def __repr__(self) -> str:
        return (
            f"FileVideoStream(path={self._path.name}, "
            f"fps={self._target_fps:.1f}/{self._fps:.1f}, "
            f"frames={self._total_frames})"
        )


class CameraStream(VideoStream):
    """从摄像头读取的实时流。

    封装 cv2.VideoCapture(device_id)，支持自定义分辨率。
    """

    def __init__(
        self,
        device_id: int = 0,
        target_fps: float = 15.0,
        target_width: int = 640,
        target_height: int = 480,
    ) -> None:
        """
        Args:
            device_id: 摄像头设备 ID（默认 0）。
            target_fps: 目标帧率。
            target_width: 输出帧宽度。
            target_height: 输出帧高度。
        """
        if not HAS_CV2:
            raise ImportError("cv2 is required for CameraStream")

        self._device_id = device_id
        self._target_fps = target_fps
        self._target_width = target_width
        self._target_height = target_height

        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open camera device: {device_id}")

        # 设置摄像头参数
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)

        self._frame_interval = 1.0 / target_fps
        self._last_read_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self._frame_count = 0

    @property
    def device_id(self) -> int:
        return self._device_id

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def get_fps(self) -> float:
        return self._target_fps

    def get_frame_count(self) -> int:
        # 实时流无固定帧数
        return -1

    def seek(self, position_sec: float) -> bool:
        # 实时流不支持跳转
        return False

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """读取下一帧（限速到 target_fps）。"""
        if not self.is_opened():
            return None

        if self._start_time is None:
            self._start_time = time.time()

        # 帧率限制
        if self._last_read_time is not None:
            elapsed = time.time() - self._last_read_time
            if elapsed < self._frame_interval:
                return None  # 未到读取间隔

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._last_read_time = time.time()
        self._frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (
            frame_rgb.shape[1] != self._target_width
            or frame_rgb.shape[0] != self._target_height
        ):
            frame_rgb = cv2.resize(
                frame_rgb,
                (self._target_width, self._target_height),
                interpolation=cv2.INTER_AREA,
            )

        timestamp = time.time() - self._start_time
        return (frame_rgb, timestamp)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._start_time = None
        self._last_read_time = None

    def __repr__(self) -> str:
        return (
            f"CameraStream(device={self._device_id}, "
            f"fps={self._target_fps}, "
            f"size={self._target_width}x{self._target_height})"
        )


class RTSPStream(VideoStream):
    """从 RTSP 网络流读取的实时流。

    封装 cv2.VideoCapture(rtsp_url)。
    内置自动重连逻辑。
    """

    _MAX_RECONNECT_ATTEMPTS = 5
    _RECONNECT_DELAY_SEC = 2.0

    def __init__(
        self,
        rtsp_url: str,
        target_fps: float = 15.0,
        target_width: int = 640,
        target_height: int = 480,
        reconnect: bool = True,
    ) -> None:
        """
        Args:
            rtsp_url: RTSP 流地址。
            target_fps: 目标帧率。
            target_width: 输出帧宽度。
            target_height: 输出帧高度。
            reconnect: 是否自动重连。
        """
        if not HAS_CV2:
            raise ImportError("cv2 is required for RTSPStream")

        self._url = rtsp_url
        self._target_fps = target_fps
        self._target_width = target_width
        self._target_height = target_height
        self._reconnect = reconnect

        self._cap = self._open_rtsp()
        self._frame_interval = 1.0 / target_fps
        self._last_read_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self._frame_count = 0
        self._reconnect_attempts = 0

    def _open_rtsp(self) -> cv2.VideoCapture:
        """打开 RTSP 连接（使用 TCP 传输以提升稳定性）。"""
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        # RTSP 优化参数
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲以减少延迟
        if not cap.isOpened():
            raise IOError(f"Cannot open RTSP stream: {self._url}")
        return cap

    def _try_reconnect(self) -> bool:
        """尝试重新连接 RTSP 流。"""
        if not self._reconnect:
            return False
        if self._reconnect_attempts >= self._MAX_RECONNECT_ATTEMPTS:
            return False

        self._reconnect_attempts += 1
        time.sleep(self._RECONNECT_DELAY_SEC)

        try:
            if self._cap is not None:
                self._cap.release()
            self._cap = self._open_rtsp()
            return True
        except IOError:
            return False

    @property
    def rtsp_url(self) -> str:
        return self._url

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def get_fps(self) -> float:
        return self._target_fps

    def get_frame_count(self) -> int:
        return -1

    def seek(self, position_sec: float) -> bool:
        return False

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """读取下一帧。断连时自动尝试重连。"""
        if not self.is_opened():
            if not self._try_reconnect():
                return None

        if self._start_time is None:
            self._start_time = time.time()

        # 帧率限制
        if self._last_read_time is not None:
            elapsed = time.time() - self._last_read_time
            if elapsed < self._frame_interval:
                return None

        ret, frame = self._cap.read()
        if not ret:
            if self._try_reconnect():
                return self.read()
            return None

        self._last_read_time = time.time()
        self._frame_count += 1
        self._reconnect_attempts = 0  # 成功读取，重置重连计数

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (
            frame_rgb.shape[1] != self._target_width
            or frame_rgb.shape[0] != self._target_height
        ):
            frame_rgb = cv2.resize(
                frame_rgb,
                (self._target_width, self._target_height),
                interpolation=cv2.INTER_AREA,
            )

        timestamp = time.time() - self._start_time
        return (frame_rgb, timestamp)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._start_time = None
        self._last_read_time = None
        self._reconnect_attempts = 0

    def __repr__(self) -> str:
        return (
            f"RTSPStream(url={self._url[:40]}..., "
            f"fps={self._target_fps})"
        )
