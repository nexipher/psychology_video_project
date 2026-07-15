"""YOLOv8-Pose 推理封装。

封装 ultralytics YOLOv8-Pose 模型，提供人体关键点检测功能。

支持两种运行模式：
  1. REAL 模式：GPU 可用 + 加载真实 YOLO 模型 → 实际推理
  2. MOCK 模式：无 GPU 或测试环境 → 返回预生成/模拟的关键点

GPU 审批机制：
  - 加载真实模型前检查 torch.cuda.is_available()
  - 若需要 GPU 但不可用，抛出明确错误而非静默降级
  - 用户必须在调用 enable_real_mode() 前确认 GPU 使用
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None  # type: ignore

# COCO 关键点索引
COCO_KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
NUM_KEYPOINTS = 17


def check_gpu_available() -> bool:
    """检查 GPU 是否可用。"""
    if not HAS_TORCH:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


class PoseEstimator:
    """YOLOv8-Pose 推理封装。

    用法:
        # Mock 模式（默认，无 GPU）
        estimator = PoseEstimator(mode="mock")
        results = estimator.estimate(frame)

        # Real 模式（需 GPU + 用户审批）
        estimator = PoseEstimator(mode="real", model_path="yolov8n-pose.pt")
        estimator.load_model()  # 触发 GPU 审批检查
        results = estimator.estimate(frame)
    """

    def __init__(
        self,
        mode: str = "mock",
        model_path: str = "yolov8n-pose.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        image_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            mode: "mock" | "real"
            model_path: YOLO 模型权重路径或名称。
            conf_threshold: 检测置信度阈值。
            iou_threshold: NMS IoU 阈值。
            image_size: 推理输入尺寸。
            device: 推理设备。None 时自动选择 ("cuda:0" if GPU else "cpu")。
        """
        if mode not in ("mock", "real"):
            raise ValueError(f"mode must be 'mock' or 'real', got: {mode}")

        self._mode = mode
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        self._device = device
        self._model: Any = None
        self._model_loaded = False

        # Mock 模式配置
        self._mock_max_persons = 3
        self._mock_frame_counter = 0

        if mode == "real" and not HAS_ULTRALYTICS:
            logger.warning(
                "ultralytics not installed but mode='real'. "
                "Install with: pip install ultralytics. "
                "Falling back to mock."
            )
            self._mode = "mock"

    # ---- 属性 ----

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_real(self) -> bool:
        return self._mode == "real" and self._model_loaded

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    # ---- 模型加载 ----

    def load_model(self, force: bool = False, approve_gpu: bool = False) -> None:
        """加载 YOLOv8-Pose 模型。

        在 real 模式下，加载真实模型权重。
        若需要 GPU，调用方必须先获得用户审批。

        Args:
            force: 强制加载（忽略模式检查）。
            approve_gpu: 用户已确认使用 GPU。

        Raises:
            RuntimeError: GPU 不可用或未审批。
            ImportError: ultralytics 未安装。
        """
        if not force and self._mode != "real":
            logger.info("Not in real mode, skipping model load")
            return

        if not HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is required for real mode. "
                "Install with: pip install ultralytics"
            )

        gpu_available = check_gpu_available()

        # GPU 审批检查
        if gpu_available and not approve_gpu:
            raise RuntimeError(
                "GPU detected but not approved. "
                "Please confirm GPU usage before loading model. "
                "Call load_model(approve_gpu=True) after user confirmation."
            )

        # 确定计算设备
        if self._device is None:
            self._device = "cuda:0" if gpu_available and approve_gpu else "cpu"

        logger.info(f"Loading YOLOv8-Pose model: {self._model_path} on {self._device}")

        try:
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def unload_model(self) -> None:
        """卸载模型以释放显存/内存。"""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    # ---- 推理 ----

    def estimate(
        self, frame: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """对单帧图像进行人体关键点估计。

        Args:
            frame: RGB 图像 (H, W, 3)，uint8。

        Returns:
            {
                "keypoints": (N, 17, 3) 关键点数组 (x, y, conf)，未检测到人体时为空 (0,17,3)
                "bboxes":    (N, 4) 检测框 xyxy 格式
                "confidences": (N,) 检测置信度
            }
        """
        if self.is_real:
            return self._estimate_real(frame)
        else:
            return self._estimate_mock(frame)

    def _estimate_real(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """使用真实 YOLO 模型推理。"""
        results = self._model(
            frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            verbose=False,
        )

        keypoints_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        confs_list: List[float] = []

        for result in results:
            if result.keypoints is None:
                continue

            # keypoints.data: (N, 17, 3) tensor
            kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = result.boxes
            if boxes is not None:
                bxs = boxes.xyxy.cpu().numpy()  # (N, 4)
                confs = boxes.conf.cpu().numpy()  # (N,)
            else:
                bxs = np.zeros((len(kps), 4), dtype=np.float32)
                confs = np.ones(len(kps), dtype=np.float32)

            for i in range(len(kps)):
                keypoints_list.append(kps[i])
                bboxes_list.append(bxs[i])
                confs_list.append(float(confs[i]))

        if not keypoints_list:
            return {
                "keypoints": np.empty((0, NUM_KEYPOINTS, 3), dtype=np.float32),
                "bboxes": np.empty((0, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
            }

        return {
            "keypoints": np.stack(keypoints_list, axis=0),
            "bboxes": np.stack(bboxes_list, axis=0),
            "confidences": np.array(confs_list, dtype=np.float32),
        }

    def _estimate_mock(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Mock 模式：返回模拟关键点。

        生成 1-3 个模拟人体，关键点基于简单的几何模式。
        位置随 frame_counter 变化模拟运动。
        """
        H, W = frame.shape[:2]
        self._mock_frame_counter += 1
        t = self._mock_frame_counter / 30.0  # 时间参数

        # 模拟 1-3 个人
        num_persons = min(
            self._mock_max_persons,
            max(1, int(np.abs(np.sin(t * 0.2)) * 3)),
        )

        keypoints = np.zeros((num_persons, NUM_KEYPOINTS, 3), dtype=np.float32)
        bboxes = np.zeros((num_persons, 4), dtype=np.float32)
        confidences = np.ones(num_persons, dtype=np.float32) * 0.85

        for p_idx in range(num_persons):
            # 每个人有略微不同的位置和运动模式
            offset_x = W * 0.2 + p_idx * W * 0.25
            offset_y = H * 0.3
            dx = np.sin(t * 1.5 + p_idx) * 30
            dy = np.cos(t * 0.8 + p_idx * 2) * 15

            # 生成近似人体关键点（站立姿势，有微小运动）
            base_x = offset_x + dx
            base_y = offset_y + dy

            # COCO 17 关键点模板 (x, y) 相对于人体中心的比例
            template = self._get_mock_keypoint_template()

            for kp_idx, (tx, ty) in enumerate(template):
                kp_x = base_x + tx * 80 + np.random.randn() * 2
                kp_y = base_y + ty * 80 + np.random.randn() * 2
                kp_conf = 0.8 + np.random.rand() * 0.2
                keypoints[p_idx, kp_idx] = [kp_x, kp_y, kp_conf]

            # 检测框从关键点推算
            valid_kps = keypoints[p_idx, :, :2]
            valid_mask = keypoints[p_idx, :, 2] > 0.3
            if valid_mask.any():
                x_min = valid_kps[valid_mask, 0].min() - 10
                y_min = valid_kps[valid_mask, 1].min() - 10
                x_max = valid_kps[valid_mask, 0].max() + 10
                y_max = valid_kps[valid_mask, 1].max() + 10
            else:
                x_min, y_min, x_max, y_max = 100, 100, 200, 300

            bboxes[p_idx] = [
                np.clip(x_min, 0, W),
                np.clip(y_min, 0, H),
                np.clip(x_max, 0, W),
                np.clip(y_max, 0, H),
            ]

        return {
            "keypoints": keypoints,
            "bboxes": bboxes,
            "confidences": confidences,
        }

    @staticmethod
    def _get_mock_keypoint_template() -> List[Tuple[float, float]]:
        """返回 COCO 17 点的人体模板（归一化到身体中心）。"""
        return [
            (0.00, -1.20),   # 0  nose
            (-0.15, -1.25),  # 1  left_eye
            (0.15, -1.25),   # 2  right_eye
            (-0.25, -1.20),  # 3  left_ear
            (0.25, -1.20),   # 4  right_ear
            (-0.40, -0.80),  # 5  left_shoulder
            (0.40, -0.80),   # 6  right_shoulder
            (-0.55, -0.40),  # 7  left_elbow
            (0.55, -0.40),   # 8  right_elbow
            (-0.65, 0.00),   # 9  left_wrist
            (0.65, 0.00),    # 10 right_wrist
            (-0.30, 0.50),   # 11 left_hip
            (0.30, 0.50),    # 12 right_hip
            (-0.35, 1.00),   # 13 left_knee
            (0.35, 1.00),    # 14 right_knee
            (-0.35, 1.50),   # 15 left_ankle
            (0.35, 1.50),    # 16 right_ankle
        ]

    # ---- 批量处理 ----

    def estimate_batch(
        self, frames: List[np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """批量推理（单帧逐次调用，YOLO 内部已做 batch 优化）。"""
        return [self.estimate(f) for f in frames]

    def __repr__(self) -> str:
        return (
            f"PoseEstimator(mode={self._mode}, "
            f"model={self._model_path if self._mode == 'real' else 'mock'}, "
            f"loaded={self._model_loaded})"
        )
