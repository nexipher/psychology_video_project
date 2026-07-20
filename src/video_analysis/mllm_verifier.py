"""Qwen2.5-VL-7B 事件驱动复核引擎 (A3)。

MLLMVerifier 封装 Qwen2.5-VL 的加载、推理和 JSON 输出校验。
支持事件驱动的视频片段采样 + 固定 System Prompt 模板。

两种运行模式：
  - REAL 模式：加载真实 Qwen2.5-VL-7B，GPU 推理（需用户审批）
  - MOCK 模式：返回符合 §6.2 Schema 的模拟 JSON（CPU 开发/测试）

用法:
    verifier = MLLMVerifier(mode="mock")
    result = verifier.verify(
        video_path="/path/to/video.mp4",
        event_type="long_inactivity",
        trigger_ts=120.0,
    )
    # result 严格符合 §6.2 JSON Schema
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.utils.frame_sampler import FrameSampler
from src.utils.schema_validator import SchemaValidator, get_validator

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _check_gpu() -> bool:
    if not HAS_TORCH:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_prompts(prompt_path: Optional[str] = None) -> Dict[str, Any]:
    """加载 MLLM Prompt 配置文件。"""
    if prompt_path is None:
        prompt_path = str(
            Path(__file__).resolve().parent.parent.parent / "configs" / "mllm_prompts.yaml"
        )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MLLMVerifier:
    """Qwen2.5-VL-7B 事件复核引擎。

    用法:
        verifier = MLLMVerifier(mode="mock")
        result = verifier.verify(video_path, event_type, trigger_ts)
    """

    # 每个事件类型的默认窗口
    DEFAULT_WINDOWS = {
        "long_inactivity": {"pre_sec": 5, "post_sec": 15},
        "social_interaction": {"pre_sec": 5, "post_sec": 10},
        "repetitive_behavior": {"pre_sec": 5, "post_sec": 15},
    }

    def __init__(
        self,
        mode: str = "mock",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt_path: Optional[str] = None,
        num_frames: int = 16,
        max_retries: int = 2,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (640, 480),
    ) -> None:
        """
        Args:
            mode: "mock" | "real"
            model_name: HuggingFace 模型名称或本地路径。
            prompt_path: mllm_prompts.yaml 路径，None 用默认。
            num_frames: 采样帧数（8-24）。
            max_retries: JSON 解析失败最大重试次数。
            device: 推理设备，None 自动选择。
            image_size: 帧统一尺寸 (w, h)。
        """
        if mode not in ("mock", "real"):
            raise ValueError(f"mode must be 'mock' or 'real', got: {mode}")

        self._mode = mode
        self._model_name = model_name
        self._num_frames = num_frames
        self._max_retries = max_retries
        self._device = device
        self._image_size = image_size

        # 加载 Prompt 模板
        self._prompts = _load_prompts(prompt_path)

        # 工具
        self._sampler = FrameSampler(target_width=image_size[0], target_height=image_size[1])
        self._validator = get_validator()

        # 模型占位
        self._model: Any = None
        self._processor: Any = None
        self._model_loaded = False

        if mode == "real" and not HAS_TRANSFORMERS:
            logger.warning("transformers not installed. Falling back to mock.")
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

    def load_model(self, approve_gpu: bool = False) -> None:
        """加载 Qwen2.5-VL-7B 模型。

        Args:
            approve_gpu: 用户已确认使用 GPU。

        Raises:
            RuntimeError: GPU 不可用或未审批。
            ImportError: transformers 未安装。
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for real mode. "
                "Install with: pip install transformers accelerate"
            )

        gpu_available = _check_gpu()

        if gpu_available and not approve_gpu:
            raise RuntimeError(
                "GPU detected but not approved. "
                "Please confirm GPU usage before loading model (预计 ~14GB 显存). "
                "Call load_model(approve_gpu=True) after user confirmation."
            )

        if self._device is None:
            self._device = "cuda:0" if gpu_available and approve_gpu else "cpu"

        logger.info(f"Loading {self._model_name} on {self._device} ...")

        try:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._model_name,
                torch_dtype=torch.bfloat16 if self._device != "cpu" else torch.float32,
                device_map=self._device if self._device != "cpu" else None,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def unload_model(self) -> None:
        """卸载模型释放显存。"""
        if self._model is not None:
            del self._model
            self._model = None
            self._processor = None
            self._model_loaded = False
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    # ---- 主接口 ----

    def verify(
        self,
        video_path: str,
        event_type: str,
        trigger_ts: float,
        pre_sec: Optional[float] = None,
        post_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """对视频中的事件片段进行 MLLM 复核。

        Args:
            video_path: 视频文件路径。
            event_type: "long_inactivity" | "social_interaction" | "repetitive_behavior"
            trigger_ts: 事件触发时间戳（视频内秒数）。
            pre_sec: 触发前采样秒数，None 用默认值。
            post_sec: 触发后采样秒数，None 用默认值。

        Returns:
            严格符合 §6.2 JSON Schema 的字典。
        """
        if event_type not in self._prompts:
            raise ValueError(
                f"Unknown event_type: {event_type}. "
                f"Available: {list(self._prompts.keys())}"
            )

        # 默认时间窗口
        window = self.DEFAULT_WINDOWS.get(event_type, {"pre_sec": 5, "post_sec": 15})
        if pre_sec is None:
            pre_sec = window["pre_sec"]
        if post_sec is None:
            post_sec = window["post_sec"]

        # 采样帧
        frames, actual_start, actual_end = self._sampler.sample_time_window(
            video_path, trigger_ts, pre_sec, post_sec, self._num_frames,
        )

        # 获取 System Prompt 模板
        prompt_config = self._prompts[event_type]
        system_prompt = prompt_config["system"]

        # 推理
        if self.is_real:
            raw_json = self._inference_real(frames, system_prompt)
        else:
            raw_json = self._inference_mock(event_type)

        # 解析 + 校验
        data, errors = self._validator.safe_parse_json(
            raw_json, schema="mllm", max_retries=self._max_retries,
        )

        if data is None:
            # 多次重试后仍失败，返回安全默认值
            logger.warning(f"MLLM JSON parse failed after retries: {errors}")
            data = self._safe_default(event_type, actual_start, actual_end)

        # 确保时间戳来自实际采样窗口
        data["start_sec"] = round(actual_start, 1)
        data["end_sec"] = round(actual_end, 1)

        # 最终校验
        is_valid, final_errors = self._validator.validate_mllm_output(data)
        if not is_valid:
            logger.warning(f"Final validation failed: {final_errors}")
            data = self._safe_default(event_type, actual_start, actual_end)

        return data

    def verify_batch(
        self,
        events: List[Dict[str, Any]],
        video_path: str,
    ) -> List[Dict[str, Any]]:
        """批量复核多个事件。

        Args:
            events: [{"event_type": str, "trigger_ts": float, ...}, ...]
            video_path: 视频文件路径。

        Returns:
            每个事件的复核结果列表。
        """
        return [
            self.verify(
                video_path=video_path,
                event_type=e["event_type"],
                trigger_ts=e["trigger_ts"],
            )
            for e in events
        ]

    # ---- 推理实现 ----

    def _inference_real(self, frames: List[np.ndarray], system_prompt: str) -> str:
        """使用真实 Qwen2.5-VL 推理。"""
        # 将 numpy 帧转为 PIL Images
        pil_images = [Image.fromarray(f) for f in frames]

        # 构建对话消息（Qwen2-VL 格式）
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": "请基于以上关键帧画面，输出你的判断 JSON："},
                ],
            },
        ]

        # 应用 chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text], images=pil_images, return_tensors="pt",
        ).to(self._device)

        # 生成
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,     # 低温度确保输出稳定
                do_sample=False,     # 贪婪解码
            )

        # 解码输出
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        return raw.strip()

    def _inference_mock(self, event_type: str) -> str:
        """Mock 推理：返回预定义的模拟 JSON。"""
        import random
        random.seed(42)

        mock_responses = {
            "long_inactivity": [
                json.dumps({
                    "event_type": "long_inactivity",
                    "cooling_period": 120,
                    "num_of_occurrences": 1,
                    "observable_evidence": "老人坐在沙发上，连续坐姿超过5分钟，无明显肢体运动，桌面有书本和茶杯",
                    "analytical_summary": "老人出现长时间坐姿但伴有阅读行为，疑似积极认知活动，需要关注日常精神状态变化",
                    "start_sec": 0, "end_sec": 20,
                    "activity_state": "sedentary",
                    "social_context": "alone",
                    "repetition_type": "none",
                    "quality_flags": [],
                    "evidence_sufficient": True,
                }),
                json.dumps({
                    "event_type": "long_inactivity",
                    "cooling_period": 120,
                    "num_of_occurrences": 1,
                    "observable_evidence": "老人坐在扶手椅上，头部低垂，眼睛闭合，周围无认知活动物品",
                    "analytical_summary": "老人出现长时间闭眼低垂姿势且无认知活动迹象，疑似消极呆坐/打盹行为，需要关注日常活动量是否下降",
                    "start_sec": 0, "end_sec": 20,
                    "activity_state": "sedentary",
                    "social_context": "alone",
                    "repetition_type": "none",
                    "quality_flags": [],
                    "evidence_sufficient": True,
                }),
            ],
            "social_interaction": [
                json.dumps({
                    "event_type": "social_interaction",
                    "cooling_period": 120,
                    "num_of_occurrences": 1,
                    "observable_evidence": "两人面对面坐在餐桌两侧，正在交谈，老人表情放松，茶几上有茶杯",
                    "analytical_summary": "老人出现多人共处场景且互动氛围放松，疑似正常家庭互动，需要关注社交频率是否异常增加或减少",
                    "start_sec": 0, "end_sec": 15,
                    "activity_state": "sedentary",
                    "social_context": "interacting",
                    "repetition_type": "none",
                    "quality_flags": [],
                    "evidence_sufficient": True,
                }),
                json.dumps({
                    "event_type": "social_interaction",
                    "cooling_period": 120,
                    "num_of_occurrences": 1,
                    "observable_evidence": "陌生人站在门口，手持文件夹向老人展示，老人站立姿态拘谨",
                    "analytical_summary": "老人出现在门口与手持文件者互动的拘谨场景，疑似陌生人推销/诈骗接触，需要关注财产安全风险",
                    "start_sec": 0, "end_sec": 15,
                    "activity_state": "active",
                    "social_context": "interacting",
                    "repetition_type": "none",
                    "quality_flags": [],
                    "evidence_sufficient": True,
                }),
            ],
            "repetitive_behavior": [
                json.dumps({
                    "event_type": "repetitive_behavior",
                    "cooling_period": 60,
                    "num_of_occurrences": 1,
                    "observable_evidence": "老人在客厅与玄关之间来回走动5次，未接触任何物品，步伐缓慢",
                    "analytical_summary": "老人出现固定路线反复走动且未接触物品，疑似无目的徘徊行为，需要关注是否存在认知功能变化",
                    "start_sec": 0, "end_sec": 20,
                    "activity_state": "active",
                    "social_context": "alone",
                    "repetition_type": "same_route",
                    "quality_flags": [],
                    "evidence_sufficient": True,
                }),
                json.dumps({
                    "event_type": "repetitive_behavior",
                    "cooling_period": 60,
                    "num_of_occurrences": 1,
                    "observable_evidence": "老人反复打开同一个抽屉4次，每次短暂查看后关上",
                    "analytical_summary": "老人出现对同一位置反复翻找行为，疑似强迫性检查/记忆减退表现，需要关注日常记忆功能变化",
                    "start_sec": 0, "end_sec": 20,
                    "activity_state": "active",
                    "social_context": "alone",
                    "repetition_type": "repeated_search",
                    "quality_flags": [],
                    "evidence_sufficient": True,
                }),
            ],
        }

        responses = mock_responses.get(event_type, [json.dumps({
            "event_type": event_type,
            "cooling_period": 60,
            "num_of_occurrences": 1,
            "observable_evidence": "No evidence available (mock)",
            "analytical_summary": "MLLM无法从当前画面中提取足够证据，无法做出有效判断",
            "start_sec": 0, "end_sec": 10,
            "activity_state": "uncertain",
            "social_context": "uncertain",
            "repetition_type": "uncertain",
            "quality_flags": [],
            "evidence_sufficient": False,
        })])
        return random.choice(responses)

    # ---- 安全默认值 ----

    def _safe_default(
        self, event_type: str, start_sec: float, end_sec: float,
    ) -> Dict[str, Any]:
        """JSON 解析失败时的安全兜底。"""
        return {
            "event_type": event_type,
            "cooling_period": 60 if event_type == "repetitive_behavior" else 120,
            "num_of_occurrences": 1,
            "observable_evidence": "MLLM output could not be parsed",
            "analytical_summary": "MLLM输出解析失败，无法生成分析总结，建议人工复核",
            "start_sec": round(start_sec, 1),
            "end_sec": round(end_sec, 1),
            "activity_state": "uncertain",
            "social_context": "uncertain",
            "repetition_type": "uncertain",
            "quality_flags": [],
            "evidence_sufficient": False,
        }

    def __repr__(self) -> str:
        return (
            f"MLLMVerifier(mode={self._mode}, model={self._model_name}, "
            f"frames={self._num_frames}, loaded={self._model_loaded})"
        )


# ============================================================
# 事件触发集成
# ============================================================

def generate_mllm_triggers(
    a2_daily_summary: Dict[str, Any],
    a2_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    min_confidence: float = 0.3,
) -> List[Dict[str, Any]]:
    """扫描 A2 检测器输出，生成需要 MLLM 复核的事件列表。

    Args:
        a2_daily_summary: SpecialBehaviorDetector.get_daily_summary() 的输出。
        a2_events: 可选，各检测器的原始事件列表（包含 timestamp）。
        min_confidence: 最低置信度阈值，低于此值不触发。

    Returns:
        [{"event_type": str, "trigger_ts": float, "reason": str}, ...]
        按优先级排序（反复行为 > 社交异常 > 久坐）。
    """
    triggers: List[Dict[str, Any]] = []

    # 1. 徘徊/重复行为 → repetitive_behavior
    rep_count = a2_daily_summary.get("daily_repetitive_path_count", 0)
    hotspot_count = a2_daily_summary.get("daily_hotspot_action_count", 0)
    if rep_count > 0 or hotspot_count > 0:
        ts = _find_event_ts(a2_events, "repetitive_path") if a2_events else 0.0
        triggers.append({
            "event_type": "repetitive_behavior",
            "trigger_ts": ts,
            "reason": f"徘徊事件={rep_count}, 热点动作={hotspot_count}",
            "priority": 1,
        })

    # 2. 社交异常 → social_interaction
    social_intensity = a2_daily_summary.get("daily_avg_social_intensity", 0)
    if social_intensity > 0.3:
        ts = _find_event_ts(a2_events, "social_interaction") if a2_events else 0.0
        triggers.append({
            "event_type": "social_interaction",
            "trigger_ts": ts,
            "reason": f"社交强度={social_intensity:.2f}",
            "priority": 2,
        })

    # 3. 久坐/静止异常 → long_inactivity
    prolonged_count = a2_daily_summary.get("daily_prolonged_inactive_count", 0)
    max_stretch = a2_daily_summary.get("max_inactive_stretch_sec", 0)
    if prolonged_count > 0 or max_stretch > 3600:
        ts = _find_event_ts(a2_events, "prolonged_inactivity") if a2_events else 0.0
        triggers.append({
            "event_type": "long_inactivity",
            "trigger_ts": ts,
            "reason": f"久坐事件={prolonged_count}, 最长静止={max_stretch:.0f}s",
            "priority": 3,
        })

    # 按优先级排序
    triggers.sort(key=lambda t: t["priority"])
    return triggers


def _find_event_ts(
    a2_events: Optional[Dict[str, List[Dict[str, Any]]]],
    event_key: str,
    default_ts: float = 0.0,
) -> float:
    """从 A2 事件列表中提取最近一次事件的时间戳。"""
    if a2_events is None:
        return default_ts
    events = a2_events.get(event_key, [])
    if not events:
        return default_ts
    return float(events[-1].get("timestamp", default_ts))
