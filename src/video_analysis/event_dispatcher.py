"""A3 实时事件调度器 (Step 2)。

A3EventDispatcher 负责 A2→A3 的实时事件转发，核心职责：
1. 接收 A2 检测器的触发信号 (event_type, trigger_ts)
2. 检查冷却期状态，决定是否调用 MLLM 复核
3. 冷却期内仅累加 _pending_count，不调用 MLLM
4. MLLM 返回后用 _pending_count 覆盖 result["num_of_occurrences"]
5. 收集所有 MLLM 复核结果

用法:
    dispatcher = A3EventDispatcher(verifier, video_path="/path/to/video.mp4")
    dispatcher.on_trigger("repetitive_behavior", 120.5)   # 可能调 MLLM
    dispatcher.on_trigger("repetitive_behavior", 130.0)   # 冷却期内，仅计数
    results = dispatcher.flush()                          # 收集所有结果
"""

from __future__ import annotations

import logging
from typing import Optional

from src.video_analysis.mllm_verifier import MLLMVerifier

logger = logging.getLogger(__name__)


class A3EventDispatcher:
    """A2→A3 实时事件调度器。

    每个 event_type 独立维护冷却期状态。同类型事件在冷却期内
    仅递增 _pending_count，冷却期结束后新一轮触发才会调用 MLLM。

    设计约束:
    - YOLO 和 Qwen 必须已加载完毕（由调用方保证）
    - 调用 MLLM.verify() 时短暂阻塞（~10s），但冷却期 60-120s 才触发一次
    """

    # event_type → 冷却期（秒）
    COOLDOWN: dict[str, int] = {
        "repetitive_behavior": 60,
        "social_interaction": 120,
        "long_inactivity": 120,
    }

    def __init__(self, verifier: MLLMVerifier, video_path: str) -> None:
        """初始化调度器。

        Args:
            verifier: 已加载 Qwen 模型的 MLLMVerifier 实例。
            video_path: 视频文件路径（传给 verify() 用于帧采样）。
        """
        self._verifier = verifier
        self._video_path = video_path

        # event_type → 冷却期结束时间戳（视频内秒数）
        self._cooldown_until: dict[str, float] = {}

        # event_type → 冷却期内累计触发次数
        self._pending_count: dict[str, int] = {}

        # 已完成的 MLLM 复核结果
        self._results: list[dict] = []

    # ---- 属性 ----

    @property
    def results(self) -> list[dict]:
        """返回所有已完成的 MLLM 复核结果（只读）。"""
        return list(self._results)

    @property
    def total_mllm_calls(self) -> int:
        """实际调用 MLLM 的次数。"""
        return len(self._results)

    @property
    def total_triggers(self) -> int:
        """总触发次数（含冷却期内未调 MLLM 的次数）。"""
        return sum(self._pending_count.values()) + self.total_mllm_calls

    # ---- 主接口 ----

    def on_trigger(self, event_type: str, trigger_ts: float) -> Optional[dict]:
        """接收 A2 检测器的触发信号。

        冷却期内：递增 _pending_count，返回 None。
        冷却期外：调用 MLLM.verify()，用 _pending_count 覆盖 num_of_occurrences。

        Args:
            event_type: 触发的事件类型。
            trigger_ts: 触发时间戳（视频内秒数）。

        Returns:
            MLLM 复核结果 dict（若实际调用），否则 None。

        Raises:
            ValueError: event_type 不在 COOLDOWN 中。
        """
        if event_type not in self.COOLDOWN:
            raise ValueError(
                f"Unknown event_type: {event_type}. "
                f"Valid: {list(self.COOLDOWN.keys())}"
            )

        # 累加触发计数
        self._pending_count[event_type] = self._pending_count.get(event_type, 0) + 1

        # 检查冷却期
        now = trigger_ts
        if event_type in self._cooldown_until and now < self._cooldown_until[event_type]:
            logger.debug(
                f"[{event_type}] cooldown active (until {self._cooldown_until[event_type]:.0f}s), "
                f"pending_count={self._pending_count[event_type]}"
            )
            return None

        # 冷却期外：设置冷却期，调用 MLLM
        cooldown_sec = self.COOLDOWN[event_type]
        self._cooldown_until[event_type] = now + cooldown_sec

        occurrence_count = self._pending_count[event_type]
        logger.info(
            f"[{event_type}] trigger at {now:.1f}s, "
            f"occurrences={occurrence_count}, calling MLLM..."
        )

        # 调用 MLLM 复核
        result = self._verifier.verify(
            video_path=self._video_path,
            event_type=event_type,
            trigger_ts=trigger_ts,
        )

        # 覆盖系统侧字段
        result["cooling_period"] = cooldown_sec
        result["num_of_occurrences"] = occurrence_count

        # 重置计数器
        self._pending_count[event_type] = 0

        self._results.append(result)
        return result

    # ---- 生命周期 ----

    def flush(self) -> list[dict]:
        """返回所有已完成的 MLLM 复核结果。

        调用后内部结果列表清空（幂等）。
        """
        results = list(self._results)
        self._results.clear()
        return results

    def reset(self) -> None:
        """重置所有内部状态（冷却期、计数器、结果列表）。"""
        self._cooldown_until.clear()
        self._pending_count.clear()
        self._results.clear()
        logger.info("A3EventDispatcher reset")

    def get_cooldown_status(self, event_type: Optional[str] = None) -> dict:
        """查询冷却期状态。

        Args:
            event_type: 指定查询的事件类型，None 返回所有。

        Returns:
            {event_type: {"cooldown_until": float|None, "pending_count": int}}
        """
        types = [event_type] if event_type else list(self.COOLDOWN.keys())
        result = {}
        for et in types:
            result[et] = {
                "cooldown_until": self._cooldown_until.get(et),
                "pending_count": self._pending_count.get(et, 0),
            }
        return result

    def __repr__(self) -> str:
        return (
            f"A3EventDispatcher(mllm_calls={self.total_mllm_calls}, "
            f"total_triggers={self.total_triggers}, "
            f"cooldowns={len(self._cooldown_until)})"
        )
