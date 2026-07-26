"""多模型一致性校验与拒判机制 (A4)。

CrossValidator 对比 A2 CV 检测结果与 A3 MLLM 复核结果，
判断两者是否一致，输出最终置信度和报警等级。

设计原则:
  - 纯 CPU 计算，无状态，无需 GPU
  - MLLM 优先级高于 CV（MLLM 语义理解更可靠）
  - 证据不足时宁可漏报不误报（安全阀）

用法:
    validator = CrossValidator()
    validated = validator.validate(a2_triggers, a3_results)
    summary = validator.summarize(validated)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CrossValidator:
    """A2(CV) ↔ A3(MLLM) 双重一致性校验器。

    每个 A3 MLLM 复核结果与触发它的 A2 事件交叉验证，
    输出 confirmed / conflict / uncertain 三种裁决。

    线程安全，无内部状态。
    """

    # 每个 event_type 在 A3 中的关键判定字段
    _VALIDATION_KEY: dict[str, str] = {
        "long_inactivity": "activity_state",
        "social_interaction": "social_context",
        "repetitive_behavior": "repetition_type",
    }

    # 一致判定：A3 关键字段值属于此集合 → CV+MLLM 一致
    _CONSISTENT_VALUES: dict[str, set] = {
        "long_inactivity": {"sedentary"},                    # CV 说静坐 → MLLM 也应说静坐
        "social_interaction": {"interacting", "co_present"}, # CV 说多人 → MLLM 也应说多人
        "repetitive_behavior": {"same_route", "repeated_search"},  # CV 说重复 → MLLM 也应说重复
    }

    # 冲突判定：A3 关键字段值属于此集合 → CV 假阳性
    _CONFLICT_VALUES: dict[str, set] = {
        "long_inactivity": {"active"},             # CV 说静坐 → MLLM 说在活动
        "social_interaction": {"alone"},           # CV 说多人 → MLLM 说单人
        "repetitive_behavior": {"none"},           # CV 说重复 → MLLM 说无重复
    }

    def validate(
        self,
        a3_results: list[dict],
        a2_events: Optional[list[dict]] = None,
    ) -> list[dict]:
        """逐事件交叉校验。

        Args:
            a3_results: A3 MLLM 复核结果列表。
            a2_events: A2 触发事件历史（可选，用于补充上下文）。
                       如果为 None，仅基于 a3_results 内部字段判断。

        Returns:
            validated 列表，每个 MLLM 结果附加 verdict + confidence + alert。
        """
        validated = []
        for i, a3 in enumerate(a3_results):
            event_type = a3.get("event_type", "")
            evidence_sufficient = a3.get("evidence_sufficient", False)
            quality_ok = len(a3.get("quality_flags", [])) == 0

            # 查找对应的 A2 事件（如果有）
            a2_signal = None
            if a2_events and i < len(a2_events):
                a2_signal = self._extract_a2_signal(a2_events[i])

            # 逐条件判定
            verdict, confidence, detail = self._evaluate(
                event_type, a3, evidence_sufficient, quality_ok, a2_signal,
            )

            validated.append({
                "verdict": verdict,                     # confirmed | conflict | uncertain
                "confidence": round(confidence, 2),     # 0-1
                "detail": detail,                       # 中文解释
                "a2_signal": a2_signal or {},           # A2 触发源信息（A3 中不包含）
                "recommend_alert": verdict == "confirmed" and evidence_sufficient,
                "alert_level": self._alert_level(verdict, evidence_sufficient),
            })
        return validated

    # ---- 核心判定 ----

    def _evaluate(
        self,
        event_type: str,
        a3: dict,
        evidence_sufficient: bool,
        quality_ok: bool,
        a2_signal: Optional[dict],
    ) -> tuple[str, float, str]:
        """逐条裁决一个 MLLM 结果。"""
        key = self._VALIDATION_KEY.get(event_type)
        key_value = a3.get(key) if key else None

        # 1. 优先拒判条件
        if not evidence_sufficient:
            return "uncertain", 0.3, "MLLM 判定画面证据不足，无法做出有效判断"

        if not quality_ok:
            flags = a3.get("quality_flags", [])
            return "uncertain", 0.35, f"画面质量问题: {', '.join(flags)}"

        if key and key_value == "uncertain":
            return "uncertain", 0.35, f"MLLM 对 {key}(={key_value}) 判断为不确定"

        # 2. 一致性判定
        if key and key_value in self._CONSISTENT_VALUES.get(event_type, set()):
            return "confirmed", self._confidence(True, True), (
                f"CV+MLLM 一致：{key}={key_value}，"
                f"证据: {a3.get('observable_evidence', 'N/A')[:80]}"
            )

        # 3. 冲突判定
        if key and key_value in self._CONFLICT_VALUES.get(event_type, set()):
            detail = (
                f"CV 检测到异常但 MLLM 不确认：{key}={key_value}。"
                f"证据: {a3.get('observable_evidence', 'N/A')[:80]}"
            )
            # social_interaction 的 alone 是经典假阳性场景
            if event_type == "social_interaction" and key_value == "alone":
                detail += "（CV 多人检测为假阳性，单人视频常见）"
            return "conflict", self._confidence(True, False), detail

        # 4. 兜底
        return "uncertain", 0.35, (
            f"关键字段 {key}={key_value} 不在既定的确认/冲突列表中"
        )

    @staticmethod
    def _confidence(evidence_ok: bool, is_confirmed: bool) -> float:
        """置信度计算公式。"""
        base = 0.7 if evidence_ok else 0.3
        if is_confirmed:
            return min(0.95, base + 0.2)
        else:
            return max(0.20, base - 0.3)

    @staticmethod
    def _alert_level(verdict: str, evidence_sufficient: bool) -> Optional[str]:
        """报警等级。"""
        if verdict == "confirmed" and evidence_sufficient:
            return "medium"
        elif verdict == "conflict":
            return "low"
        return None  # uncertain → 不报警

    @staticmethod
    def _extract_a2_signal(a2_event: dict) -> dict:
        """从 A2 事件中提取关键信号。"""
        return {
            k: v for k, v in a2_event.items()
            if k in ("event_type", "reason", "source", "hotspot_count",
                     "social_intensity", "inactive_stretch_sec")
        } if a2_event else {}

    # ---- 汇总 ----

    def summarize(self, validated: list[dict]) -> dict:
        """对所有验证结果做整体评估。"""
        total = len(validated)
        if total == 0:
            return {
                "overall_status": "normal",
                "total_events": 0,
                "confirmed": 0,
                "conflict": 0,
                "uncertain": 0,
                "recommendation": "无异常事件",
            }

        counts = {"confirmed": 0, "conflict": 0, "uncertain": 0}
        for v in validated:
            counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1

        confirmed_ratio = counts["confirmed"] / total
        conflict_ratio = counts["conflict"] / total

        if confirmed_ratio >= 0.7:
            overall = "alert"
            rec = f"{counts['confirmed']}/{total} 事件经 MLLM 确认属实，建议关注"
        elif conflict_ratio >= 0.5:
            overall = "normal"
            rec = f"大部分事件（{counts['conflict']}/{total}）为 CV 假阳性，系统运行正常"
        elif counts["uncertain"] >= total * 0.5:
            overall = "uncertain"
            rec = "大量事件证据不足，建议排查摄像头位置或光照条件"
        else:
            overall = "caution"
            rec = "事件判断结果混合，建议人工复核"

        return {
            "overall_status": overall,
            "total_events": total,
            "confirmed": counts["confirmed"],
            "conflict": counts["conflict"],
            "uncertain": counts["uncertain"],
            "recommendation": rec,
        }

    def __repr__(self) -> str:
        return "CrossValidator()"
