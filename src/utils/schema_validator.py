"""JSON Schema 校验工具。

对模块输出进行格式校验，确保数据接口符合 §6.1 和 §6.2 规范。
纯 CPU 实现。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# ============================================================
# §6.1 日级统计输出 Schema
# ============================================================
DAILY_METRICS_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Daily Metrics Output",
    "type": "object",
    "properties": {
        "user_id": {"type": "string"},
        "date": {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
            "description": "YYYY-MM-DD",
        },
        "daily_metrics": {
            "type": "object",
            "properties": {
                "active_minutes": {"type": "number", "minimum": 0},
                "sedentary_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "room_transition_count": {"type": "integer", "minimum": 0},
                "night_activity_count": {"type": "integer", "minimum": 0},
                "social_interaction_minutes": {"type": "number", "minimum": 0},
                "repetitive_path_count": {"type": "integer", "minimum": 0},
                "movement_speed": {"type": "number", "minimum": 0},
                "coverage_minutes": {"type": "number", "minimum": 0},
                "feature_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "active_minutes",
                "sedentary_ratio",
                "room_transition_count",
                "night_activity_count",
                "social_interaction_minutes",
                "repetitive_path_count",
                "movement_speed",
                "coverage_minutes",
                "feature_confidence",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["user_id", "date", "daily_metrics"],
    "additionalProperties": False,
}

# ============================================================
# §6.2 Qwen2.5-VL 事件复核输出 Schema
# ============================================================
QWEN_VL_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Qwen2.5_VL_Event_Verification",
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": ["long_inactivity", "social_interaction", "repetitive_behavior"],
        },
        "observable_evidence": {
            "type": "string",
            "description": "只描述画面可见事实",
        },
        "start_sec": {"type": "number"},
        "end_sec": {"type": "number"},
        "activity_state": {
            "type": "string",
            "enum": ["active", "sedentary", "uncertain"],
        },
        "social_context": {
            "type": "string",
            "enum": ["alone", "co_present", "interacting", "uncertain"],
        },
        "repetition_type": {
            "type": "string",
            "enum": ["same_route", "repeated_search", "none", "uncertain"],
        },
        "quality_flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["occlusion", "low_light", "off_camera"],
            },
        },
        "evidence_sufficient": {"type": "boolean"},
    },
    "required": [
        "event_type",
        "observable_evidence",
        "start_sec",
        "end_sec",
        "activity_state",
        "social_context",
        "repetition_type",
        "evidence_sufficient",
    ],
    "additionalProperties": False,
}


class SchemaValidator:
    """JSON Schema 校验器。

    对系统输出进行格式和字段校验，确保符合接口规范。
    """

    def __init__(self) -> None:
        self._use_jsonschema = HAS_JSONSCHEMA

    def validate_daily_metrics(self, data: dict) -> Tuple[bool, List[str]]:
        """校验日级统计输出是否符合 §6.1 Schema。

        Args:
            data: 待校验的日级统计字典。

        Returns:
            (is_valid, error_messages)
        """
        errors: List[str] = []

        # 基本字段存在性检查
        for field in ("user_id", "date", "daily_metrics"):
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if "date" in data:
            import re
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(data["date"])):
                errors.append(f"Invalid date format: {data['date']} (expected YYYY-MM-DD)")

        if "daily_metrics" in data and isinstance(data["daily_metrics"], dict):
            metrics = data["daily_metrics"]
            required_metrics = [
                "active_minutes",
                "sedentary_ratio",
                "room_transition_count",
                "night_activity_count",
                "social_interaction_minutes",
                "repetitive_path_count",
                "movement_speed",
                "coverage_minutes",
                "feature_confidence",
            ]
            for m in required_metrics:
                if m not in metrics:
                    errors.append(f"daily_metrics missing required field: {m}")

        if self._use_jsonschema:
            try:
                jsonschema.validate(instance=data, schema=DAILY_METRICS_SCHEMA)
            except jsonschema.ValidationError as e:
                errors.append(f"jsonschema validation error: {e.message}")
            except jsonschema.SchemaError as e:
                errors.append(f"Schema error: {e.message}")

        return (len(errors) == 0, errors)

    def validate_mllm_output(self, data: dict) -> Tuple[bool, List[str]]:
        """校验 Qwen2.5-VL 输出是否符合 §6.2 Schema。

        Args:
            data: 待校验的 MLLM 输出字典。

        Returns:
            (is_valid, error_messages)
        """
        errors: List[str] = []

        required_fields = [
            "event_type",
            "observable_evidence",
            "start_sec",
            "end_sec",
            "activity_state",
            "social_context",
            "repetition_type",
            "evidence_sufficient",
        ]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if not errors:
            # 枚举值检查
            if data.get("event_type") not in (
                "long_inactivity",
                "social_interaction",
                "repetitive_behavior",
            ):
                errors.append(
                    f"Invalid event_type: {data.get('event_type')}"
                )

            if data.get("activity_state") not in ("active", "sedentary", "uncertain"):
                errors.append(
                    f"Invalid activity_state: {data.get('activity_state')}"
                )

            if data.get("social_context") not in (
                "alone", "co_present", "interacting", "uncertain",
            ):
                errors.append(
                    f"Invalid social_context: {data.get('social_context')}"
                )

            if data.get("repetition_type") not in (
                "same_route", "repeated_search", "none", "uncertain",
            ):
                errors.append(
                    f"Invalid repetition_type: {data.get('repetition_type')}"
                )

            if not isinstance(data.get("evidence_sufficient"), bool):
                errors.append(
                    f"evidence_sufficient must be boolean, got: {type(data.get('evidence_sufficient'))}"
                )

        if self._use_jsonschema:
            try:
                jsonschema.validate(instance=data, schema=QWEN_VL_EVENT_SCHEMA)
            except jsonschema.ValidationError as e:
                errors.append(f"jsonschema validation error: {e.message}")
            except jsonschema.SchemaError as e:
                errors.append(f"Schema error: {e.message}")

        return (len(errors) == 0, errors)

    def safe_parse_json(
        self,
        raw: str,
        schema: str = "mllm",
        max_retries: int = 2,
    ) -> Tuple[Optional[dict], List[str]]:
        """安全解析 JSON 字符串并校验。

        Args:
            raw: 原始 JSON 字符串（可能包含 Markdown 包装）。
            schema: "daily" 或 "mllm"。
            max_retries: 最大重试次数（尝试剥离 Markdown 代码块）。

        Returns:
            (parsed_dict_or_None, error_messages)
        """
        errors: List[str] = []
        text = raw.strip()

        for attempt in range(max_retries + 1):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # 尝试剥离 Markdown 代码块
                    text = self._strip_markdown_fence(text)
                    continue
                errors.append(f"JSON parse failed after {max_retries} retries: {e}")
                return (None, errors)

            # 校验
            if schema == "daily":
                is_valid, val_errors = self.validate_daily_metrics(data)
            else:
                is_valid, val_errors = self.validate_mllm_output(data)

            if is_valid:
                return (data, [])
            else:
                errors.extend(val_errors)
                # 尝试自动修复缺失字段
                if schema == "mllm" and attempt < max_retries:
                    data = self._apply_mllm_defaults(data)
                    text = json.dumps(data)
                    continue
                return (None, errors)

        return (None, errors)

    def _strip_markdown_fence(self, text: str) -> str:
        """剥离 Markdown 代码块标记。"""
        import re
        # 移除 ```json ... ``` 或 ``` ... ```
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 尝试只移除开头/结尾的 ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _apply_mllm_defaults(self, data: dict) -> dict:
        """为缺失的 MLLM 输出字段填充安全默认值。"""
        defaults = {
            "event_type": "long_inactivity",
            "observable_evidence": "No evidence provided",
            "start_sec": 0,
            "end_sec": 0,
            "activity_state": "uncertain",
            "social_context": "uncertain",
            "repetition_type": "uncertain",
            "quality_flags": [],
            "evidence_sufficient": False,
        }
        for key, val in defaults.items():
            if key not in data:
                data[key] = val
        return data


# 模块级便捷函数
_DEFAULT_VALIDATOR: Optional[SchemaValidator] = None


def get_validator() -> SchemaValidator:
    global _DEFAULT_VALIDATOR
    if _DEFAULT_VALIDATOR is None:
        _DEFAULT_VALIDATOR = SchemaValidator()
    return _DEFAULT_VALIDATOR
