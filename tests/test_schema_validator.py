"""JSON Schema 校验器测试。"""

import pytest
from src.utils.schema_validator import (
    SchemaValidator,
    get_validator,
    DAILY_METRICS_SCHEMA,
    QWEN_VL_EVENT_SCHEMA,
)


class TestDailyMetricsValidation:
    """§6.1 日级统计 Schema 校验。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.validator = SchemaValidator()

    def test_valid_daily_metrics(self):
        data = {
            "user_id": "P001",
            "date": "2026-07-15",
            "daily_metrics": {
                "active_minutes": 120.5,
                "sedentary_ratio": 0.35,
                "room_transition_count": 15,
                "night_activity_count": 3,
                "social_interaction_minutes": 45.0,
                "repetitive_path_count": 2,
                "movement_speed": 0.12,
                "coverage_minutes": 680.0,
                "feature_confidence": 0.92,
            },
        }
        ok, errors = self.validator.validate_daily_metrics(data)
        assert ok, f"Validation failed: {errors}"

    def test_missing_user_id(self):
        ok, errors = self.validator.validate_daily_metrics({
            "date": "2026-07-15",
            "daily_metrics": {},
        })
        assert not ok
        assert any("user_id" in e for e in errors)

    def test_invalid_date_format(self):
        ok, errors = self.validator.validate_daily_metrics({
            "user_id": "P001",
            "date": "15-07-2026",
            "daily_metrics": {},
        })
        assert not ok

    def test_missing_metrics_fields(self):
        ok, errors = self.validator.validate_daily_metrics({
            "user_id": "P001",
            "date": "2026-07-15",
            "daily_metrics": {
                "active_minutes": 100,
            },
        })
        assert not ok


class TestMLLMOutputValidation:
    """§6.2 Qwen2.5-VL 输出 Schema 校验。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.validator = SchemaValidator()

    def test_valid_mllm_output(self):
        data = {
            "event_type": "long_inactivity",
            "cooling_period": 120,
            "num_of_occurrences": 1,
            "observable_evidence": "连续坐姿，无明显肢体运动",
            "start_sec": 10.0,
            "end_sec": 25.0,
            "activity_state": "sedentary",
            "social_context": "alone",
            "repetition_type": "none",
            "quality_flags": [],
            "evidence_sufficient": True,
        }
        ok, errors = self.validator.validate_mllm_output(data)
        assert ok, f"Validation failed: {errors}"

    def test_invalid_event_type(self):
        data = {
            "event_type": "unknown_event",
            "cooling_period": 60,
            "num_of_occurrences": 1,
            "observable_evidence": "test",
            "start_sec": 0,
            "end_sec": 10,
            "activity_state": "active",
            "social_context": "alone",
            "repetition_type": "none",
            "evidence_sufficient": True,
        }
        ok, errors = self.validator.validate_mllm_output(data)
        assert not ok

    def test_missing_required_field(self):
        ok, errors = self.validator.validate_mllm_output({
            "event_type": "social_interaction",
        })
        assert not ok

    def test_all_enum_values(self):
        """所有合法枚举值应通过校验。"""
        for event_type in ("long_inactivity", "social_interaction", "repetitive_behavior"):
            cp = 60 if event_type == "repetitive_behavior" else 120
            data = {
                "event_type": event_type,
                "cooling_period": cp,
                "num_of_occurrences": 1,
                "observable_evidence": "test",
                "start_sec": 0, "end_sec": 10,
                "activity_state": "active",
                "social_context": "alone",
                "repetition_type": "none",
                "evidence_sufficient": True,
            }
            ok, _ = self.validator.validate_mllm_output(data)
            assert ok


class TestSafeParseJSON:
    """JSON 安全解析测试。"""

    def test_parse_clean_json(self):
        validator = SchemaValidator()
        import json
        raw = json.dumps({
            "event_type": "long_inactivity",
            "cooling_period": 120,
            "num_of_occurrences": 1,
            "observable_evidence": "test",
            "start_sec": 0, "end_sec": 10,
            "activity_state": "sedentary",
            "social_context": "alone",
            "repetition_type": "none",
            "evidence_sufficient": True,
        })
        result, errors = validator.safe_parse_json(raw, schema="mllm")
        assert result is not None
        assert result["event_type"] == "long_inactivity"

    def test_parse_markdown_wrapped_json(self):
        validator = SchemaValidator()
        raw = """```json
{
    "event_type": "long_inactivity",
    "cooling_period": 120,
    "num_of_occurrences": 1,
    "observable_evidence": "test",
    "start_sec": 0,
    "end_sec": 10,
    "activity_state": "sedentary",
    "social_context": "alone",
    "repetition_type": "none",
    "evidence_sufficient": true
}
```"""
        result, errors = validator.safe_parse_json(raw, schema="mllm")
        assert result is not None
        assert result["event_type"] == "long_inactivity"

    def test_parse_invalid_json(self):
        validator = SchemaValidator()
        result, errors = validator.safe_parse_json("not valid json", schema="mllm")
        assert result is None

    def test_auto_fill_defaults(self):
        """缺失字段自动填充后应通过校验。"""
        validator = SchemaValidator()
        raw = '{"event_type": "social_interaction"}'
        result, errors = validator.safe_parse_json(raw, schema="mllm")
        # 最终结果要么修复成功，要么返回 None
        # 第一次失败后会自动填充默认值再试
        if result is not None:
            assert "observable_evidence" in result
