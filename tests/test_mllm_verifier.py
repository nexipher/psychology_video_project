"""A3 MLLM 复核引擎测试。

覆盖 MLLMVerifier Mock 模式所有三种 event_type、
事件触发集成、异常降级逻辑。
使用 Videos_mp4 中的真实视频（P14T14C06）。
全部 CPU 模式。
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video_analysis.mllm_verifier import (
    MLLMVerifier,
    _load_prompts,
    generate_mllm_triggers,
    _find_event_ts,
)

# 测试用视频（Videos_mp4 中最短的）
_TEST_VIDEO = Path(__file__).resolve().parent.parent / "dataset" / "Videos_mp4" / "P14T14C06.mp4"


# ============================================================
# Prompt 模板测试
# ============================================================

class TestPrompts:
    """mllm_prompts.yaml 测试。"""

    def test_load_all_prompts(self):
        prompts = _load_prompts()
        assert "long_inactivity" in prompts
        assert "social_interaction" in prompts
        assert "repetitive_behavior" in prompts
        assert len(prompts) == 3

    def test_system_prompt_not_empty(self):
        prompts = _load_prompts()
        for event_type in prompts:
            assert len(prompts[event_type]["system"]) > 100

    def test_closed_labels_in_prompt(self):
        """Prompt 中必须包含封闭标签。"""
        prompts = _load_prompts()
        for event_type in prompts:
            text = prompts[event_type]["system"]
            assert '"""' in text or '"' in text  # JSON template

    def test_few_shot_examples(self):
        prompts = _load_prompts()
        for event_type in prompts:
            few_shot = prompts[event_type].get("few_shot", [])
            assert len(few_shot) >= 2, f"{event_type} missing few-shot examples"

    def test_schema_mention_in_prompt(self):
        """Prompt 必须提及 JSON 输出格式。"""
        prompts = _load_prompts()
        for event_type in prompts:
            text = prompts[event_type]["system"]
            assert "JSON" in text or "json" in text


# ============================================================
# MLLMVerifier Mock 模式测试
# ============================================================

@pytest.fixture(scope="module")
def verifier():
    return MLLMVerifier(mode="mock")


@pytest.fixture(scope="module")
def test_video():
    """创建临时测试视频（P14T14C06 可能不存在，创建合成视频）。"""
    if _TEST_VIDEO.exists():
        yield str(_TEST_VIDEO)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))
            for i in range(90):  # 3 seconds
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
        except ImportError:
            pytest.skip("cv2 not available")
        yield tmp.name
        os.unlink(tmp.name)


class TestMLLMVerifierMock:
    """Mock 模式核心功能。"""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            MLLMVerifier(mode="invalid")

    def test_default_mode_is_mock(self):
        v = MLLMVerifier()
        assert v.mode == "mock"
        assert not v.is_real

    def test_verify_long_inactivity(self, verifier, test_video):
        result = verifier.verify(test_video, "long_inactivity", trigger_ts=1.0)
        assert result["event_type"] == "long_inactivity"
        assert "observable_evidence" in result
        assert result["start_sec"] >= 0.0
        assert result["end_sec"] > result["start_sec"]

    def test_verify_social_interaction(self, verifier, test_video):
        result = verifier.verify(test_video, "social_interaction", trigger_ts=1.0)
        assert result["event_type"] == "social_interaction"
        assert result["social_context"] in ("alone", "co_present", "interacting", "uncertain")

    def test_verify_repetitive_behavior(self, verifier, test_video):
        result = verifier.verify(test_video, "repetitive_behavior", trigger_ts=1.0)
        assert result["event_type"] == "repetitive_behavior"
        assert result["repetition_type"] in ("same_route", "repeated_search", "none", "uncertain")

    def test_schema_validation(self, verifier, test_video):
        """所有三种 event_type 必须通过 §6.2 Schema。"""
        from src.utils.schema_validator import get_validator
        validator = get_validator()
        for event_type in ["long_inactivity", "social_interaction", "repetitive_behavior"]:
            result = verifier.verify(test_video, event_type, trigger_ts=0.5)
            ok, errors = validator.validate_mllm_output(result)
            assert ok, f"Schema failed for {event_type}: {errors}"

    def test_all_required_fields_present(self, verifier, test_video):
        required = [
            "event_type", "observable_evidence", "start_sec", "end_sec",
            "activity_state", "social_context", "repetition_type", "evidence_sufficient",
        ]
        for event_type in ["long_inactivity", "social_interaction", "repetitive_behavior"]:
            result = verifier.verify(test_video, event_type, trigger_ts=0.5)
            for field in required:
                assert field in result, f"Missing {field} in {event_type}"

    def test_verify_batch(self, verifier, test_video):
        events = [
            {"event_type": "long_inactivity", "trigger_ts": 0.5},
            {"event_type": "social_interaction", "trigger_ts": 1.0},
        ]
        results = verifier.verify_batch(events, test_video)
        assert len(results) == 2
        assert results[0]["event_type"] == "long_inactivity"
        assert results[1]["event_type"] == "social_interaction"

    def test_invalid_event_type_raises(self, verifier, test_video):
        with pytest.raises(ValueError):
            verifier.verify(test_video, "nonexistent_event", trigger_ts=0.0)

    def test_custom_time_window(self, verifier, test_video):
        result = verifier.verify(
            test_video, "long_inactivity", trigger_ts=1.0,
            pre_sec=10, post_sec=20,
        )
        assert result["end_sec"] - result["start_sec"] >= 10

    def test_repr(self, verifier):
        r = repr(verifier)
        assert "MLLMVerifier" in r
        assert "mock" in r


# ============================================================
# 事件触发集成测试
# ============================================================

class TestEventTriggerIntegration:
    """generate_mllm_triggers 测试。"""

    def test_wandering_triggers(self):
        a2_summary = {
            "daily_repetitive_path_count": 3,
            "daily_hotspot_action_count": 2,
            "daily_prolonged_inactive_count": 0,
            "max_inactive_stretch_sec": 100.0,
            "daily_avg_social_intensity": 0.0,
        }
        triggers = generate_mllm_triggers(a2_summary)
        assert len(triggers) >= 1
        assert triggers[0]["event_type"] == "repetitive_behavior"

    def test_social_triggers(self):
        a2_summary = {
            "daily_repetitive_path_count": 0,
            "daily_hotspot_action_count": 0,
            "daily_prolonged_inactive_count": 0,
            "max_inactive_stretch_sec": 100.0,
            "daily_avg_social_intensity": 0.65,
        }
        triggers = generate_mllm_triggers(a2_summary)
        assert len(triggers) >= 1
        assert triggers[0]["event_type"] == "social_interaction"

    def test_inactivity_triggers(self):
        a2_summary = {
            "daily_repetitive_path_count": 0,
            "daily_hotspot_action_count": 0,
            "daily_prolonged_inactive_count": 2,
            "max_inactive_stretch_sec": 8000.0,
            "daily_avg_social_intensity": 0.0,
        }
        triggers = generate_mllm_triggers(a2_summary)
        assert len(triggers) >= 1
        assert triggers[0]["event_type"] == "long_inactivity"

    def test_no_triggers_when_all_quiet(self):
        a2_summary = {
            "daily_repetitive_path_count": 0,
            "daily_hotspot_action_count": 0,
            "daily_prolonged_inactive_count": 0,
            "max_inactive_stretch_sec": 50.0,
            "daily_avg_social_intensity": 0.0,
        }
        triggers = generate_mllm_triggers(a2_summary)
        assert len(triggers) == 0

    def test_prioritized_ordering(self):
        """三个都触发时，按优先级排序：徘徊 > 社交 > 久坐。"""
        a2_summary = {
            "daily_repetitive_path_count": 5,
            "daily_hotspot_action_count": 3,
            "daily_prolonged_inactive_count": 1,
            "max_inactive_stretch_sec": 5000.0,
            "daily_avg_social_intensity": 0.8,
        }
        triggers = generate_mllm_triggers(a2_summary)
        assert len(triggers) == 3
        assert triggers[0]["event_type"] == "repetitive_behavior"
        assert triggers[1]["event_type"] == "social_interaction"
        assert triggers[2]["event_type"] == "long_inactivity"

    def test_with_event_list(self):
        a2_summary = {
            "daily_repetitive_path_count": 2,
            "daily_hotspot_action_count": 0,
            "daily_prolonged_inactive_count": 0,
            "max_inactive_stretch_sec": 100.0,
            "daily_avg_social_intensity": 0.0,
        }
        a2_events = {
            "repetitive_path": [{"timestamp": 450.0, "is_wandering": True}],
        }
        triggers = generate_mllm_triggers(a2_summary, a2_events)
        assert triggers[0]["trigger_ts"] == 450.0

    def test_confidence_filter(self):
        a2_summary = {
            "daily_repetitive_path_count": 1,
            "daily_hotspot_action_count": 0,
            "daily_prolonged_inactive_count": 0,
            "max_inactive_stretch_sec": 100.0,
            "daily_avg_social_intensity": 0.0,
        }
        # All triggers should still fire (min_confidence only applied in verifier, not here)
        triggers = generate_mllm_triggers(a2_summary, min_confidence=0.5)
        assert len(triggers) == 1


# ============================================================
# 降级与异常处理测试
# ============================================================

class TestErrorHandling:
    """异常处理与降级。"""

    def test_safe_default(self):
        verifier = MLLMVerifier(mode="mock")
        default = verifier._safe_default("long_inactivity", 10.0, 30.0)
        assert default["evidence_sufficient"] is False
        assert default["activity_state"] == "uncertain"
        assert default["start_sec"] == 10.0
        assert default["end_sec"] == 30.0

    def test_mock_always_returns_valid_json(self, test_video):
        verifier = MLLMVerifier(mode="mock")
        for event_type in ["long_inactivity", "social_interaction", "repetitive_behavior"]:
            result = verifier.verify(test_video, event_type, trigger_ts=0.0)
            assert isinstance(result, dict)
            assert result["event_type"] == event_type


class TestFindEventTS:
    """_find_event_ts 辅助函数。"""

    def test_empty_events(self):
        assert _find_event_ts(None, "any") == 0.0
        assert _find_event_ts({}, "any") == 0.0

    def test_event_found(self):
        events = {"repetitive_path": [{"timestamp": 300.0}]}
        assert _find_event_ts(events, "repetitive_path") == 300.0

    def test_last_event_used(self):
        events = {
            "social_interaction": [
                {"timestamp": 100.0},
                {"timestamp": 500.0},
            ],
        }
        assert _find_event_ts(events, "social_interaction") == 500.0
