"""A3EventDispatcher 单元测试。

覆盖:
  - 冷却期逻辑（冷却期内/外触发）
  - pending_count 累加与覆盖
  - 多 event_type 独立冷却期
  - 边界条件（reset, flush, 无效 event_type）
"""

import json
import pytest
from unittest.mock import patch

from src.video_analysis.event_dispatcher import A3EventDispatcher
from src.video_analysis.mllm_verifier import MLLMVerifier


def _mock_verify(video_path, event_type, trigger_ts, pre_sec=None, post_sec=None):
    """Mock MLLMVerifier.verify() — 返回符合 §6.2 的模拟 JSON。"""
    cooldowns = {"repetitive_behavior": 60, "social_interaction": 120, "long_inactivity": 120}
    return {
        "event_type": event_type,
        "cooling_period": cooldowns.get(event_type, 60),
        "num_of_occurrences": 1,
        "observable_evidence": f"Mock evidence for {event_type} at {trigger_ts}s",
        "analytical_summary": f"老人出现{event_type}相关行为，疑似需要关注",
        "start_sec": max(0, trigger_ts - 5),
        "end_sec": trigger_ts + 10,
        "activity_state": "active",
        "social_context": "alone",
        "repetition_type": "none",
        "quality_flags": [],
        "evidence_sufficient": True,
    }


@pytest.fixture
def dispatcher():
    """创建 A3EventDispatcher，verify() 已 mock。"""
    verifier = MLLMVerifier(mode="mock")
    dispatcher = A3EventDispatcher(verifier, video_path="/fake/path.mp4")
    # Mock verify 避免真实视频 IO
    with patch.object(verifier, "verify", side_effect=_mock_verify):
        yield dispatcher


class TestCooldown:
    """冷却期核心逻辑测试。"""

    def test_first_trigger_calls_mllm(self, dispatcher):
        """首次触发应调用 MLLM。"""
        result = dispatcher.on_trigger("repetitive_behavior", 100.0)
        assert result is not None
        assert result["event_type"] == "repetitive_behavior"
        assert result["cooling_period"] == 60
        assert result["num_of_occurrences"] == 1

    def test_trigger_within_cooldown_returns_none(self, dispatcher):
        """冷却期内触发应返回 None。"""
        # 首次触发
        dispatcher.on_trigger("repetitive_behavior", 100.0)
        # 冷却期内（100 + 60 = 160 之前）
        result = dispatcher.on_trigger("repetitive_behavior", 130.0)
        assert result is None

    def test_trigger_after_cooldown_calls_mllm_again(self, dispatcher):
        """冷却期结束后再次触发应调用 MLLM。"""
        dispatcher.on_trigger("social_interaction", 100.0)  # cooldown until 220
        # 冷却期内
        assert dispatcher.on_trigger("social_interaction", 150.0) is None
        # 冷却期后
        result = dispatcher.on_trigger("social_interaction", 230.0)
        assert result is not None
        assert result["num_of_occurrences"] == 2  # 第一次 MLLM(1) + 冷却期内再触发(1)

    def test_exact_cooldown_boundary(self, dispatcher):
        """刚好冷却期结束时触发。"""
        dispatcher.on_trigger("repetitive_behavior", 100.0)  # cooldown until 160
        # 160.0 == cooldown_until，应视为冷却期外
        result = dispatcher.on_trigger("repetitive_behavior", 160.0)
        assert result is not None


class TestPendingCount:
    """pending_count 累加与覆盖测试。"""

    def test_single_occurrence_count_is_one(self, dispatcher):
        """首次触发 num_of_occurrences=1。"""
        result = dispatcher.on_trigger("long_inactivity", 50.0)
        assert result["num_of_occurrences"] == 1

    def test_multiple_triggers_accumulate(self, dispatcher):
        """冷却期内多次触发应累加计数。"""
        dispatcher.on_trigger("repetitive_behavior", 100.0)  # MLLM called, reset
        # 冷却期内触发 3 次
        for ts in [110, 120, 130]:
            assert dispatcher.on_trigger("repetitive_behavior", ts) is None
        # 冷却期后触发，应包含这 3 次 + 本次
        result = dispatcher.on_trigger("repetitive_behavior", 170.0)
        assert result["num_of_occurrences"] == 4  # 3 in cooldown + 1 this trigger

    def test_pending_count_resets_after_mllm_call(self, dispatcher):
        """MLLM 调用后 pending_count 应重置为 0。"""
        dispatcher.on_trigger("repetitive_behavior", 100.0)
        status = dispatcher.get_cooldown_status("repetitive_behavior")
        assert status["repetitive_behavior"]["pending_count"] == 0


class TestIndependentCooldowns:
    """不同 event_type 冷却期独立。"""

    def test_different_types_independent_cooldown(self, dispatcher):
        """repetitive 冷却不应影响 social。"""
        dispatcher.on_trigger("repetitive_behavior", 100.0)  # cooldown until 160
        # social 不应受影响
        result = dispatcher.on_trigger("social_interaction", 110.0)
        assert result is not None  # 应正常调用 MLLM

    def test_all_three_event_types(self, dispatcher):
        """三个 event_type 可各自独立触发。"""
        results = []
        for et in ["repetitive_behavior", "social_interaction", "long_inactivity"]:
            r = dispatcher.on_trigger(et, 0.0)
            assert r is not None
            assert r["event_type"] == et
            results.append(r)
        assert len(results) == 3
        assert dispatcher.total_mllm_calls == 3

    def test_correct_cooldown_durations(self, dispatcher):
        """每个 event_type 冷却期值正确。"""
        for et, expected_cd in A3EventDispatcher.COOLDOWN.items():
            r = dispatcher.on_trigger(et, 0.0)
            assert r["cooling_period"] == expected_cd, f"{et} cooldown mismatch"


class TestLifecycle:
    """生命周期方法测试。"""

    def test_flush_returns_and_clears(self, dispatcher):
        """flush 应返回结果并清空内部列表。"""
        dispatcher.on_trigger("repetitive_behavior", 0.0)
        assert dispatcher.total_mllm_calls == 1
        results = dispatcher.flush()
        assert len(results) == 1
        assert dispatcher.total_mllm_calls == 0  # 已清空

    def test_flush_idempotent(self, dispatcher):
        """重复 flush 应返回空列表。"""
        dispatcher.on_trigger("repetitive_behavior", 0.0)
        dispatcher.flush()
        results = dispatcher.flush()
        assert results == []

    def test_reset_clears_everything(self, dispatcher):
        """reset 清空冷却期、计数器和结果。"""
        dispatcher.on_trigger("repetitive_behavior", 0.0)
        dispatcher.on_trigger("social_interaction", 10.0)
        dispatcher.reset()
        assert dispatcher.total_mllm_calls == 0
        assert dispatcher.total_triggers == 0
        assert dispatcher.get_cooldown_status()["repetitive_behavior"]["cooldown_until"] is None

    def test_get_cooldown_status_single(self, dispatcher):
        """查询单个 event_type 冷却期状态。"""
        dispatcher.on_trigger("repetitive_behavior", 100.0)
        status = dispatcher.get_cooldown_status("repetitive_behavior")
        r = status["repetitive_behavior"]
        assert r["cooldown_until"] == 160.0
        assert r["pending_count"] == 0

    def test_get_cooldown_status_all(self, dispatcher):
        """查询所有 event_type 冷却期状态。"""
        status = dispatcher.get_cooldown_status()
        assert len(status) == 3
        for et in A3EventDispatcher.COOLDOWN:
            assert et in status


class TestEdgeCases:
    """边界条件测试。"""

    def test_invalid_event_type_raises(self, dispatcher):
        """无效 event_type 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Unknown event_type"):
            dispatcher.on_trigger("invalid_type", 0.0)

    def test_total_triggers_count(self, dispatcher):
        """total_triggers 应统计所有触发（含冷却期内）。"""
        dispatcher.on_trigger("repetitive_behavior", 0.0)    # MLLM调用
        dispatcher.on_trigger("repetitive_behavior", 10.0)   # 冷却期，仅计数
        dispatcher.on_trigger("repetitive_behavior", 20.0)   # 冷却期，仅计数
        assert dispatcher.total_triggers == 3
        assert dispatcher.total_mllm_calls == 1

    def test_repr(self, dispatcher):
        """__repr__ 无异常。"""
        dispatcher.on_trigger("repetitive_behavior", 0.0)
        s = repr(dispatcher)
        assert "mllm_calls=1" in s
