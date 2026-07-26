"""CrossValidator 单元测试。

覆盖:
  - 三种 verdict（confirmed / conflict / uncertain）
  - 拒判条件（evidence_sufficient=false, quality_flags, uncertain 关键字段值）
  - 三种 event_type 的分场景判定
  - 置信度计算边界
  - 汇总评估
"""

import pytest
from src.video_analysis.cross_validator import CrossValidator


@pytest.fixture
def validator():
    return CrossValidator()


# ---- 确认(confirmed)场景 ----

def test_repetitive_behavior_confirmed(validator):
    """CV 说徘徊 → MLLM 说 same_route：确认。"""
    a3 = [{"event_type": "repetitive_behavior", "repetition_type": "same_route",
           "activity_state": "active", "social_context": "alone",
           "observable_evidence": "来回走动", "analytical_summary": "分析",
           "evidence_sufficient": True, "quality_flags": [], "start_sec": 100.0, "end_sec": 120.0}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "confirmed"
    assert r[0]["confidence"] >= 0.8
    assert r[0]["recommend_alert"] is True
    assert r[0]["alert_level"] == "medium"

def test_social_interaction_confirmed(validator):
    """CV 说多人 → MLLM 说 interacting：确认。"""
    a3 = [{"event_type": "social_interaction", "social_context": "interacting",
           "activity_state": "sedentary", "repetition_type": "none",
           "observable_evidence": "两人交谈", "analytical_summary": "分析",
           "evidence_sufficient": True, "quality_flags": [], "start_sec": 50.0, "end_sec": 65.0}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "confirmed"

def test_long_inactivity_confirmed(validator):
    """CV 说久坐 → MLLM 说 sedentary：确认。"""
    a3 = [{"event_type": "long_inactivity", "activity_state": "sedentary",
           "social_context": "alone", "repetition_type": "none",
           "observable_evidence": "老人静坐", "analytical_summary": "分析",
           "evidence_sufficient": True, "quality_flags": [], "start_sec": 200.0, "end_sec": 220.0}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "confirmed"


# ---- 冲突(conflict)场景 ----

def test_social_interaction_conflict_alone(validator):
    """CV 说多人 → MLLM 说 alone：冲突（经典 CV 假阳性）。"""
    a3 = [{"event_type": "social_interaction", "social_context": "alone",
           "activity_state": "sedentary", "repetition_type": "none",
           "observable_evidence": "单人活动", "analytical_summary": "分析",
           "evidence_sufficient": True, "quality_flags": [], "start_sec": 50.0, "end_sec": 65.0}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "conflict"
    assert r[0]["recommend_alert"] is False
    assert r[0]["alert_level"] == "low"
    assert "假阳性" in r[0]["detail"]

def test_repetitive_behavior_conflict_none(validator):
    """CV 说重复 → MLLM 说 none：冲突。"""
    a3 = [{"event_type": "repetitive_behavior", "repetition_type": "none",
           "activity_state": "active", "social_context": "alone",
           "observable_evidence": "正常走动", "analytical_summary": "分析",
           "evidence_sufficient": True, "quality_flags": [], "start_sec": 100.0, "end_sec": 120.0}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "conflict"

def test_long_inactivity_conflict_active(validator):
    """CV 说久坐 → MLLM 说 active：冲突。"""
    a3 = [{"event_type": "long_inactivity", "activity_state": "active",
           "social_context": "alone", "repetition_type": "none",
           "observable_evidence": "老人在走动", "analytical_summary": "分析",
           "evidence_sufficient": True, "quality_flags": [], "start_sec": 200.0, "end_sec": 220.0}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "conflict"


# ---- 不确定(uncertain) / 拒判场景 ----

def test_evidence_insufficient(validator):
    """evidence_sufficient=false → uncertain，不报警。"""
    a3 = [{"event_type": "repetitive_behavior", "repetition_type": "same_route",
           "evidence_sufficient": False, "quality_flags": [],
           "observable_evidence": "", "analytical_summary": "",
           "activity_state": "uncertain", "social_context": "alone",
           "start_sec": 0, "end_sec": 10}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "uncertain"
    assert r[0]["recommend_alert"] is False
    assert r[0]["alert_level"] is None

def test_quality_flags_occlusion(validator):
    """质量标记 → uncertain。"""
    a3 = [{"event_type": "social_interaction", "social_context": "interacting",
           "evidence_sufficient": True, "quality_flags": ["occlusion"],
           "observable_evidence": "", "analytical_summary": "",
           "activity_state": "sedentary", "repetition_type": "none",
           "start_sec": 0, "end_sec": 10}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "uncertain"
    assert "occlusion" in r[0]["detail"]

def test_key_value_uncertain(validator):
    """MLLM 关键字段 = uncertain → uncertain。"""
    a3 = [{"event_type": "repetitive_behavior", "repetition_type": "uncertain",
           "evidence_sufficient": True, "quality_flags": [],
           "observable_evidence": "看不清", "analytical_summary": "",
           "activity_state": "active", "social_context": "alone",
           "start_sec": 0, "end_sec": 10}]
    r = validator.validate(a3)
    assert r[0]["verdict"] == "uncertain"


# ---- 置信度范围 ----

def test_confidence_in_range(validator):
    """所有置信度在 0-1 之间。"""
    all_a3 = []
    for et in ["long_inactivity", "social_interaction", "repetitive_behavior"]:
        for state in ["confirmed", "conflict", "uncertain"]:
            a3 = {"event_type": et, "evidence_sufficient": True, "quality_flags": [],
                  "observable_evidence": "test", "analytical_summary": "test",
                  "start_sec": 0, "end_sec": 10}
            if state == "confirmed":
                a3["activity_state"] = "sedentary"
                a3["social_context"] = "interacting"
                a3["repetition_type"] = "same_route"
            elif state == "conflict":
                a3["activity_state"] = "active"
                a3["social_context"] = "alone"
                a3["repetition_type"] = "none"
            else:
                a3["evidence_sufficient"] = False
            all_a3.append(a3)

    results = validator.validate(all_a3)
    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0, f"confidence out of range: {r}"


# ---- 多个事件 ----

def test_multiple_events(validator):
    """多个事件各自独立判定。"""
    a3 = [
        {"event_type": "repetitive_behavior", "repetition_type": "same_route",
         "evidence_sufficient": True, "quality_flags": [],
         "observable_evidence": "e1", "analytical_summary": "a1",
         "activity_state": "active", "social_context": "alone",
         "start_sec": 0, "end_sec": 10},
        {"event_type": "social_interaction", "social_context": "alone",
         "evidence_sufficient": True, "quality_flags": [],
         "observable_evidence": "e2", "analytical_summary": "a2",
         "activity_state": "sedentary", "repetition_type": "none",
         "start_sec": 20, "end_sec": 30},
        {"event_type": "long_inactivity", "activity_state": "sedentary",
         "evidence_sufficient": True, "quality_flags": [],
         "observable_evidence": "e3", "analytical_summary": "a3",
         "social_context": "alone", "repetition_type": "none",
         "start_sec": 40, "end_sec": 60},
    ]
    r = validator.validate(a3)
    assert len(r) == 3
    assert r[0]["verdict"] == "confirmed"
    assert r[1]["verdict"] == "conflict"
    assert r[2]["verdict"] == "confirmed"


# ---- 汇总 ----

def test_summarize_empty(validator):
    """空事件列表 → normal。"""
    s = validator.summarize([])
    assert s["overall_status"] == "normal"
    assert s["total_events"] == 0

def test_summarize_all_confirmed(validator):
    """全部确认 → alert。"""
    validated = [{"verdict": "confirmed"}] * 8
    s = validator.summarize(validated)
    assert s["overall_status"] == "alert"

def test_summarize_mostly_conflict(validator):
    """大部分冲突 → normal（假阳性正常）。"""
    validated = [{"verdict": "conflict"}] * 6 + [{"verdict": "confirmed"}] * 2
    s = validator.summarize(validated)
    assert s["overall_status"] == "normal"

def test_summarize_mixed(validator):
    """confirmed 占多数 → caution。"""
    validated = [{"verdict": "confirmed"}] * 5 + [{"verdict": "conflict"}] * 3 + [{"verdict": "uncertain"}]
    s = validator.summarize(validated)
    assert s["overall_status"] == "caution"

def test_summarize_half_conflict(validator):
    """一半冲突一半确认 → normal（假阳性正常）。"""
    validated = [{"verdict": "conflict"}] * 3 + [{"verdict": "confirmed"}] * 3
    s = validator.summarize(validated)
    assert s["overall_status"] == "normal"

def test_repr(validator):
    assert "CrossValidator" in repr(validator)
