"""
tests/unit/test_conflict_detector.py

Task 8 — hardened ConflictDetector tests.

Classes:
  1.  ConflictTypes        — all 7 conflict types detected correctly
  2.  ConflictSeverity     — severity levels assigned correctly
  3.  SemanticAliases      — known aliases do NOT produce conflicts
  4.  UnitNormalisation    — same measurement in different units not a conflict
  5.  ConfidenceSuppress   — low-confidence new extraction suppressed
  6.  AutoResolution       — range violations and low-severity auto-resolved
  7.  ConflictStore        — lifecycle: add/resolve/dismiss/escalate/dedup
  8.  CrossField           — age+experience, BMI, workout+rest day rules
  9.  ClarificationQ       — per-type natural clarification questions
  10. ResolveFromUser      — parse user message to win a conflict
  11. ConflictReport       — session stats populated correctly
  12. TurnMap              — turn numbers tracked and used in conflict
  13. EngineIntegration    — conflict detected in full engine pipeline
  14. V1Compatibility      — old check() / resolve() API still works
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.intelligence.conflict_detector import (
    ConflictDetector,
    ConflictStore,
    ConflictType,
    ConflictSeverity,
    ConflictStatus,
    ConflictEvidence,
    Conflict,
    _aliases_match,
    _try_parse_with_unit,
    _should_auto_resolve,
)

FIELDS_CONFIG = {
    "name":                  {"type": "text"},
    "age":                   {"type": "integer", "min": 1,  "max": 120},
    "weight_kg":             {"type": "number",  "min": 30, "max": 300},
    "height_cm":             {"type": "number",  "min": 100,"max": 250},
    "smoker":                {"type": "boolean"},
    "primary_goal":          {"type": "text",
                              "allowed_values": ["lose weight","build muscle","run 5k"]},
    "activity_level":        {"type": "text",
                              "allowed_values": ["sedentary","lightly active","very active"]},
    "workout_days_per_week": {"type": "integer", "min": 1,  "max": 7},
    "diet":                  {"type": "text"},
    "equipment_access":      {"type": "text"},
    "work_experience_years": {"type": "integer"},
    "rest_days_per_week":    {"type": "integer"},
}

COLLECTED = {
    "name":                  "Priya",
    "age":                   28,
    "weight_kg":             65.0,
    "height_cm":             163.0,
    "smoker":                "no",
    "primary_goal":          "lose weight",
    "activity_level":        "moderately active",
    "workout_days_per_week": 4,
    "diet":                  "vegetarian",
    "equipment_access":      "gym membership",
}

DET = ConflictDetector()

def check(new: dict, collected: dict = None, cfg: dict = None, turn: int = 5,
          turn_map: dict = None, confidences: dict = None) -> list:
    return DET.check(
        new_extractions   = new,
        collected         = collected if collected is not None else dict(COLLECTED),
        fields_config     = cfg if cfg is not None else FIELDS_CONFIG,
        current_turn      = turn,
        turn_map          = turn_map or {k: 1 for k in (collected or COLLECTED)},
        field_confidences = confidences,
    )

class TestConflictTypes:

    def test_numeric_mismatch_detected(self):
        cs = check({"age": 35})
        assert len(cs) == 1
        assert cs[0].conflict_type == ConflictType.NUMERIC_MISMATCH

    def test_boolean_flip_yes_to_no(self):
        cs = check({"smoker": "yes"})
        assert len(cs) == 1
        assert cs[0].conflict_type == ConflictType.BOOLEAN_FLIP

    def test_boolean_flip_no_to_yes(self):
        cs = check({"smoker": "yes"}, collected={"smoker": "no"})
        assert cs[0].conflict_type == ConflictType.BOOLEAN_FLIP

    def test_categorical_flip_enum_field(self):
        cs = check({"primary_goal": "build muscle"})
        assert len(cs) == 1
        assert cs[0].conflict_type == ConflictType.CATEGORICAL_FLIP

    def test_range_violation_below_min(self):
        cs = check({"age": 0})
        assert any(c.conflict_type == ConflictType.RANGE_VIOLATION for c in cs)

    def test_range_violation_above_max(self):
        cs = check({"age": 200})
        assert any(c.conflict_type == ConflictType.RANGE_VIOLATION for c in cs)

    def test_text_contradiction_detected(self):
        cs = check(
            {"diet": "strict carnivore meat-based diet"},
            collected={"diet": "committed vegan no animal products whatsoever"},
        )
        assert len(cs) >= 1

    def test_semantic_contradiction_smoking(self):
        cs = check(
            {"diet": "I smoke 10 cigarettes daily"},
            collected={"diet": "I don't smoke at all"},
        )

        assert len(cs) >= 1

    def test_no_conflict_same_value(self):
        cs = check({"age": 28})
        assert len(cs) == 0

    def test_no_conflict_first_time_field(self):
        cs = check({"height_cm": 163}, collected={})
        assert len(cs) == 0

class TestConflictSeverity:

    def test_large_numeric_diff_critical(self):
        cs = check({"age": 70})
        assert cs[0].severity == ConflictSeverity.CRITICAL

    def test_medium_numeric_diff_high(self):
        cs = check({"age": 38})
        assert cs[0].severity == ConflictSeverity.HIGH

    def test_small_diff_medium(self):

        cs = check({"age": 32})
        assert cs[0].severity in (ConflictSeverity.MEDIUM, ConflictSeverity.HIGH)

    def test_boolean_flip_always_high(self):
        cs = check({"smoker": "yes"})
        assert cs[0].severity == ConflictSeverity.HIGH

    def test_range_violation_always_critical(self):
        cs = check({"age": 200})
        assert cs[0].severity == ConflictSeverity.CRITICAL

    def test_categorical_flip_high(self):
        cs = check({"primary_goal": "build muscle"})
        assert cs[0].severity == ConflictSeverity.HIGH

class TestSemanticAliases:

    def test_gym_aliases_match(self):
        assert _aliases_match("gym", "gym membership") is True
        assert _aliases_match("at the gym", "gym") is True
        assert _aliases_match("fitness center", "gym") is True

    def test_weight_loss_aliases_match(self):
        assert _aliases_match("lose weight", "weight loss") is True
        assert _aliases_match("lose fat", "lose weight") is True
        assert _aliases_match("get lean", "lose weight") is True

    def test_bool_aliases_match(self):
        assert _aliases_match("yes", "yeah") is True
        assert _aliases_match("no", "nope") is True

    def test_no_alias_across_groups(self):
        assert _aliases_match("yes", "no") is False
        assert _aliases_match("gym", "home workout") is False

    def test_gym_rephrase_no_conflict(self):
        cs = check(
            {"equipment_access": "at the gym"},
            collected={"equipment_access": "gym membership"},
        )
        assert len(cs) == 0

    def test_weight_loss_rephrase_no_conflict(self):
        cs = check(
            {"primary_goal": "weight loss"},
            collected={"primary_goal": "lose weight"},
            cfg={**FIELDS_CONFIG,
                 "primary_goal": {"type": "text",
                                  "allowed_values": ["lose weight","weight loss","build muscle"]}},
        )
        assert len(cs) == 0

    def test_sedentary_rephrase_no_conflict(self):
        cs = check(
            {"activity_level": "mostly sitting"},
            collected={"activity_level": "sedentary"},
        )
        assert len(cs) == 0

class TestUnitNormalisation:

    def test_parse_kg(self):
        val = _try_parse_with_unit("65 kg", "weight_kg")
        assert val == pytest.approx(65.0, abs=0.1)

    def test_parse_lbs_to_kg(self):
        val = _try_parse_with_unit("143 lbs", "weight_kg")

        assert val == pytest.approx(64.86, abs=0.5)

    def test_parse_cm(self):
        val = _try_parse_with_unit("163 cm", "height_cm")
        assert val == pytest.approx(163.0)

    def test_parse_metres_to_cm(self):
        val = _try_parse_with_unit("1.63 m", "height_cm")
        assert val == pytest.approx(163.0, abs=1.0)

    def test_same_weight_different_unit_no_conflict(self):

        cs = check(
            {"weight_kg": "143 lbs"},
            collected={"weight_kg": "65 kg"},
        )

        assert all(c.conflict_type != ConflictType.UNIT_MISMATCH for c in cs)

    def test_genuinely_different_weight_unit_mismatch(self):

        cs = check(
            {"weight_kg": "100 lbs"},
            collected={"weight_kg": "65 kg"},
        )
        assert len(cs) >= 1

    def test_unknown_field_no_unit_check(self):

        val = _try_parse_with_unit("28 years", "age")
        assert val is None

class TestConfidenceSuppression:

    def test_low_confidence_new_suppressed(self):

        cs = check(
            {"age": 35},
            confidences={"age": 0.95},
        )

        det = ConflictDetector()
        result = det._compare_field(
            field_name="age", old_val=28, new_val=35,
            field_cfg={"type": "integer"},
            turn_old=1, turn_new=5,
            old_confidence=0.95,
            new_confidence=0.30,
        )

        assert result is None

    def test_high_confidence_new_not_suppressed(self):
        det = ConflictDetector()
        result = det._compare_field(
            field_name="age", old_val=28, new_val=35,
            field_cfg={"type": "integer"},
            turn_old=1, turn_new=5,
            old_confidence=0.70,
            new_confidence=0.90,
        )
        assert result is not None
        assert result.conflict_type == ConflictType.NUMERIC_MISMATCH

    def test_equal_confidence_not_suppressed(self):
        det = ConflictDetector()
        result = det._compare_field(
            field_name="age", old_val=28, new_val=35,
            field_cfg={"type": "integer"},
            turn_old=1, turn_new=5,
            old_confidence=0.80,
            new_confidence=0.80,
        )
        assert result is not None

class TestAutoResolution:

    def _make_conflict(self, ctype, severity, old=28, new=200,
                       old_conf=0.80, new_conf=0.80) -> Conflict:
        c = ConflictDetector._make_conflict(
            "age", old, new, ctype, severity, 1, 5,
            ConflictEvidence(old_confidence=old_conf, new_confidence=new_conf),
        )
        return c

    def test_range_violation_auto_resolves_to_old(self):
        c = self._make_conflict(ConflictType.RANGE_VIOLATION, ConflictSeverity.CRITICAL)
        should, val = _should_auto_resolve(c)
        assert should is True
        assert val == c.old_value

    def test_low_severity_auto_resolves_to_new(self):
        c = self._make_conflict(ConflictType.TEXT_CONTRADICTION, ConflictSeverity.LOW)
        should, val = _should_auto_resolve(c)
        assert should is True
        assert val == c.new_value

    def test_high_new_confidence_auto_resolves_to_new(self):
        c = self._make_conflict(ConflictType.NUMERIC_MISMATCH, ConflictSeverity.MEDIUM,
                                old_conf=0.30, new_conf=0.90)
        should, val = _should_auto_resolve(c)
        assert should is True
        assert val == c.new_value

    def test_critical_equal_confidence_not_auto_resolved(self):
        c = self._make_conflict(ConflictType.BOOLEAN_FLIP, ConflictSeverity.CRITICAL)
        should, val = _should_auto_resolve(c)
        assert should is False
        assert val is None

    def test_check_and_store_auto_resolves_range_violation(self):
        store = ConflictStore()
        det   = ConflictDetector()
        collected = dict(COLLECTED)
        result = det.check_and_store(
            new_extractions = {"age": 200},
            collected       = collected,
            fields_config   = FIELDS_CONFIG,
            current_turn    = 5,
            store           = store,
            turn_map        = {"age": 1},
        )

        assert len(result) == 0

        assert any(c.status == ConflictStatus.AUTO_RESOLVED for c in store._conflicts)

        assert collected["age"] == COLLECTED["age"]

class TestConflictStore:

    def _make_conflict(self, field="age", old=28, new=35) -> Conflict:
        return ConflictDetector._make_conflict(
            field, old, new,
            ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 1, 5,
        )

    def test_add_conflict(self):
        store = ConflictStore()
        c     = self._make_conflict()
        added = store.add(c)
        assert added is True
        assert len(store.open_conflicts) == 1

    def test_duplicate_not_added(self):
        store = ConflictStore()
        c1 = self._make_conflict()
        c2 = self._make_conflict()
        store.add(c1)
        added = store.add(c2)
        assert added is False
        assert len(store.open_conflicts) == 1

    def test_resolve_by_id(self):
        store     = ConflictStore()
        c         = self._make_conflict()
        store.add(c)
        collected = {"age": 28}
        store.resolve(c.id, 35, 6, collected)
        assert collected["age"] == 35
        assert c.status == ConflictStatus.RESOLVED
        assert len(store.open_conflicts) == 0

    def test_dismiss(self):
        store = ConflictStore()
        c = self._make_conflict()
        store.add(c)
        store.dismiss(c.id)
        assert c.status == ConflictStatus.DISMISSED
        assert len(store.open_conflicts) == 0

    def test_escalate(self):
        store = ConflictStore()
        c = self._make_conflict()
        store.add(c)
        store.escalate(c.id)
        assert c.status == ConflictStatus.ESCALATED

        assert len(store.open_conflicts) == 1

    def test_most_severe_open(self):
        store = ConflictStore()
        c1 = self._make_conflict("age", 28, 35)
        c1.severity = ConflictSeverity.MEDIUM
        c2 = self._make_conflict("weight_kg", 65, 80)
        c2.severity = ConflictSeverity.CRITICAL
        store.add(c1)
        store.add(c2)
        most_severe = store.most_severe_open()
        assert most_severe is not None
        assert most_severe.severity == ConflictSeverity.CRITICAL

    def test_open_for_field(self):
        store = ConflictStore()
        c = self._make_conflict("age")
        store.add(c)
        assert len(store.open_for_field("age")) == 1
        assert len(store.open_for_field("weight_kg")) == 0

    def test_has_open(self):
        store = ConflictStore()
        assert store.has_open() is False
        c = self._make_conflict()
        store.add(c)
        assert store.has_open() is True

    def test_to_dicts_serialisable(self):
        store = ConflictStore()
        c = self._make_conflict()
        store.add(c)
        dicts = store.to_dicts()
        assert len(dicts) == 1
        d = dicts[0]
        assert "id" in d
        assert "conflict_type" in d
        assert "severity" in d

class TestCrossField:

    def test_age_work_experience_conflict(self):
        det = ConflictDetector()
        collected = {"age": 22, "work_experience_years": 20}
        cs = det.check_cross_field(collected, FIELDS_CONFIG, current_turn=5)
        assert len(cs) >= 1
        assert any(c.conflict_type == ConflictType.CROSS_FIELD for c in cs)
        assert any(c.field == "age" for c in cs)

    def test_age_work_experience_valid_no_conflict(self):
        det = ConflictDetector()
        collected = {"age": 45, "work_experience_years": 20}
        cs = det.check_cross_field(collected, FIELDS_CONFIG, current_turn=5)
        assert len(cs) == 0

    def test_bmi_conflict_too_low(self):
        det = ConflictDetector()

        collected = {"weight_kg": 10, "height_cm": 163}
        cs = det.check_cross_field(collected, FIELDS_CONFIG, current_turn=5)
        assert any(c.conflict_type == ConflictType.CROSS_FIELD for c in cs)

    def test_bmi_valid_no_conflict(self):
        det = ConflictDetector()
        collected = {"weight_kg": 65, "height_cm": 163}
        cs = det.check_cross_field(collected, FIELDS_CONFIG, current_turn=5)
        bmi_conflicts = [c for c in cs if "height_cm" in (c.field, c.related_field or "")]
        assert len(bmi_conflicts) == 0

    def test_workout_rest_days_conflict(self):
        det = ConflictDetector()
        cfg = {**FIELDS_CONFIG, "rest_days_per_week": {"type": "integer"}}
        collected = {"workout_days_per_week": 6, "rest_days_per_week": 6}
        cs = det.check_cross_field(collected, cfg, current_turn=5)
        assert any(c.conflict_type == ConflictType.CROSS_FIELD for c in cs)

    def test_cross_field_only_when_both_collected(self):
        det = ConflictDetector()
        collected = {"age": 22}
        cs = det.check_cross_field(collected, FIELDS_CONFIG, current_turn=5)
        assert len(cs) == 0

    def test_cross_field_uses_store(self):
        det   = ConflictDetector()
        store = ConflictStore()
        collected = {"age": 22, "work_experience_years": 20}
        det.check_cross_field(collected, FIELDS_CONFIG, current_turn=5, store=store)
        assert store.has_open()

class TestClarificationQuestions:

    def _conflict(self, field, old, new, ctype, severity=ConflictSeverity.HIGH):
        return ConflictDetector._make_conflict(
            field, old, new, ctype, severity, 1, 5
        )

    def test_numeric_clarification(self):
        c = self._conflict("age", 28, 35, ConflictType.NUMERIC_MISMATCH)
        q = c.clarification_question({"label": "age"})
        assert "28" in q or "35" in q
        assert "?" in q

    def test_boolean_clarification(self):
        c = self._conflict("smoker", "no", "yes", ConflictType.BOOLEAN_FLIP)
        q = c.clarification_question({"label": "smoker"})
        assert "no" in q.lower() or "yes" in q.lower()

    def test_categorical_clarification(self):
        c = self._conflict("primary_goal", "lose weight", "build muscle",
                            ConflictType.CATEGORICAL_FLIP)
        q = c.clarification_question()
        assert "lose weight" in q or "build muscle" in q

    def test_range_violation_clarification(self):
        c = self._conflict("age", 28, 200, ConflictType.RANGE_VIOLATION)
        q = c.clarification_question({"label": "age"})
        assert "range" in q.lower() or "200" in q

    def test_cross_field_clarification(self):
        c = self._conflict("age", 22, 20, ConflictType.CROSS_FIELD)
        q = c.clarification_question()
        assert "?" in q
        assert len(q) > 10

    def test_all_conflict_types_have_question(self):
        for ctype in ConflictType:
            c = self._conflict("age", 28, 35, ctype)
            q = c.clarification_question()
            assert isinstance(q, str)
            assert len(q) > 10
            assert "?" in q

class TestResolveFromUser:

    def _setup(self):
        store = ConflictStore()
        det   = ConflictDetector()
        c = ConflictDetector._make_conflict(
            "age", 28, 35,
            ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 1, 5,
        )
        store.add(c)
        return det, store, c

    def test_prefers_old_on_signal(self):
        det, store, c = self._setup()
        collected = {"age": 28}
        resolved = det.resolve_from_user_input(
            "I meant my first answer, stick with 28", store, collected, 6
        )
        assert c.id in resolved
        assert collected["age"] == 28

    def test_prefers_new_on_signal(self):
        det, store, c = self._setup()
        collected = {"age": 28}
        resolved = det.resolve_from_user_input(
            "Actually I'm 35, let me correct that", store, collected, 6
        )
        assert c.id in resolved
        assert collected["age"] == 35

    def test_extracts_number_as_resolution(self):
        det, store, c = self._setup()
        collected = {"age": 28}
        resolved = det.resolve_from_user_input(
            "The right answer is 30", store, collected, 6,
            fields_config={"age": {"type": "integer"}}
        )
        assert c.id in resolved
        assert collected["age"] == 30

    def test_old_value_in_message_resolves_to_old(self):
        det, store, c = self._setup()
        collected = {"age": 28}
        resolved = det.resolve_from_user_input("28 is correct", store, collected, 6)
        assert c.id in resolved
        assert collected["age"] == 28

    def test_ambiguous_message_not_resolved(self):
        det, store, c = self._setup()
        collected = {"age": 28}
        resolved = det.resolve_from_user_input("I'm not sure", store, collected, 6)
        assert len(resolved) == 0
        assert store.has_open()

    def test_multiple_conflicts_resolved_at_once(self):
        store = ConflictStore()
        det   = ConflictDetector()
        c1 = ConflictDetector._make_conflict("age", 28, 35,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 1, 5)
        c2 = ConflictDetector._make_conflict("smoker", "no", "yes",
                ConflictType.BOOLEAN_FLIP, ConflictSeverity.HIGH, 2, 5)
        store.add(c1)
        store.add(c2)
        collected = {"age": 28, "smoker": "no"}
        resolved = det.resolve_from_user_input(
            "Actually 35, and yes I smoke",
            store, collected, 6,
            fields_config={"age": {"type": "integer"}, "smoker": {"type": "boolean"}},
        )

        assert len(resolved) >= 1

class TestConflictReport:

    def test_empty_store_report(self):
        store  = ConflictStore()
        report = store.report("s1", total_turns=10)
        assert report.total_detected == 0
        assert report.open_count == 0
        assert report.conflict_rate == 0.0

    def test_report_counts_correctly(self):
        store = ConflictStore()
        for i in range(3):
            c = ConflictDetector._make_conflict(
                f"field_{i}", i, i+10,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 1, 5,
            )
            store.add(c)

        collected = {f"field_{i}": i for i in range(3)}
        store.resolve(store._conflicts[0].id, 99, 6, collected)
        report = store.report("s1", total_turns=10)
        assert report.total_detected == 3
        assert report.open_count     == 2
        assert report.resolved_count == 1

    def test_report_most_conflicted_fields(self):
        store = ConflictStore()

        c1 = ConflictDetector._make_conflict("age", 28, 35,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 1, 3)
        c2 = ConflictDetector._make_conflict("age", 35, 40,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 3, 6)
        c3 = ConflictDetector._make_conflict("weight_kg", 65, 70,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.MEDIUM, 1, 4)
        for c in (c1, c2, c3):
            store._conflicts.append(c)
        report = store.report("s1", total_turns=10)
        assert report.most_conflicted_fields[0] == "age"

    def test_report_to_dict_structure(self):
        store = ConflictStore()
        report = store.report("s1", total_turns=5)
        d = report.to_dict()
        for key in ["session_id", "total_detected", "open_count",
                    "resolved_count", "auto_resolved", "severity_counts",
                    "most_conflicted_fields", "conflict_rate"]:
            assert key in d

    def test_conflict_rate_calculation(self):
        store = ConflictStore()
        for i in range(4):
            c = ConflictDetector._make_conflict(
                f"f{i}", i, i+10,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH, 1, 5,
            )
            store._conflicts.append(c)
        report = store.report("s1", total_turns=10)
        assert report.conflict_rate == pytest.approx(0.4)

class TestTurnMap:

    def test_conflict_records_correct_turns(self):
        cs = check(
            {"age": 35},
            turn     = 7,
            turn_map = {"age": 3},
        )
        assert len(cs) == 1
        assert cs[0].turn_old == 3
        assert cs[0].turn_new == 7

    def test_no_turn_map_defaults_to_zero(self):
        cs = DET.check(
            new_extractions = {"age": 35},
            collected       = {"age": 28},
            fields_config   = FIELDS_CONFIG,
            current_turn    = 5,
        )
        assert cs[0].turn_old == 0
        assert cs[0].turn_new == 5

    def test_turn_gap_preserved_in_to_dict(self):
        cs = check({"age": 35}, turn=8, turn_map={"age": 2})
        d = cs[0].to_dict()
        assert d["turn_old"] == 2
        assert d["turn_new"] == 8

class TestEngineIntegration:

    GOAL = {
        "id": "conflict_test",
        "fields": [
            {"name": "age",    "type": "integer", "required": True,
             "question": "How old are you?", "min": 1, "max": 120},
            {"name": "name",   "type": "text",    "required": True,
             "question": "What is your name?"},
            {"name": "smoker", "type": "boolean", "required": True,
             "question": "Do you smoke?"},
        ],
        "persona": {"name": "Bot", "tone": "neutral"},
        "output": {"format": "json"},
    }

    @pytest.mark.asyncio
    async def test_engine_detects_conflict_mid_conversation(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient

        mock = MockLLMClient(
            responses={"extract": '{"extractions": []}'},
            default="Got it. Tell me more.",
        )
        router = LLMRouter()
        for m in ["gemini-3.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        engine = TrueNorthEngine(goal_config=self.GOAL, router=router)
        await engine.start()

        engine.state.set_field("age", 28, confidence=0.90)

        from truenorth.intelligence.conflict_detector import ConflictDetector
        det = ConflictDetector()
        conflicts = det.check(
            new_extractions   = {"age": 35},
            collected         = engine.state.collected_fields,
            fields_config     = engine.state.fields_config,
            current_turn      = 3,
            turn_map          = {"age": 1},
            field_confidences = {"age": 0.90},
        )
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.NUMERIC_MISMATCH

    @pytest.mark.asyncio
    async def test_engine_conflict_does_not_overwrite_field(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient

        mock = MockLLMClient(default="Got it.")
        router = LLMRouter()
        for m in ["gemini-3.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        engine = TrueNorthEngine(goal_config=self.GOAL, router=router)
        await engine.start()

        engine.state.set_field("age", 28)
        engine.state.active_conflicts.append({
            "field": "age", "old_value": 28, "new_value": 35, "resolved": False,
        })

        await engine.process_message("I'm Alex")

        assert engine.state.collected_fields.get("age") == 28

class TestV1Compatibility:

    def test_v1_check_returns_list(self):
        det = ConflictDetector()
        result = det.check(
            new_extractions={"age": 35},
            collected={"age": 25},
            fields_config={"age": {"type": "integer"}},
            current_turn=5,
        )
        assert isinstance(result, list)
        assert len(result) == 1

    def test_v1_conflict_has_to_dict(self):
        det = ConflictDetector()
        result = det.check(
            new_extractions={"age": 35},
            collected={"age": 25},
            fields_config={"age": {"type": "integer"}},
            current_turn=5,
        )
        d = result[0].to_dict()
        assert "field" in d
        assert "old_value" in d
        assert "new_value" in d
        assert "conflict_type" in d

    def test_v1_resolve_updates_collected(self):
        det = ConflictDetector()
        conflicts = det.check(
            new_extractions={"age": 35},
            collected={"age": 25},
            fields_config={"age": {"type": "integer"}},
            current_turn=5,
        )
        collected = {"age": 25}
        active    = [conflicts[0].to_dict()]
        det.resolve(conflicts[0], 35, collected, active)
        assert collected["age"] == 35
        assert conflicts[0].resolved is True

    def test_v1_no_conflict_empty_list(self):
        det = ConflictDetector()
        result = det.check(
            new_extractions={"age": 25},
            collected={"age": 25},
            fields_config={"age": {"type": "integer"}},
            current_turn=3,
        )
        assert result == []

    def test_v1_first_time_field_no_conflict(self):
        det = ConflictDetector()
        result = det.check(
            new_extractions={"height_cm": 163},
            collected={},
            fields_config={"height_cm": {"type": "number"}},
            current_turn=2,
        )
        assert result == []

    def test_conflict_type_enum_values_unchanged(self):

        assert ConflictType.NUMERIC_MISMATCH.value  == "numeric_mismatch"
        assert ConflictType.BOOLEAN_FLIP.value       == "boolean_flip"
        assert ConflictType.CATEGORICAL_FLIP.value   == "categorical_flip"
        assert ConflictType.RANGE_VIOLATION.value    == "range_violation"
        assert ConflictType.TEXT_CONTRADICTION.value == "text_contradiction"
