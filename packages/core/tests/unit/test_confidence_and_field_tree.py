"""
tests/unit/test_confidence_and_field_tree.py

Task 7 tests — confidence scorer hardening + conditional field trees.

Classes:
  1.  TypeValidation       — all type validators, range checks, allowed_values
  2.  ExtractionMethod     — method score affects composite
  3.  SourceQuality        — source text scoring nuances
  4.  TemporalStability    — prior_extractions consistency scoring
  5.  ConflictHistory      — conflict_history count affects score
  6.  CrossFieldConsistency— age/weight/height cross-checks
  7.  ConfidenceBands      — HIGH/MEDIUM/LOW/UNCONFIDENT thresholds
  8.  SessionHealth        — full session report
  9.  ConfidenceScoreAll   — batch scoring + backward compat
  10. FieldTree_Basic      — always-visible fields
  11. FieldTree_IfTrue     — if_true condition
  12. FieldTree_IfValueIs  — if_value_is condition
  13. FieldTree_IfValueIn  — if_value_in condition
  14. FieldTree_IfValueNot — if_value_not condition
  15. FieldTree_Numeric    — if_numeric_gt / if_numeric_lt
  16. FieldTree_Compound   — if_all_of / if_any_of
  17. FieldTree_Navigation — next_required / next_optional / all_collected
  18. ReasonerWithFieldTree— reasoner uses FieldTree for field navigation
  19. FitnessYAMLConditions— end-to-end with updated fitness_plan.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.intelligence.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceBand,
    ConfidenceScore,
    ExtractionMethod,
    _validate_type,
)
from truenorth.core.field_tree import FieldTree
from truenorth.core.reasoner  import Reasoner, ReasonerAction
from truenorth.core.graph_state import GraphState

SCORER = ConfidenceScorer()

# ─────────────────────────────────────────────────────────────────────────────
#  1. Type validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeValidation:

    def test_valid_integer(self):
        score, issues = _validate_type(28, {"type": "integer"})
        assert score >= 0.88
        assert not issues

    def test_string_integer_valid(self):
        score, issues = _validate_type("28", {"type": "integer"})
        assert score >= 0.88

    def test_float_fails_integer(self):
        score, issues = _validate_type("28.5", {"type": "integer"})
        assert score < 0.50
        assert issues

    def test_valid_number(self):
        score, _ = _validate_type(65.5, {"type": "number"})
        assert score >= 0.85

    def test_valid_email(self):
        score, _ = _validate_type("priya@example.com", {"type": "email"})
        assert score >= 0.90

    def test_invalid_email(self):
        score, issues = _validate_type("not-an-email", {"type": "email"})
        assert score < 0.20
        assert issues

    def test_valid_boolean_yes(self):
        for v in ("yes", "true", "1", "y", "True", "YES"):
            score, _ = _validate_type(v, {"type": "boolean"})
            assert score >= 0.88, f"Failed for {v!r}"

    def test_valid_boolean_no(self):
        for v in ("no", "false", "0", "n", "False", "NO"):
            score, _ = _validate_type(v, {"type": "boolean"})
            assert score >= 0.88, f"Failed for {v!r}"

    def test_range_below_min_penalised(self):
        score, issues = _validate_type(5, {"type": "integer", "min": 16, "max": 100})
        assert score < 0.65
        assert any("minimum" in i for i in issues)

    def test_range_above_max_penalised(self):
        score, issues = _validate_type(150, {"type": "integer", "min": 1, "max": 100})
        assert score < 0.65
        assert any("maximum" in i for i in issues)

    def test_value_in_allowed_list(self):
        score, issues = _validate_type(
            "lose weight",
            {"type": "text", "allowed_values": ["lose weight", "build muscle"]}
        )
        assert not any("allowed" in i for i in issues)

    def test_value_not_in_allowed_list(self):
        score, issues = _validate_type(
            "random goal",
            {"type": "text", "allowed_values": ["lose weight", "build muscle"]}
        )
        assert any("allowed" in i for i in issues)
        assert score < 0.65

    def test_text_no_validation_medium(self):
        score, _ = _validate_type("some text value", {"type": "text"})
        assert 0.50 <= score <= 0.80

    def test_none_value_scores_zero(self):
        score, _ = _validate_type(None, {"type": "integer"})
        assert score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  2. Extraction method
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionMethod:

    def test_direct_quote_highest(self):
        s1 = SCORER.score("age", 28, {"type": "integer"},
                          extraction_confidence=0.8, source_text="I am 28 years old",
                          method=ExtractionMethod.DIRECT_QUOTE)
        s2 = SCORER.score("age", 28, {"type": "integer"},
                          extraction_confidence=0.8, source_text="I am 28 years old",
                          method=ExtractionMethod.RULE_BASED)
        assert s1.score > s2.score

    def test_user_confirm_near_direct_quote(self):
        s = SCORER.score("age", 28, {"type": "integer"},
                         extraction_confidence=0.9,
                         method=ExtractionMethod.USER_CONFIRM)
        assert s.band in (ConfidenceBand.HIGH, ConfidenceBand.MEDIUM)

    def test_rule_based_lower_than_llm(self):
        s_llm  = SCORER.score("age", 28, {"type": "integer"},
                               extraction_confidence=0.85,
                               method=ExtractionMethod.LLM_EXTRACT)
        s_rule = SCORER.score("age", 28, {"type": "integer"},
                               extraction_confidence=0.85,
                               method=ExtractionMethod.RULE_BASED)
        assert s_llm.score > s_rule.score

    def test_claimed_direct_quote_not_in_source_penalised(self):
        s = SCORER.score("age", 28, {"type": "integer"},
                         extraction_confidence=0.9, source_text="I am pretty old",
                         method=ExtractionMethod.DIRECT_QUOTE)
        # Should be penalised for claiming direct quote but value not in source
        assert any("verbatim" in i for i in s.issues)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Source quality
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceQuality:

    def test_rich_context_scores_high(self):
        s = SCORER.score("age", 28, source_text="I was born in 1996 and I am 28 years old now")
        assert s.score > SCORER.score("age", 28, source_text="28").score

    def test_bare_value_scores_lower(self):
        s_rich = SCORER.score("weight_kg", 65, source_text="My weight is 65 kg and I've been tracking it")
        s_bare = SCORER.score("weight_kg", 65, source_text="65")
        assert s_rich.score > s_bare.score

    def test_missing_source_penalised(self):
        s_no_src = SCORER.score("age", 28, source_text="")
        s_with   = SCORER.score("age", 28, source_text="I am 28 years old")
        assert s_with.score > s_no_src.score

    def test_value_not_in_source_penalised(self):
        s = SCORER.score("age", 28, source_text="I am quite young actually")
        # 28 is not in source text → lower source quality
        assert s.score < 0.85


# ─────────────────────────────────────────────────────────────────────────────
#  4. Temporal stability
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalStability:

    def test_no_prior_extractions_neutral(self):
        s = SCORER.score("age", 28, prior_extractions=[])
        # 0.75 stability contribution — neutral
        assert s.score > 0.0

    def test_consistent_prior_extractions_boost(self):
        s_stable   = SCORER.score("age", 28, prior_extractions=[28, 28, 28])
        s_unstable = SCORER.score("age", 28, prior_extractions=[25, 30, 28])
        assert s_stable.score > s_unstable.score

    def test_all_different_priors_penalty(self):
        s = SCORER.score("age", 28, prior_extractions=[22, 25, 30, 35])
        assert s.score < 0.75

    def test_perfect_stability_contributes_max(self):
        s = SCORER.score("age", 28,
                          extraction_confidence=1.0,
                          source_text="I am 28 years old today",
                          user_confirmed=True,
                          prior_extractions=[28, 28, 28, 28])
        assert s.band == ConfidenceBand.HIGH


# ─────────────────────────────────────────────────────────────────────────────
#  5. Conflict history
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictHistory:

    def test_no_conflict_no_penalty(self):
        s = SCORER.score("age", 28, extraction_confidence=0.9, in_conflict=False)
        assert s.factors["conflict"] == pytest.approx(0.10, abs=0.01)

    def test_active_conflict_severe_penalty(self):
        s_conflict = SCORER.score("age", 28, extraction_confidence=0.9, in_conflict=True)
        s_clean    = SCORER.score("age", 28, extraction_confidence=0.9, in_conflict=False)
        assert s_conflict.score < s_clean.score

    def test_one_past_conflict_mild_penalty(self):
        s_hist = SCORER.score("age", 28, extraction_confidence=0.9,
                               in_conflict=False, conflict_history=1)
        s_none = SCORER.score("age", 28, extraction_confidence=0.9,
                               in_conflict=False, conflict_history=0)
        assert s_hist.score < s_none.score

    def test_repeated_conflicts_flagged_in_issues(self):
        s = SCORER.score("age", 28, conflict_history=3)
        assert any("conflict" in i.lower() for i in s.issues)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Cross-field consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFieldConsistency:

    def test_valid_bmi_no_issue(self):
        # BMI = 65 / 1.63^2 = 24.5 — normal
        s = SCORER.score(
            "weight_kg", 65,
            field_config={"type": "number"},
            collected_fields={"weight_kg": 65, "height_cm": 163},
        )
        assert not any("bmi" in i.lower() for i in s.issues)

    def test_impossible_bmi_flagged(self):
        # BMI = 10 / 1.63^2 = 3.8 — clearly wrong
        s = SCORER.score(
            "weight_kg", 10,
            field_config={"type": "number"},
            collected_fields={"weight_kg": 10, "height_cm": 163},
        )
        assert any("consistency" in i.lower() or "bmi" in i.lower() for i in s.issues)
        assert s.score < 0.85

    def test_age_out_of_range_flagged(self):
        s = SCORER.score(
            "age", 200,
            field_config={"type": "integer"},
            collected_fields={"age": 200},
        )
        assert any("consistency" in i.lower() for i in s.issues)

    def test_no_cross_field_data_passes(self):
        s = SCORER.score("age", 28, field_config={"type": "integer"})
        assert s.score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  7. Confidence bands
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceBands:

    def test_high_band(self):
        s = SCORER.score("age", 28,
                          extraction_confidence=0.95,
                          source_text="I am 28 years old and have been for a year",
                          user_confirmed=True,
                          method=ExtractionMethod.DIRECT_QUOTE)
        assert s.band == ConfidenceBand.HIGH

    def test_low_confidence_triggers_needs_confirm(self):
        s = SCORER.score("age", 28,
                          extraction_confidence=0.20,
                          source_text="",
                          in_conflict=True)
        assert s.needs_confirm is True

    def test_high_confidence_no_confirm_needed(self):
        s = SCORER.score("age", 28,
                          extraction_confidence=0.95,
                          source_text="I am exactly 28 years old",
                          user_confirmed=True)
        assert s.needs_confirm is False

    def test_band_boundaries(self):
        assert SCORER._band(0.85) == ConfidenceBand.HIGH
        assert SCORER._band(0.70) == ConfidenceBand.MEDIUM
        assert SCORER._band(0.50) == ConfidenceBand.LOW
        assert SCORER._band(0.30) == ConfidenceBand.UNCONFIDENT

    def test_to_dict_has_band(self):
        s = SCORER.score("age", 28)
        d = s.to_dict()
        assert "band" in d
        assert d["band"] in ("high", "medium", "low", "unconfident")


# ─────────────────────────────────────────────────────────────────────────────
#  8. Session health report
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionHealth:

    COLLECTED = {"name": "Priya", "age": 28, "weight_kg": 65.0, "primary_goal": "lose weight"}
    FIELDS    = {
        "name":         {"type": "text",    "required": True},
        "age":          {"type": "integer", "required": True,  "min": 1, "max": 100},
        "weight_kg":    {"type": "number",  "required": True,  "min": 30, "max": 300},
        "primary_goal": {"type": "text",    "required": True,  "allowed_values": ["lose weight", "build muscle"]},
    }
    REQUIRED = ["name", "age", "weight_kg", "primary_goal"]

    def test_report_has_required_keys(self):
        report = SCORER.session_health("s1", self.COLLECTED, self.FIELDS, self.REQUIRED)
        d = report.to_dict()
        for key in ["session_id", "overall_score", "overall_band",
                    "needs_confirm", "high_confidence", "ready_for_output"]:
            assert key in d

    def test_all_high_confidence_ready_for_output(self):
        meta = {
            f: {"confidence": 0.95, "source_text": f"I am {v}", "confirmed": True}
            for f, v in self.COLLECTED.items()
        }
        report = SCORER.session_health("s1", self.COLLECTED, self.FIELDS, self.REQUIRED, meta)
        assert report.ready_for_output is True

    def test_low_confidence_not_ready(self):
        meta = {f: {"confidence": 0.20, "in_conflict": True} for f in self.COLLECTED}
        report = SCORER.session_health("s1", self.COLLECTED, self.FIELDS, self.REQUIRED, meta)
        assert report.ready_for_output is False

    def test_needs_confirm_populated(self):
        meta = {f: {"confidence": 0.20} for f in self.COLLECTED}
        report = SCORER.session_health("s1", self.COLLECTED, self.FIELDS, self.REQUIRED, meta)
        assert len(report.needs_confirm) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  9. score_all + backward compat
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreAll:

    def test_score_all_returns_dict(self):
        collected = {"age": 28, "name": "Priya"}
        fields    = {"age": {"type": "integer"}, "name": {"type": "text"}}
        results   = SCORER.score_all(collected, fields)
        assert "age"  in results
        assert "name" in results
        assert all(isinstance(v, ConfidenceScore) for v in results.values())

    def test_overall_session_confidence_backward_compat(self):
        collected = {"age": 28}
        fields    = {"age": {"type": "integer"}}
        results   = SCORER.score_all(collected, fields)
        avg = SCORER.overall_session_confidence(results)
        assert 0.0 <= avg <= 1.0

    def test_score_all_with_conflict_history(self):
        collected = {"age": 28}
        fields    = {"age": {"type": "integer"}}
        results_clean    = SCORER.score_all(collected, fields, conflict_history={"age": 0})
        results_conflict = SCORER.score_all(collected, fields, conflict_history={"age": 3})
        assert results_clean["age"].score > results_conflict["age"].score


# ─────────────────────────────────────────────────────────────────────────────
#  10. FieldTree — basic (no conditions)
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeBasic:

    def _ft(self, fields_config):
        return FieldTree(fields_config)

    def test_field_with_no_conditions_always_visible(self):
        ft = self._ft({"age": {"type": "integer", "required": True}})
        assert ft.is_visible("age", {}) is True
        assert ft.is_visible("age", {"age": 28}) is True

    def test_visible_fields_all_when_no_conditions(self):
        ft = self._ft({
            "name": {"type": "text"},
            "age":  {"type": "integer"},
        })
        assert set(ft.visible_fields({})) == {"name", "age"}

    def test_dependency_summary_empty_when_no_conditions(self):
        ft = self._ft({"age": {"type": "integer"}, "name": {"type": "text"}})
        assert ft.dependency_summary() == {}


# ─────────────────────────────────────────────────────────────────────────────
#  11. FieldTree — if_true
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeIfTrue:

    FIELDS = {
        "has_injury": {"type": "boolean", "required": True},
        "injury_desc": {
            "type": "text", "required": True,
            "if_true": "has_injury",
        },
    }

    def test_conditional_field_hidden_when_gate_not_set(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("injury_desc", {}) is False

    def test_conditional_field_hidden_when_gate_false(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("injury_desc", {"has_injury": False}) is False
        assert ft.is_visible("injury_desc", {"has_injury": "no"}) is False
        assert ft.is_visible("injury_desc", {"has_injury": "false"}) is False

    def test_conditional_field_visible_when_gate_true(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("injury_desc", {"has_injury": True}) is True
        assert ft.is_visible("injury_desc", {"has_injury": "yes"}) is True
        assert ft.is_visible("injury_desc", {"has_injury": "1"}) is True

    def test_dependency_summary_shows_gate(self):
        ft = FieldTree(self.FIELDS)
        deps = ft.dependency_summary()
        assert "injury_desc" in deps
        assert "has_injury" in deps["injury_desc"]


# ─────────────────────────────────────────────────────────────────────────────
#  12. FieldTree — if_value_is
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeIfValueIs:

    FIELDS = {
        "smoker": {"type": "boolean", "required": True},
        "quit_date": {
            "type": "date", "required": False,
            "if_value_is": {"field": "smoker", "value": "no"},
        },
        "cigarettes_per_day": {
            "type": "integer", "required": True,
            "if_value_is": {"field": "smoker", "value": "yes"},
        },
    }

    def test_hidden_when_gate_field_not_collected(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("quit_date", {}) is False
        assert ft.is_visible("cigarettes_per_day", {}) is False

    def test_visible_when_value_matches(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("quit_date", {"smoker": "no"}) is True
        assert ft.is_visible("cigarettes_per_day", {"smoker": "yes"}) is True

    def test_hidden_when_value_does_not_match(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("quit_date", {"smoker": "yes"}) is False
        assert ft.is_visible("cigarettes_per_day", {"smoker": "no"}) is False

    def test_case_insensitive_comparison(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("quit_date", {"smoker": "NO"}) is True
        assert ft.is_visible("quit_date", {"smoker": "No"}) is True


# ─────────────────────────────────────────────────────────────────────────────
#  13. FieldTree — if_value_in
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeIfValueIn:

    FIELDS = {
        "primary_goal": {"type": "text", "required": True},
        "target_muscles": {
            "type": "text", "required": True,
            "if_value_in": {
                "field": "primary_goal",
                "values": ["build muscle", "powerlifting"],
            },
        },
    }

    def test_visible_when_value_in_list(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("target_muscles", {"primary_goal": "build muscle"}) is True
        assert ft.is_visible("target_muscles", {"primary_goal": "powerlifting"}) is True

    def test_hidden_when_value_not_in_list(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("target_muscles", {"primary_goal": "lose weight"}) is False
        assert ft.is_visible("target_muscles", {"primary_goal": "run 5k"}) is False

    def test_hidden_when_gate_not_collected(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("target_muscles", {}) is False


# ─────────────────────────────────────────────────────────────────────────────
#  14. FieldTree — if_value_not
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeIfValueNot:

    FIELDS = {
        "pain_level": {"type": "text", "required": True},
        "pain_treatment": {
            "type": "text", "required": True,
            "if_value_not": {"field": "pain_level", "value": "none"},
        },
    }

    def test_visible_when_value_differs(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("pain_treatment", {"pain_level": "moderate"}) is True
        assert ft.is_visible("pain_treatment", {"pain_level": "severe"}) is True

    def test_hidden_when_value_matches_excluded(self):
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("pain_treatment", {"pain_level": "none"}) is False

    def test_visible_when_gate_not_collected(self):
        # Not set → not equal to "none" → condition passes
        ft = FieldTree(self.FIELDS)
        assert ft.is_visible("pain_treatment", {}) is True


# ─────────────────────────────────────────────────────────────────────────────
#  15. FieldTree — if_numeric_gt / if_numeric_lt
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeNumeric:

    FIELDS_GT = {
        "age": {"type": "integer", "required": True},
        "recovery_focus": {
            "type": "text", "required": False,
            "if_numeric_gt": {"field": "age", "value": 45},
        },
    }
    FIELDS_LT = {
        "bmi": {"type": "number", "required": True},
        "underweight_plan": {
            "type": "text", "required": True,
            "if_numeric_lt": {"field": "bmi", "value": 18.5},
        },
    }

    def test_if_numeric_gt_visible_when_above(self):
        ft = FieldTree(self.FIELDS_GT)
        assert ft.is_visible("recovery_focus", {"age": 50}) is True
        assert ft.is_visible("recovery_focus", {"age": 46}) is True

    def test_if_numeric_gt_hidden_when_at_threshold(self):
        ft = FieldTree(self.FIELDS_GT)
        assert ft.is_visible("recovery_focus", {"age": 45}) is False

    def test_if_numeric_gt_hidden_when_below(self):
        ft = FieldTree(self.FIELDS_GT)
        assert ft.is_visible("recovery_focus", {"age": 30}) is False

    def test_if_numeric_lt_visible_when_below(self):
        ft = FieldTree(self.FIELDS_LT)
        assert ft.is_visible("underweight_plan", {"bmi": 16.0}) is True

    def test_if_numeric_lt_hidden_when_above(self):
        ft = FieldTree(self.FIELDS_LT)
        assert ft.is_visible("underweight_plan", {"bmi": 22.0}) is False

    def test_hidden_when_gate_not_collected(self):
        ft = FieldTree(self.FIELDS_GT)
        assert ft.is_visible("recovery_focus", {}) is False


# ─────────────────────────────────────────────────────────────────────────────
#  16. FieldTree — compound (if_all_of / if_any_of)
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeCompound:

    FIELDS_ALL = {
        "goal":   {"type": "text"},
        "active": {"type": "text"},
        "elite_program": {
            "type": "text", "required": True,
            "if_all_of": [
                {"if_value_is": {"field": "goal",   "value": "build muscle"}},
                {"if_value_is": {"field": "active", "value": "very active"}},
            ],
        },
    }
    FIELDS_ANY = {
        "activity_level": {"type": "text"},
        "beginner_guide": {
            "type": "boolean", "required": False,
            "if_any_of": [
                {"if_value_is": {"field": "activity_level", "value": "sedentary"}},
                {"if_value_is": {"field": "activity_level", "value": "lightly active"}},
            ],
        },
    }

    def test_if_all_of_visible_when_all_pass(self):
        ft = FieldTree(self.FIELDS_ALL)
        collected = {"goal": "build muscle", "active": "very active"}
        assert ft.is_visible("elite_program", collected) is True

    def test_if_all_of_hidden_when_one_fails(self):
        ft = FieldTree(self.FIELDS_ALL)
        assert ft.is_visible("elite_program", {"goal": "build muscle", "active": "sedentary"}) is False
        assert ft.is_visible("elite_program", {"goal": "lose weight",  "active": "very active"}) is False

    def test_if_any_of_visible_when_one_passes(self):
        ft = FieldTree(self.FIELDS_ANY)
        assert ft.is_visible("beginner_guide", {"activity_level": "sedentary"}) is True
        assert ft.is_visible("beginner_guide", {"activity_level": "lightly active"}) is True

    def test_if_any_of_hidden_when_none_pass(self):
        ft = FieldTree(self.FIELDS_ANY)
        assert ft.is_visible("beginner_guide", {"activity_level": "very active"}) is False
        assert ft.is_visible("beginner_guide", {"activity_level": "moderately active"}) is False


# ─────────────────────────────────────────────────────────────────────────────
#  17. FieldTree navigation
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldTreeNavigation:

    FIELDS = {
        "name":         {"type": "text",    "required": True},
        "has_injury":   {"type": "boolean", "required": True},
        "injury_desc":  {"type": "text",    "required": True, "if_true": "has_injury"},
        "diet":         {"type": "text",    "required": False},
    }

    def test_next_required_respects_conditions(self):
        ft = FieldTree(self.FIELDS)
        # injury_desc should NOT appear before has_injury is answered
        nxt = ft.next_required({"name": "Alex"})
        assert nxt == "has_injury"   # not injury_desc

    def test_injury_desc_appears_after_has_injury_true(self):
        ft = FieldTree(self.FIELDS)
        nxt = ft.next_required({"name": "Alex", "has_injury": True})
        assert nxt == "injury_desc"

    def test_injury_desc_skipped_when_has_injury_false(self):
        ft = FieldTree(self.FIELDS)
        # has_injury=false → injury_desc never visible → all_required_collected
        all_done = ft.all_required_collected({"name": "Alex", "has_injury": False})
        assert all_done is True

    def test_all_required_not_collected_with_missing(self):
        ft = FieldTree(self.FIELDS)
        assert ft.all_required_collected({"name": "Alex"}) is False

    def test_next_optional_returns_diet(self):
        ft = FieldTree(self.FIELDS)
        nxt = ft.next_optional({"name": "Alex", "has_injury": False})
        assert nxt == "diet"

    def test_next_optional_none_when_max_reached(self):
        ft = FieldTree(self.FIELDS)
        nxt = ft.next_optional(
            {"name": "Alex", "has_injury": False},
            asked_optional={"diet"},
            max_optional=1,
        )
        assert nxt is None


# ─────────────────────────────────────────────────────────────────────────────
#  18. Reasoner uses FieldTree
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonerWithFieldTree:

    def _make_state(self, fields_config, collected=None):
        from truenorth.core.yaml_loader import YAMLLoader
        import yaml, tempfile, os
        goal = {
            "id": "test",
            "fields": [
                {"name": fn, **{k: v for k, v in cfg.items()}}
                for fn, cfg in fields_config.items()
            ],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output": {"format": "json"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            import yaml as _yaml
            _yaml.dump(goal, f)
            tmp = f.name
        try:
            config = YAMLLoader.load(tmp)
        finally:
            os.unlink(tmp)

        state = GraphState.from_goal_config(config, session_id="r-test")
        for k, v in (collected or {}).items():
            state.set_field(k, v)
        return state

    def test_reasoner_asks_gate_first(self):
        fields = {
            "has_injury": {"type": "boolean", "required": True,
                           "question": "Do you have an injury?"},
            "injury_desc": {"type": "text", "required": True,
                            "question": "Describe it.",
                            "if_true": "has_injury"},
        }
        state    = self._make_state(fields)
        reasoner = Reasoner()
        decision = reasoner.decide(state)
        assert decision.action == ReasonerAction.ASK_FIELD
        assert decision.target_field == "has_injury"

    def test_reasoner_skips_conditional_when_gate_false(self):
        fields = {
            "has_injury": {"type": "boolean", "required": True,
                           "question": "Do you have an injury?"},
            "injury_desc": {"type": "text", "required": True,
                            "question": "Describe it.",
                            "if_true": "has_injury"},
        }
        state    = self._make_state(fields, collected={"has_injury": "no"})
        reasoner = Reasoner()
        decision = reasoner.decide(state)
        # has_injury=no → injury_desc never visible → generate output
        assert decision.action == ReasonerAction.GENERATE_OUTPUT

    def test_reasoner_asks_conditional_when_gate_true(self):
        fields = {
            "has_injury": {"type": "boolean", "required": True,
                           "question": "Do you have an injury?"},
            "injury_desc": {"type": "text", "required": True,
                            "question": "Describe it.",
                            "if_true": "has_injury"},
        }
        state    = self._make_state(fields, collected={"has_injury": "yes"})
        reasoner = Reasoner()
        decision = reasoner.decide(state)
        assert decision.action == ReasonerAction.ASK_FIELD
        assert decision.target_field == "injury_desc"


# ─────────────────────────────────────────────────────────────────────────────
#  19. End-to-end: fitness YAML with conditional fields
# ─────────────────────────────────────────────────────────────────────────────

class TestFitnessYAMLConditions:

    FITNESS = Path(__file__).parent.parent.parent / "examples" / "goals" / "fitness_plan.yaml"

    @pytest.fixture
    def ft(self):
        if not self.FITNESS.exists():
            pytest.skip("fitness_plan.yaml not found")
        from truenorth.core.yaml_loader import YAMLLoader
        cfg = YAMLLoader.load(str(self.FITNESS))
        fields = {f["name"]: f for f in cfg["fields"]}
        return FieldTree(fields)

    def test_muscle_specific_fields_hidden_for_weight_loss(self, ft):
        collected = {"primary_goal": "lose weight"}
        assert ft.is_visible("target_muscle_groups", collected) is False
        assert ft.is_visible("current_lifts_kg", collected) is False

    def test_muscle_specific_fields_visible_for_muscle_goal(self, ft):
        collected = {"primary_goal": "build muscle"}
        assert ft.is_visible("target_muscle_groups", collected) is True

    def test_5k_field_hidden_for_muscle_goal(self, ft):
        collected = {"primary_goal": "build muscle"}
        assert ft.is_visible("target_5k_time_minutes", collected) is False

    def test_5k_field_visible_for_run_goal(self, ft):
        collected = {"primary_goal": "run 5k"}
        assert ft.is_visible("target_5k_time_minutes", collected) is True

    def test_injury_desc_hidden_until_has_injury_true(self, ft):
        assert ft.is_visible("injury_description", {}) is False
        assert ft.is_visible("injury_description", {"has_injury": "no"}) is False
        assert ft.is_visible("injury_description", {"has_injury": "yes"}) is True

    def test_recovery_focus_hidden_for_young_user(self, ft):
        assert ft.is_visible("recovery_focus", {"age": 30}) is False

    def test_recovery_focus_visible_for_older_user(self, ft):
        assert ft.is_visible("recovery_focus", {"age": 50}) is True

    def test_beginner_guidance_visible_for_sedentary(self, ft):
        assert ft.is_visible("beginner_guidance", {"activity_level": "sedentary"}) is True

    def test_beginner_guidance_hidden_for_active_user(self, ft):
        assert ft.is_visible("beginner_guidance", {"activity_level": "very active"}) is False

    def test_has_injury_required_always_visible(self, ft):
        assert ft.is_visible("has_injury", {}) is True

    def test_dependency_summary_contains_conditional_fields(self, ft):
        deps = ft.dependency_summary()
        assert "injury_description" in deps
        assert "target_muscle_groups" in deps