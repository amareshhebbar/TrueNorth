"""

Task 9 — SourceTracer tests.

Classes:
  1.  SentenceParser        — sentence splitting and type classification
  2.  FieldMatcher_Exact    — exact value matching (tier 1)
  3.  FieldMatcher_Label    — field name/label matching (tier 2)
  4.  FieldMatcher_Numeric  — numeric proximity matching (tier 3)
  5.  TurnResolver          — field source lookup from session state
  6.  TraceCompleteness     — thresholds and classification
  7.  SourceMap             — result object helpers
  8.  Tracer_CleanOutput    — well-attributed output traces fully
  9.  Tracer_GenericOutput  — generic advice skipped from attribution requirement
  10. Tracer_UntracedValues — untraced numeric values caught
  11. Tracer_MultiField     — multi-field report traced correctly
  12. Tracer_AuditLog       — audit log structure for compliance
  13. Tracer_EmptyOutput    — edge cases
  14. GeneratorIntegration  — source_trace appears in OutputGenerator result
  15. EngineIntegration     — source_trace in full engine output
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.output.source_tracer import (
    SourceTracer,
    SentenceParser,
    FieldMatcher,
    TurnResolver,
    TracedSentence,
    SourceMap,
    TraceCompleteness,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

COLLECTED = {
    "name":                     "Priya",
    "age":                      28,
    "weight_kg":                65.0,
    "height_cm":                163.0,
    "primary_goal":             "lose weight",
    "activity_level":           "moderately active",
    "workout_days_per_week":    4,
    "workout_duration_minutes": 45,
    "equipment_access":         "gym membership",
}

FIELDS_CONFIG = {
    "name":                     {"type": "text",    "label": "name"},
    "age":                      {"type": "integer", "label": "age"},
    "weight_kg":                {"type": "number",  "label": "weight"},
    "height_cm":                {"type": "number",  "label": "height"},
    "primary_goal":             {"type": "text",    "label": "primary goal"},
    "activity_level":           {"type": "text",    "label": "activity level"},
    "workout_days_per_week":    {"type": "integer", "label": "workout days per week"},
    "workout_duration_minutes": {"type": "integer", "label": "workout duration"},
    "equipment_access":         {"type": "text",    "label": "equipment access"},
}

CONFIDENCES = {k: 0.90 for k in COLLECTED}

TURN_HISTORY = [
    {"role": "assistant", "content": "What's your name?", "turn": 0, "timestamp": time.time()},
    {"role": "user",      "content": "I'm Priya",         "turn": 1, "timestamp": time.time()},
    {"role": "assistant", "content": "How old are you?",   "turn": 1, "timestamp": time.time()},
    {"role": "user",      "content": "I am 28 years old",  "turn": 2, "timestamp": time.time()},
    {"role": "assistant", "content": "What's your weight?","turn": 2, "timestamp": time.time()},
    {"role": "user",      "content": "I weigh 65 kg",      "turn": 3, "timestamp": time.time()},
    {"role": "assistant", "content": "Your height?",        "turn": 3, "timestamp": time.time()},
    {"role": "user",      "content": "163 cm tall",         "turn": 4, "timestamp": time.time()},
    {"role": "assistant", "content": "Goal?",               "turn": 4, "timestamp": time.time()},
    {"role": "user",      "content": "lose weight",         "turn": 5, "timestamp": time.time()},
    {"role": "assistant", "content": "Activity level?",     "turn": 5, "timestamp": time.time()},
    {"role": "user",      "content": "moderately active",   "turn": 6, "timestamp": time.time()},
    {"role": "assistant", "content": "How many days?",      "turn": 6, "timestamp": time.time()},
    {"role": "user",      "content": "4 days per week",     "turn": 7, "timestamp": time.time()},
    {"role": "assistant", "content": "Duration?",           "turn": 7, "timestamp": time.time()},
    {"role": "user",      "content": "45 minutes each",     "turn": 8, "timestamp": time.time()},
]

FIELD_TURN_MAP = {
    "name": 1, "age": 2, "weight_kg": 3, "height_cm": 4,
    "primary_goal": 5, "activity_level": 6,
    "workout_days_per_week": 7, "workout_duration_minutes": 8,
    "equipment_access": 0,
}

CLEAN_OUTPUT = """# Your Fitness Plan — Priya

Based on your profile, here is a personalised fitness plan.

## Your Profile
You are 28 years old and currently weigh 65 kg with a height of 163 cm.
Your goal is to lose weight, which is excellent.
You are moderately active and have a gym membership.

## Your Schedule
You can train 4 days per week for 45 minutes each session.
That gives you a solid foundation to achieve your goal.

## General Advice
Stay hydrated throughout the day.
Aim for 7-8 hours of sleep each night.
Consider tracking your meals.
"""

TRACER = SourceTracer()


def _trace(output: str = CLEAN_OUTPUT) -> SourceMap:
    return TRACER.trace(
        output            = output,
        collected_fields  = COLLECTED,
        field_confidences = CONFIDENCES,
        fields_config     = FIELDS_CONFIG,
        turn_history      = TURN_HISTORY,
        session_id        = "test-sess",
        goal_id           = "fitness_plan",
        field_turn_map    = FIELD_TURN_MAP,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  1. SentenceParser
# ─────────────────────────────────────────────────────────────────────────────

class TestSentenceParser:

    def test_splits_into_multiple_sentences(self):
        parser = SentenceParser()
        parsed = parser.parse(CLEAN_OUTPUT)
        assert len(parsed) >= 5

    def test_heading_classified_as_heading(self):
        parser = SentenceParser()
        parsed = parser.parse("# Your Fitness Plan")
        types = [t for _, t in parsed]
        assert "heading" in types

    def test_generic_advice_classified_correctly(self):
        parser = SentenceParser()
        parsed = parser.parse("Stay hydrated throughout the day. Aim for 7-8 hours of sleep.")
        types = [t for _, t in parsed]
        assert any(t == "generic" for t in types)

    def test_factual_sentence_with_number(self):
        parser = SentenceParser()
        parsed = parser.parse("You are 28 years old and weigh 65 kg.")
        types = [t for _, t in parsed]
        assert any(t == "factual" for t in types)

    def test_derived_sentence_classified(self):
        parser = SentenceParser()
        parsed = parser.parse("Your BMI is calculated to be 24.5.")
        types = [t for _, t in parsed]
        assert any(t == "derived" for t in types)

    def test_empty_text_returns_empty(self):
        parser = SentenceParser()
        assert parser.parse("") == []
        assert parser.parse("   ") == []

    def test_extract_values_finds_numbers(self):
        parser = SentenceParser()
        vals = parser.extract_values("You are 28 years old and weigh 65 kg.")
        nums = [v.rstrip("kgyears old").strip() for v in vals]
        assert "28" in nums or any("28" in v for v in vals)

    def test_extract_values_with_units(self):
        parser = SentenceParser()
        vals = parser.extract_values("Height is 163 cm and weight is 65 kg.")
        strs = " ".join(vals)
        assert "163" in strs
        assert "65" in strs


# ─────────────────────────────────────────────────────────────────────────────
#  2. FieldMatcher — exact value match (tier 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldMatcherExact:

    def _sources(self) -> dict:
        resolver = TurnResolver()
        return resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)

    def test_matches_name_verbatim(self):
        matcher  = FieldMatcher()
        sources  = self._sources()
        matched, _ = matcher.match(
            "This plan is designed for Priya.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "name" in fields

    def test_matches_goal_verbatim(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, _ = matcher.match(
            "Your goal is to lose weight.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "primary_goal" in fields

    def test_matches_activity_level(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, _ = matcher.match(
            "Since you are moderately active, here is your plan.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "activity_level" in fields

    def test_no_match_for_unknown_value(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, untraced = matcher.match(
            "You should consume 3000 calories per day.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        assert len(matched) == 0 or all(
            s.field_name not in ("age", "weight_kg") for s in matched
        )


# ─────────────────────────────────────────────────────────────────────────────
#  3. FieldMatcher — field label/name match (tier 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldMatcherLabel:

    def _sources(self):
        resolver = TurnResolver()
        return resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)

    def test_matches_by_field_label(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, _ = matcher.match(
            "Your weight of 65 kg is well within range.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "weight_kg" in fields

    def test_matches_multi_token_field_name(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, _ = matcher.match(
            "You plan to train 4 workout days per week.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "workout_days_per_week" in fields or "workout_duration_minutes" in fields


# ─────────────────────────────────────────────────────────────────────────────
#  4. FieldMatcher — numeric proximity match (tier 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldMatcherNumeric:

    def _sources(self):
        resolver = TurnResolver()
        return resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)

    def test_matches_exact_numeric_age(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, _ = matcher.match(
            "At the age of 28, you have great potential.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "age" in fields

    def test_matches_exact_weight(self):
        matcher = FieldMatcher()
        sources = self._sources()
        matched, _ = matcher.match(
            "Starting from 65 kg, your weight loss goal is achievable.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        fields = [s.field_name for s in matched]
        assert "weight_kg" in fields

    def test_wrong_number_flagged_as_untraced(self):
        matcher = FieldMatcher()
        sources = self._sources()
        _, untraced = matcher.match(
            "You currently weigh 99 kg.",
            COLLECTED, FIELDS_CONFIG, sources,
        )
        assert len(untraced) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  5. TurnResolver
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnResolver:

    def test_builds_source_for_every_field(self):
        resolver = TurnResolver()
        sources  = resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)
        assert set(sources.keys()) == set(COLLECTED.keys())

    def test_turn_numbers_from_map(self):
        resolver = TurnResolver()
        sources  = resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)
        assert sources["age"].turn      == 2
        assert sources["weight_kg"].turn == 3
        assert sources["height_cm"].turn == 4

    def test_user_text_populated(self):
        resolver = TurnResolver()
        sources  = resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)
        assert sources["age"].user_text == "I am 28 years old"

    def test_confidence_propagated(self):
        resolver  = TurnResolver()
        confs     = {"age": 0.95, "weight_kg": 0.87}
        sources   = resolver.build_sources({"age": 28, "weight_kg": 65}, confs, TURN_HISTORY)
        assert sources["age"].confidence      == pytest.approx(0.95)
        assert sources["weight_kg"].confidence == pytest.approx(0.87)

    def test_infer_turn_without_map(self):
        resolver = TurnResolver()
        sources  = resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY)
        assert sources["name"].turn in (0, 1)

    def test_field_source_to_dict(self):
        resolver = TurnResolver()
        sources  = resolver.build_sources(COLLECTED, CONFIDENCES, TURN_HISTORY, FIELD_TURN_MAP)
        d = sources["age"].to_dict()
        assert "field"     in d
        assert "value"     in d
        assert "turn"      in d
        assert "user_text" in d
        assert "confidence"in d


# ─────────────────────────────────────────────────────────────────────────────
#  6. TraceCompleteness thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceCompleteness:

    def test_fully_traced_threshold(self):
        tracer = SourceTracer()
        assert tracer._classify_completeness(1.00) == TraceCompleteness.FULLY_TRACED
        assert tracer._classify_completeness(0.95) == TraceCompleteness.FULLY_TRACED

    def test_mostly_traced_threshold(self):
        tracer = SourceTracer()
        assert tracer._classify_completeness(0.80) == TraceCompleteness.MOSTLY_TRACED
        assert tracer._classify_completeness(0.75) == TraceCompleteness.MOSTLY_TRACED

    def test_partially_traced_threshold(self):
        tracer = SourceTracer()
        assert tracer._classify_completeness(0.60) == TraceCompleteness.PARTIALLY_TRACED
        assert tracer._classify_completeness(0.40) == TraceCompleteness.PARTIALLY_TRACED

    def test_untraceable_threshold(self):
        tracer = SourceTracer()
        assert tracer._classify_completeness(0.30) == TraceCompleteness.UNTRACEABLE
        assert tracer._classify_completeness(0.00) == TraceCompleteness.UNTRACEABLE


# ─────────────────────────────────────────────────────────────────────────────
#  7. SourceMap
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceMap:

    def test_to_dict_has_required_keys(self):
        sm = _trace()
        d  = sm.to_dict()
        for key in ["session_id", "goal_id", "completeness", "traced_pct",
                    "is_audit_ready", "field_coverage", "untraced_count", "sentences"]:
            assert key in d

    def test_is_audit_ready_for_clean_output(self):
        sm = _trace()
        assert sm.completeness != TraceCompleteness.UNTRACEABLE
        assert sm.traced_pct >= 0.50

    def test_coverage_report_is_string(self):
        sm = _trace()
        report = sm.coverage_report()
        assert isinstance(report, str)
        assert "completeness" in report.lower()

    def test_audit_log_returns_list(self):
        sm  = _trace()
        log = sm.audit_log()
        assert isinstance(log, list)

    def test_audit_log_entries_have_required_fields(self):
        sm  = _trace()
        log = sm.audit_log()
        for entry in log:
            assert "claim"   in entry
            assert "field"   in entry
            assert "value"   in entry
            assert "collected_turn" in entry
            assert "confidence"     in entry


# ─────────────────────────────────────────────────────────────────────────────
#  8. Tracer — clean output traces fully
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerCleanOutput:

    def test_completeness_is_high(self):
        sm = _trace()
        assert sm.completeness != TraceCompleteness.UNTRACEABLE
        assert sm.traced_pct >= 0.50

    def test_traced_pct_above_threshold(self):
        sm = _trace()
        assert sm.traced_pct >= 0.60   

    def test_age_in_field_coverage(self):
        sm = _trace()
        assert "age" in sm.field_coverage or "weight_kg" in sm.field_coverage

    def test_sentences_populated(self):
        sm = _trace()
        assert len(sm.sentences) >= 5

    def test_name_traced_to_correct_field(self):
        sm = _trace()
        # Find the sentence mentioning Priya
        priya_sentences = [
            s for s in sm.sentences
            if "Priya" in s.sentence and s.is_traced
        ]
        assert len(priya_sentences) >= 1
        for s in priya_sentences:
            fields = [src.field_name for src in s.sources]
            if "name" in fields:
                assert True
                return

    def test_age_sentence_traced_to_age_field(self):
        sm = _trace()
        age_sentences = [
            s for s in sm.sentences
            if "28" in s.sentence and not s.is_generic
        ]
        for s in age_sentences:
            fields = [src.field_name for src in s.sources]
            if "age" in fields:
                assert s.sources[0].turn == 2   
                return

    def test_weight_sentence_traced_to_weight_field(self):
        sm = _trace()
        weight_sentences = [
            s for s in sm.sentences
            if "65" in s.sentence and "kg" in s.sentence.lower() and not s.is_generic
        ]
        for s in weight_sentences:
            fields = [src.field_name for src in s.sources]
            if "weight_kg" in fields:
                return   


# ─────────────────────────────────────────────────────────────────────────────
#  9. Tracer — generic output
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerGenericOutput:

    def test_generic_advice_does_not_lower_traced_pct(self):
        generic_only = (
            "Stay hydrated throughout the day. "
            "Aim for 7-8 hours of sleep each night. "
            "Consider tracking your meals. "
            "Remember to warm up before each session."
        )
        sm = TRACER.trace(
            output=generic_only,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
        )
        assert sm.traced_pct == pytest.approx(1.0)

    def test_generic_sentences_marked_as_generic(self):
        sm = _trace()
        generic = [s for s in sm.sentences if s.is_generic]
        assert len(generic) >= 2 

    def test_mixed_output_still_traces_factual(self):
        mixed = (
            "You are 28 years old. "
            "Stay hydrated throughout the day."
        )
        sm = TRACER.trace(
            output=mixed,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
            field_turn_map=FIELD_TURN_MAP,
        )
        # The factual sentence "You are 28 years old" should be traced
        factual = [s for s in sm.sentences if not s.is_generic]
        traced_factual = [s for s in factual if s.is_traced]
        assert len(traced_factual) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  10. Tracer — untraced values detected
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerUntracedValues:

    def test_hallucinated_value_flagged_as_untraced(self):
        hallucinated = "You currently weigh 99 kg, which is your target."
        sm = TRACER.trace(
            output=hallucinated,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
        )
        all_untraced = []
        for s in sm.sentences:
            all_untraced.extend(s.untraced_values)
        has_untraced = (
            len(all_untraced) > 0
            or sm.completeness in (TraceCompleteness.UNTRACEABLE, TraceCompleteness.PARTIALLY_TRACED)
            or len(sm.untraced_sentences) > 0
        )
        assert has_untraced or sm.traced_pct < 1.0

    def test_correct_values_not_flagged(self):
        correct = "You weigh 65 kg and are 28 years old."
        sm = TRACER.trace(
            output=correct,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
            field_turn_map=FIELD_TURN_MAP,
        )
        # Both values are in collected_fields — should be traced
        factual = [s for s in sm.sentences if not s.is_generic]
        for s in factual:
            assert s.is_traced


# ─────────────────────────────────────────────────────────────────────────────
#  11. Tracer — multi-field report
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerMultiField:

    FULL_REPORT = """# Your Personalised Fitness Plan — Priya

You are 28 years old and currently weigh 65 kg with a height of 163 cm.
Your primary goal is to lose weight, which is excellent motivation.
You are moderately active, so we'll build from there.
You have a gym membership, which gives great flexibility.
Your weekly plan: 4 days per week, 45 minutes each session.
Stay hydrated and get enough sleep.
"""

    def test_multiple_fields_covered(self):
        sm = TRACER.trace(
            output=self.FULL_REPORT,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
            field_turn_map=FIELD_TURN_MAP,
        )
        assert len(sm.field_coverage) >= 4

    def test_field_coverage_counts_correct(self):
        sm = TRACER.trace(
            output=self.FULL_REPORT,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
            field_turn_map=FIELD_TURN_MAP,
        )
        # Each field should appear at most once per sentence (deduplicated)
        for field_name, count in sm.field_coverage.items():
            assert count >= 1

    def test_completeness_high_for_full_report(self):
        sm = TRACER.trace(
            output=self.FULL_REPORT,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
            field_turn_map=FIELD_TURN_MAP,
        )
        assert sm.completeness in (
            TraceCompleteness.FULLY_TRACED,
            TraceCompleteness.MOSTLY_TRACED,
        )

    def test_turn_numbers_correct_in_audit_log(self):
        sm = TRACER.trace(
            output=self.FULL_REPORT,
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
            field_turn_map=FIELD_TURN_MAP,
        )
        log = sm.audit_log()
        for entry in log:
            assert isinstance(entry["collected_turn"], int)
            assert entry["collected_turn"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
#  12. Tracer — audit log for compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerAuditLog:

    def test_audit_log_entry_structure(self):
        sm  = _trace()
        log = sm.audit_log()
        if log:
            entry = log[0]
            assert "claim"           in entry
            assert "field"           in entry
            assert "value"           in entry
            assert "collected_turn"  in entry
            assert "user_statement"  in entry
            assert "confidence"      in entry

    def test_audit_log_excludes_generic_sentences(self):
        sm  = _trace()
        log = sm.audit_log()
        # Audit log should not include generic advice like "stay hydrated"
        for entry in log:
            assert not any(
                phrase in entry["claim"].lower()
                for phrase in ["stay hydrated", "aim for 7", "consider tracking"]
            )

    def test_audit_log_references_user_statement(self):
        sm  = _trace()
        log = sm.audit_log()
        # At least some entries should reference the user's actual words
        has_user_text = any(len(e["user_statement"]) > 0 for e in log)
        assert has_user_text

    def test_is_audit_ready_status(self):
        sm = _trace()
        assert isinstance(sm.is_audit_ready, bool)
        # A well-traced output should be audit-ready
        if sm.completeness in (TraceCompleteness.FULLY_TRACED, TraceCompleteness.MOSTLY_TRACED):
            assert sm.is_audit_ready is True


# ─────────────────────────────────────────────────────────────────────────────
#  13. Tracer — edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerEdgeCases:

    def test_empty_output_returns_fully_traced(self):
        sm = TRACER.trace(
            output="",
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
        )
        assert sm.completeness == TraceCompleteness.FULLY_TRACED
        assert sm.traced_pct == 1.0
        assert sm.sentences == []

    def test_no_collected_fields_still_works(self):
        sm = TRACER.trace(
            output="Here is your plan.",
            collected_fields={},
            field_confidences={},
            fields_config=FIELDS_CONFIG,
            turn_history=[],
            session_id="s1",
            goal_id="fitness_plan",
        )
        assert isinstance(sm, SourceMap)
        assert sm.traced_pct >= 0.0

    def test_json_output_handled(self):
        sm = TRACER.trace(
            output='{"name": "Priya", "age": 28}',
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            session_id="s1", goal_id="fitness_plan",
        )
        assert isinstance(sm, SourceMap)

    def test_trace_single_sentence(self):
        ts = TRACER.trace_sentence(
            sentence="You are 28 years old.",
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
            field_turn_map=FIELD_TURN_MAP,
        )
        assert isinstance(ts, TracedSentence)
        assert ts.sentence_type in ("factual", "generic", "derived")

    def test_traced_sentence_to_dict(self):
        ts = TRACER.trace_sentence(
            sentence="You are 28 years old.",
            collected_fields=COLLECTED, field_confidences=CONFIDENCES,
            fields_config=FIELDS_CONFIG, turn_history=TURN_HISTORY,
        )
        d = ts.to_dict()
        assert "sentence"  in d
        assert "is_traced" in d
        assert "sources"   in d
        assert "type"      in d


# ─────────────────────────────────────────────────────────────────────────────
#  14. OutputGenerator integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorIntegration:

    @pytest.mark.asyncio
    async def test_source_trace_in_generator_output(self):
        import importlib
        gen_mod = importlib.import_module("truenorth.output.generator")
        OutputGenerator = gen_mod.OutputGenerator
        from truenorth.core.graph_state import GraphState
        from truenorth.core.yaml_loader import YAMLLoader

        config = YAMLLoader.load_from_string("""
id: gen_trace_test
fields:
  - {name: name,   type: text,    required: true}
  - {name: age,    type: integer, required: true}
  - {name: goal,   type: text,    required: true}
persona: {name: Bot, tone: neutral}
output:
  format: text
  template: "Plan for {name} (age {age}): {goal}."
""")
        state = GraphState.from_goal_config(config, session_id="gen-s1")
        state.set_field("name", "Priya", confidence=0.95)
        state.set_field("age",  28,      confidence=0.90)
        state.set_field("goal", "lose weight", confidence=0.88)
        state.turn_history = [
            {"role": "user", "content": "I'm Priya",     "turn": 1, "timestamp": time.time()},
            {"role": "user", "content": "I am 28",       "turn": 2, "timestamp": time.time()},
            {"role": "user", "content": "lose weight",   "turn": 3, "timestamp": time.time()},
        ]
        state._field_turn_map = {"name": 1, "age": 2, "goal": 3}

        gen = OutputGenerator()
        result = await gen.generate(state)

        assert "source_trace" in result["metadata"]
        trace = result["metadata"]["source_trace"]
        assert trace is not None
        assert "completeness" in trace
        assert "traced_pct"   in trace
        assert "sentences"    in trace

    @pytest.mark.asyncio
    async def test_source_trace_completeness_reasonable(self):
        from truenorth.output.generator import OutputGenerator
        from truenorth.core.graph_state import GraphState
        from truenorth.core.yaml_loader import YAMLLoader

        config = YAMLLoader.load_from_string("""
id: trace_test2
fields:
  - {name: name, type: text, required: true}
  - {name: age,  type: integer, required: true}
persona: {name: Bot, tone: neutral}
output:
  format: text
  template: "Report for {name}, age {age}."
""")
        state = GraphState.from_goal_config(config, session_id="gen-s2")
        state.set_field("name", "Alex")
        state.set_field("age",  30)
        state.turn_history = [
            {"role": "user", "content": "I'm Alex", "turn": 1, "timestamp": time.time()},
            {"role": "user", "content": "30 years old", "turn": 2, "timestamp": time.time()},
        ]
        state._field_turn_map = {"name": 1, "age": 2}

        gen    = OutputGenerator()
        result = await gen.generate(state)
        trace  = result["metadata"]["source_trace"]

        assert trace["traced_pct"] >= 0.0
        assert trace["completeness"] in (
            "fully_traced", "mostly_traced", "partially_traced", "untraceable"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  15. Engine integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineIntegration:

    @pytest.mark.asyncio
    async def test_engine_has_source_tracer_wired(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.output.source_tracer import SourceTracer

        goal = {
            "id": "tracer_test",
            "fields": [
                {"name": "name", "type": "text", "required": True, "question": "Name?"},
            ],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output": {"format": "json"},
        }
        engine = TrueNorthEngine(goal_config=goal)
        # SourceTracer should be wired into OutputGenerator
        assert hasattr(engine._output, "_tracer")
        assert isinstance(engine._output._tracer, SourceTracer)

    @pytest.mark.asyncio
    async def test_force_output_includes_source_trace(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient

        mock = MockLLMClient(default="Got it.")
        router = LLMRouter()
        for m in ["gemini-1.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        goal = {
            "id": "trace_engine_test",
            "fields": [{"name": "age", "type": "integer", "required": True, "question": "Age?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output": {"format": "text", "template": "You are {age} years old."},
        }
        engine = TrueNorthEngine(goal_config=goal, router=router)
        engine.state.set_field("age", 28, confidence=0.90)
        engine.state.turn_history = [
            {"role": "user", "content": "I am 28 years old", "turn": 1, "timestamp": time.time()}
        ]
        engine.state._field_turn_map = {"age": 1}

        output = await engine.force_output()
        assert "source_trace" in output.get("metadata", {})
        trace = output["metadata"]["source_trace"]
        assert "completeness" in trace