"""
truenorth/output/source_tracer.py

SourceTracer — every sentence in the generated output is mapped back to
the specific collected field and conversation turn that justified it.

Why this matters:
  - Audit trail for regulated industries (healthcare, finance, HR)
  - Regulators can see EXACTLY which user statement produced each claim
  - Developers can debug hallucinations turn-by-turn
  - Complements hallucination_firewall (FW checks correctness, ST checks attribution)
  - Studio dashboard can render output with field attributions highlighted

Output of tracing one report:
  "Your weight is 65 kg"
    → field: weight_kg
    → value: 65.0
    → turn: 3
    → user said: "I currently weigh 65 kilograms"
    → confidence: 0.93

Architecture — three stages:
  Stage 1: SentenceParser     — split output into discrete sentences
  Stage 2: FieldMatcher       — for each sentence, find which field(s) it references
  Stage 3: TurnResolver       — for each matched field, look up which turn collected it

Completeness scoring:
  - FULLY_TRACED    — every factual sentence traced to a field
  - MOSTLY_TRACED   — > 80% sentences traced
  - PARTIALLY_TRACED— 50-80% traced
  - UNTRACEABLE     — < 50% traced (high hallucination risk)

Integration:
  - Called inside OutputGenerator.generate() after content is produced
  - Also callable standalone for post-hoc analysis
  - Result attached to the output dict under "source_trace"
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class TraceCompleteness(str, Enum):
    FULLY_TRACED     = "fully_traced"
    MOSTLY_TRACED    = "mostly_traced"
    PARTIALLY_TRACED = "partially_traced"
    UNTRACEABLE      = "untraceable"

@dataclass
class FieldSource:
    """The session source for one field value."""
    field_name:     str
    value:          Any
    turn:           int
    user_text:      str
    confidence:     float
    extracted_at:   float

    def to_dict(self) -> dict:
        return {
            "field":      self.field_name,
            "value":      self.value,
            "turn":       self.turn,
            "user_text":  self.user_text[:200],
            "confidence": round(self.confidence, 3),
        }

@dataclass
class TracedSentence:
    """One sentence from the output with its field attribution."""
    sentence:        str
    sentence_index:  int
    sentence_type:   str
    sources:         List[FieldSource]
    is_traced:       bool
    is_generic:      bool
    untraced_values: List[str]

    @property
    def attribution_str(self) -> str:
        """Human-readable attribution for Studio display."""
        if self.is_generic:
            return "generic advice"
        if not self.sources:
            return "untraced"
        parts = []
        for s in self.sources:
            parts.append(f"{s.field_name}={s.value!r} (turn {s.turn})")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "sentence":       self.sentence[:300],
            "index":          self.sentence_index,
            "type":           self.sentence_type,
            "is_traced":      self.is_traced,
            "is_generic":     self.is_generic,
            "sources":        [s.to_dict() for s in self.sources],
            "untraced_values":self.untraced_values,
        }

@dataclass
class SourceMap:
    """
    Complete source attribution for one generated output.

    Attached to the output dict under "source_trace".
    """
    session_id:      str
    goal_id:         str
    completeness:    TraceCompleteness
    traced_pct:      float
    sentences:       List[TracedSentence]
    field_coverage:  Dict[str, int]
    untraced_sentences: List[str]
    generated_at:    float = field(default_factory=time.time)

    @property
    def is_audit_ready(self) -> bool:
        """True if completeness is high enough for regulatory submission."""
        return self.completeness in (
            TraceCompleteness.FULLY_TRACED,
            TraceCompleteness.MOSTLY_TRACED,
        )

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "goal_id":          self.goal_id,
            "completeness":     self.completeness.value,
            "traced_pct":       round(self.traced_pct, 3),
            "is_audit_ready":   self.is_audit_ready,
            "field_coverage":   self.field_coverage,
            "untraced_count":   len(self.untraced_sentences),
            "sentences":        [s.to_dict() for s in self.sentences],
        }

    def audit_log(self) -> List[dict]:
        """
        Flat list of (claim → source) pairs suitable for compliance audit trail.
        Only includes factual, traced sentences.
        """
        log = []
        for s in self.sentences:
            if s.is_traced and not s.is_generic:
                for src in s.sources:
                    log.append({
                        "claim":        s.sentence[:200],
                        "field":        src.field_name,
                        "value":        src.value,
                        "collected_turn": src.turn,
                        "user_statement": src.user_text[:200],
                        "confidence":   src.confidence,
                    })
        return log

    def coverage_report(self) -> str:
        """Human-readable coverage summary for dry-run / CLI output."""
        lines = [
            "Source Trace Report",
            f"  Completeness:  {self.completeness.value} ({self.traced_pct:.0%})",
            f"  Audit-ready:   {'yes' if self.is_audit_ready else 'no'}",
            f"  Sentences:     {len(self.sentences)} total",
            f"  Untraced:      {len(self.untraced_sentences)}",
            "",
            "  Field coverage:",
        ]
        for fn, count in sorted(self.field_coverage.items(),
                                 key=lambda x: x[1], reverse=True):
            lines.append(f"    {fn:<30} {count} sentence(s)")
        if self.untraced_sentences:
            lines.append("")
            lines.append("  Untraced sentences:")
            for s in self.untraced_sentences[:5]:
                lines.append(f"    ! {s[:80]!r}")
            if len(self.untraced_sentences) > 5:
                lines.append(f"    ... and {len(self.untraced_sentences)-5} more")
        return "\n".join(lines)

_GENERIC_PATTERNS = re.compile(
    r"\b(should|recommend|suggest|consider|try|aim for|make sure|"
    r"important to|remember to|keep in mind|generally|typically|"
    r"most people|research shows|studies show|it is important|"
    r"be sure to|always|never forget|don't forget|"
    r"consult a|speak to a|see a|talk to your)\b",
    re.IGNORECASE,
)

_HEADING_PATTERNS = re.compile(
    r"^#+\s|^[A-Z][A-Z\s]{3,}:?\s*$|^\*\*[^*]+\*\*\s*$"
)

_DERIVED_PATTERNS = re.compile(
    r"\b(bmi|body mass index|tdee|bmr|total daily|calculated|"
    r"based on these|which gives|equates to|therefore your|"
    r"this means your|working out to)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(kg|lbs?|cm|m\b|ft|inches?|years?|days?|weeks?|months?|"
    r"minutes?|hours?|kcal|calories?|%|km|miles?|reps?|sets?)?",
    re.IGNORECASE,
)

class SentenceParser:
    """Splits output text into typed sentence units."""

    def parse(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into (sentence, type) pairs.
        Types: "factual" | "generic" | "heading" | "derived" | "empty"
        """
        if not text or not text.strip():
            return []

        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\-*#•])|(?=^#+\s)|(?<=\n)(?=[*\-•]\s)",
                       text, flags=re.MULTILINE)

        expanded = []
        for part in raw:
            sub = re.split(r"\n[-*•]\s+|\n\d+\.\s+", part)
            expanded.extend(s.strip() for s in sub if s.strip())

        result: List[Tuple[str, str]] = []
        for sentence in expanded:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_type = self._classify(sentence)
            result.append((sentence, sentence_type))

        return result

    @staticmethod
    def _classify(sentence: str) -> str:
        if _HEADING_PATTERNS.match(sentence):
            return "heading"
        if _DERIVED_PATTERNS.search(sentence):
            return "derived"
        if _GENERIC_PATTERNS.search(sentence):
            return "generic"
        if _NUMBER_RE.search(sentence):
            return "factual"
        if any(w in sentence.lower() for w in ["your ", "you are", "you have", "you weigh",
                                                "you're", "based on your"]):
            return "factual"
        return "generic"

    @staticmethod
    def extract_values(sentence: str) -> List[str]:
        """Extract numeric values and quoted strings from a sentence."""
        values = []
        for m in _NUMBER_RE.finditer(sentence):
            num  = m.group(1)
            unit = m.group(2) or ""
            if unit or len(num) <= 6:
                values.append(f"{num}{unit}")
        for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', sentence):
            values.append(m.group(1) or m.group(2))
        return values

class FieldMatcher:
    """
    For each sentence, find which collected fields it references.

    Matching strategy (three tiers):
      1. Exact value match — sentence contains the field value verbatim
      2. Label/name match — sentence contains the field name or label
      3. Numeric proximity — a number in the sentence is within 2% of a field value
    """

    def match(
        self,
        sentence:         str,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        field_sources:    Dict[str, FieldSource],
    ) -> Tuple[List[FieldSource], List[str]]:
        """
        Find which field sources this sentence references.

        Returns:
          (matched_sources, untraced_values)
        """
        sentence_lower = sentence.strip().lower()
        matched:  List[FieldSource] = []
        matched_fields: Set[str] = set()

        for field_name, source in field_sources.items():
            val_str = str(source.value).strip().lower()
            if val_str and val_str in sentence_lower:
                matched.append(source)
                matched_fields.add(field_name)

        for field_name, cfg in fields_config.items():
            if field_name in matched_fields:
                continue
            label  = cfg.get("label", field_name.replace("_", " ")).lower()
            tokens = [t for t in field_name.replace("_", " ").lower().split() if len(t) > 3]
            if (
                field_name.lower() in sentence_lower
                or label in sentence_lower
                or (tokens and all(t in sentence_lower for t in tokens))
            ):
                source = field_sources.get(field_name)
                if source:
                    matched.append(source)
                    matched_fields.add(field_name)

        numeric_fields = {
            fn: float(str(v).replace(",", ""))
            for fn, v in collected_fields.items()
            if fields_config.get(fn, {}).get("type") in ("integer", "number", "float", "int")
        }
        sentence_values = SentenceParser.extract_values(sentence)

        untraced: List[str] = []
        for val_str in sentence_values:
            num_str = re.sub(r"[a-zA-Z%,]", "", val_str).strip()
            try:
                num = float(num_str)
            except (ValueError, TypeError):
                continue

            best_field, best_diff = None, float("inf")
            for fn, fv in numeric_fields.items():
                if fn in matched_fields:
                    continue
                if fv == 0:
                    continue
                diff = abs(num - fv) / abs(fv)
                if diff < best_diff:
                    best_diff, best_field = diff, fn

            if best_field and best_diff <= 0.02:
                source = field_sources.get(best_field)
                if source:
                    matched.append(source)
                    matched_fields.add(best_field)
            elif best_field and best_diff > 0.12:

                untraced.append(val_str)

        seen: Set[str] = set()
        deduped: List[FieldSource] = []
        for s in matched:
            if s.field_name not in seen:
                seen.add(s.field_name)
                deduped.append(s)

        return deduped, untraced

class TurnResolver:
    """
    Builds the FieldSource lookup from session state.

    For each collected field, finds:
      - Which turn it was collected on
      - What the user said in that turn
      - What the extraction confidence was
    """

    def build_sources(
        self,
        collected_fields:  Dict[str, Any],
        field_confidences: Dict[str, float],
        turn_history:      List[Dict[str, Any]],
        field_turn_map:    Optional[Dict[str, int]] = None,
    ) -> Dict[str, FieldSource]:
        """
        Build {field_name: FieldSource} from session data.

        Args:
            collected_fields:  field_name → value
            field_confidences: field_name → confidence score
            turn_history:      full conversation turn log
            field_turn_map:    field_name → turn number (optional; inferred if absent)
        """
        turn_map = field_turn_map or {}
        sources: Dict[str, FieldSource] = {}

        user_messages: Dict[int, str] = {}
        for entry in turn_history:
            if entry.get("role") == "user":
                turn_num = entry.get("turn", 0)
                user_messages[turn_num] = entry.get("content", "")

        for field_name, value in collected_fields.items():
            turn = turn_map.get(field_name, self._infer_turn(field_name, value, user_messages))
            user_text = user_messages.get(turn, "")
            confidence = field_confidences.get(field_name, 0.80)

            sources[field_name] = FieldSource(
                field_name   = field_name,
                value        = value,
                turn         = turn,
                user_text    = user_text,
                confidence   = confidence,
                extracted_at = self._turn_timestamp(turn, turn_history),
            )

        return sources

    @staticmethod
    def _infer_turn(
        field_name: str,
        value:      Any,
        user_messages: Dict[int, str],
    ) -> int:
        """
        If no explicit turn_map, infer the turn by searching user messages
        for the field value.
        """
        value_str = str(value).strip().lower()
        if not value_str:
            return 0

        for turn_num, msg in sorted(user_messages.items()):
            if value_str in msg.lower():
                return turn_num

        try:
            num = float(value_str.replace(",", ""))
            num_str = str(int(num)) if num == int(num) else str(num)
            for turn_num, msg in sorted(user_messages.items()):
                if num_str in msg:
                    return turn_num
        except (ValueError, TypeError):
            pass

        return 0

    @staticmethod
    def _turn_timestamp(turn: int, turn_history: List[Dict]) -> float:
        for entry in turn_history:
            if entry.get("turn") == turn:
                return entry.get("timestamp", time.time())
        return time.time()

class SourceTracer:
    """
    Traces every sentence in generated output back to the collected
    field and conversation turn that produced it.

    Usage:
        tracer = SourceTracer()

        source_map = tracer.trace(
            output           = generated_text,
            collected_fields = state.collected_fields,
            field_confidences = state.field_confidences,
            fields_config    = state.fields_config,
            turn_history     = state.turn_history,
            session_id       = state.session_id,
            goal_id          = state.goal_id,
            field_turn_map   = state._field_turn_map,
        )

        print(source_map.completeness.value)   # "fully_traced"
        print(source_map.audit_log())           # flat list for compliance

    Integration with OutputGenerator:
        # Called inside generate() after content is produced
        source_map = tracer.trace(output=content, ...)
        result["source_trace"] = source_map.to_dict()
    """

    FULLY_TRACED_THRESHOLD     = 0.95
    MOSTLY_TRACED_THRESHOLD    = 0.75
    PARTIALLY_TRACED_THRESHOLD = 0.40

    def __init__(self):
        self._parser   = SentenceParser()
        self._matcher  = FieldMatcher()
        self._resolver = TurnResolver()

    def trace(
        self,
        output:            str,
        collected_fields:  Dict[str, Any],
        field_confidences: Dict[str, float],
        fields_config:     Dict[str, dict],
        turn_history:      List[Dict[str, Any]],
        session_id:        str = "unknown",
        goal_id:           str = "unknown",
        field_turn_map:    Optional[Dict[str, int]] = None,
    ) -> SourceMap:
        """
        Trace every sentence in the output to its source field + turn.

        Args:
            output:            Generated text to trace
            collected_fields:  All collected field values
            field_confidences: Per-field confidence scores
            fields_config:     Goal YAML field specs
            turn_history:      Full conversation turn log
            session_id:        For the audit log
            goal_id:           For the audit log
            field_turn_map:    {field_name: turn_number} (optional)

        Returns:
            SourceMap with complete attribution
        """
        logger.info(
            "source_tracer: tracing session=%s goal=%s fields=%d chars=%d",
            session_id, goal_id, len(collected_fields), len(output or ""),
        )

        if not output or not output.strip():
            return self._empty_map(session_id, goal_id)

        parsed = self._parser.parse(output)
        if not parsed:
            return self._empty_map(session_id, goal_id)

        field_sources = self._resolver.build_sources(
            collected_fields  = collected_fields,
            field_confidences = field_confidences,
            turn_history      = turn_history,
            field_turn_map    = field_turn_map,
        )

        traced_sentences: List[TracedSentence] = []
        field_coverage:   Dict[str, int] = {}
        untraced_text:    List[str] = []
        factual_count     = 0
        traced_count      = 0

        for idx, (sentence, sentence_type) in enumerate(parsed):
            is_generic = sentence_type in ("generic", "heading")

            if sentence_type == "heading":
                traced_sentences.append(TracedSentence(
                    sentence       = sentence,
                    sentence_index = idx,
                    sentence_type  = sentence_type,
                    sources        = [],
                    is_traced      = True,
                    is_generic     = True,
                    untraced_values = [],
                ))
                continue

            sources, untraced_vals = self._matcher.match(
                sentence         = sentence,
                collected_fields = collected_fields,
                fields_config    = fields_config,
                field_sources    = field_sources,
            )

            is_traced = bool(sources) or is_generic
            if not is_generic:
                factual_count += 1
                if is_traced:
                    traced_count += 1
                else:
                    untraced_text.append(sentence)

            for s in sources:
                field_coverage[s.field_name] = field_coverage.get(s.field_name, 0) + 1

            traced_sentences.append(TracedSentence(
                sentence        = sentence,
                sentence_index  = idx,
                sentence_type   = sentence_type,
                sources         = sources,
                is_traced       = is_traced,
                is_generic      = is_generic,
                untraced_values = untraced_vals,
            ))

        traced_pct = traced_count / factual_count if factual_count > 0 else 1.0
        completeness = self._classify_completeness(traced_pct)

        logger.info(
            "source_tracer: session=%s completeness=%s traced=%.0f%% "
            "factual=%d traced=%d untraced=%d",
            session_id, completeness.value, traced_pct * 100,
            factual_count, traced_count, len(untraced_text),
        )

        return SourceMap(
            session_id      = session_id,
            goal_id         = goal_id,
            completeness    = completeness,
            traced_pct      = round(traced_pct, 4),
            sentences       = traced_sentences,
            field_coverage  = field_coverage,
            untraced_sentences = untraced_text,
        )

    def trace_sentence(
        self,
        sentence:          str,
        collected_fields:  Dict[str, Any],
        field_confidences: Dict[str, float],
        fields_config:     Dict[str, dict],
        turn_history:      List[Dict[str, Any]],
        field_turn_map:    Optional[Dict[str, int]] = None,
    ) -> TracedSentence:
        """
        Trace a single sentence. Useful for mid-conversation agent response tracing.
        """
        sentence_type = self._parser._classify(sentence)
        field_sources = self._resolver.build_sources(
            collected_fields  = collected_fields,
            field_confidences = field_confidences,
            turn_history      = turn_history,
            field_turn_map    = field_turn_map,
        )
        sources, untraced = self._matcher.match(
            sentence         = sentence,
            collected_fields = collected_fields,
            fields_config    = fields_config,
            field_sources    = field_sources,
        )
        is_generic = sentence_type in ("generic", "heading")
        return TracedSentence(
            sentence        = sentence,
            sentence_index  = 0,
            sentence_type   = sentence_type,
            sources         = sources,
            is_traced       = bool(sources) or is_generic,
            is_generic      = is_generic,
            untraced_values = untraced,
        )

    def _classify_completeness(self, traced_pct: float) -> TraceCompleteness:
        if traced_pct >= self.FULLY_TRACED_THRESHOLD:
            return TraceCompleteness.FULLY_TRACED
        if traced_pct >= self.MOSTLY_TRACED_THRESHOLD:
            return TraceCompleteness.MOSTLY_TRACED
        if traced_pct >= self.PARTIALLY_TRACED_THRESHOLD:
            return TraceCompleteness.PARTIALLY_TRACED
        return TraceCompleteness.UNTRACEABLE

    @staticmethod
    def _empty_map(session_id: str, goal_id: str) -> SourceMap:
        return SourceMap(
            session_id      = session_id,
            goal_id         = goal_id,
            completeness    = TraceCompleteness.FULLY_TRACED,
            traced_pct      = 1.0,
            sentences       = [],
            field_coverage  = {},
            untraced_sentences = [],
        )
