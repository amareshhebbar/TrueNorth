"""
truenorth/intelligence/confidence_scorer.py

Scores the confidence (0.0–1.0) of each extracted field value.

Confidence is computed from:
  - Extraction confidence reported by the LLM (primary signal)
  - Field type validation (is the value the right type/format?)
  - User confirmation signal (did user explicitly confirm this value?)
  - Conflict presence (is this field involved in a known conflict?)
  - Source quality (full sentence vs. one word)

Used by:
  - Reasoner — low confidence triggers a clarification ask
  - OutputGenerator — flags low-confidence fields in the final report
  - Analytics — tracks per-field average confidence across sessions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceScore:
    field:       str
    value:       Any
    score:       float    # 0.0–1.0
    factors:     dict     # contributing factor → score adjustment
    needs_confirm: bool   # True when score < confirm_threshold

    def to_dict(self) -> dict:
        return {
            "field":         self.field,
            "value":         self.value,
            "score":         round(self.score, 3),
            "factors":       {k: round(v, 3) for k, v in self.factors.items()},
            "needs_confirm": self.needs_confirm,
        }


# ---------------------------------------------------------------------------
# Type validators
# ---------------------------------------------------------------------------

_EMAIL_RE    = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")
_PHONE_RE    = re.compile(r"^[\d\s\+\-\(\)]{7,15}$")
_DATE_RE     = re.compile(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2}")
_NUMBER_RE   = re.compile(r"^-?\d+\.?\d*$")
_BOOLEAN_RE  = re.compile(r"^(yes|no|true|false|y|n|1|0)$", re.IGNORECASE)


def _validate_type(value: Any, expected_type: str) -> float:
    """Return a type-validation score (0.5 = cannot validate, 1.0 = confirmed, 0.2 = wrong type)."""
    if value is None:
        return 0.0

    v = str(value).strip()

    validators = {
        "email":   (_EMAIL_RE.match,   0.95, 0.15),
        "phone":   (_PHONE_RE.match,   0.90, 0.20),
        "date":    (_DATE_RE.search,   0.85, 0.25),
        "number":  (_NUMBER_RE.match,  0.90, 0.10),
        "integer": (_NUMBER_RE.match,  0.90, 0.10),
        "boolean": (_BOOLEAN_RE.match, 0.90, 0.15),
    }

    if expected_type in validators:
        fn, ok_score, fail_score = validators[expected_type]
        return ok_score if fn(v) else fail_score

    return 0.65  # text / unknown — no validation possible


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """
    Computes a confidence score for each extracted field value.

    Usage:
        scorer = ConfidenceScorer()
        score = scorer.score(
            field="age",
            value=28,
            field_config={"type": "integer", "required": True},
            extraction_confidence=0.85,
            source_text="I am 28 years old",
        )
    """

    CONFIRM_THRESHOLD: float = 0.55

    def score(
        self,
        field:                  str,
        value:                  Any,
        field_config:           Optional[dict] = None,
        extraction_confidence:  float = 1.0,    # confidence reported by field extractor LLM
        source_text:            str   = "",      # original user text the value was extracted from
        user_confirmed:         bool  = False,   # True if user explicitly said "yes that's right"
        in_conflict:            bool  = False,   # True if this field has an active conflict
    ) -> ConfidenceScore:
        """
        Compute a composite confidence score.

        Args:
            field:                 Field name
            value:                 Extracted value
            field_config:          Field specification dict from YAML
            extraction_confidence: LLM-reported confidence (0-1)
            source_text:           The user message this was extracted from
            user_confirmed:        Whether user explicitly confirmed
            in_conflict:           Whether a contradiction was detected for this field
        """
        cfg       = field_config or {}
        ftype     = cfg.get("type", "text")
        factors:  Dict[str, float] = {}

        # --- Factor 1: LLM extraction confidence (40% weight)
        factors["extraction"] = extraction_confidence * 0.40

        # --- Factor 2: Type validation (20% weight)
        type_score = _validate_type(value, ftype)
        factors["type_validation"] = type_score * 0.20

        # --- Factor 3: Source text quality (20% weight)
        source_score = self._source_quality(value, source_text)
        factors["source_quality"] = source_score * 0.20

        # --- Factor 4: User confirmation bonus (15% weight)
        confirm_bonus = 0.90 if user_confirmed else 0.40
        factors["confirmation"] = confirm_bonus * 0.15

        # --- Factor 5: Conflict penalty (5% weight)
        conflict_penalty = 0.10 if in_conflict else 0.90
        factors["conflict"] = conflict_penalty * 0.05

        raw_score = sum(factors.values())
        final     = round(min(max(raw_score, 0.0), 1.0), 4)

        return ConfidenceScore(
            field         = field,
            value         = value,
            score         = final,
            factors       = factors,
            needs_confirm = final < self.CONFIRM_THRESHOLD,
        )

    def score_all(
        self,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        extraction_meta:  Optional[Dict[str, dict]] = None,
    ) -> Dict[str, ConfidenceScore]:
        """
        Score all collected fields at once.

        Args:
            collected_fields: {field_name: value}
            fields_config:    {field_name: field_spec}
            extraction_meta:  {field_name: {confidence, source_text, confirmed, in_conflict}}
        """
        meta    = extraction_meta or {}
        results = {}
        for field_name, value in collected_fields.items():
            field_meta = meta.get(field_name, {})
            results[field_name] = self.score(
                field                 = field_name,
                value                 = value,
                field_config          = fields_config.get(field_name, {}),
                extraction_confidence = field_meta.get("confidence", 0.80),
                source_text           = field_meta.get("source_text", ""),
                user_confirmed        = field_meta.get("confirmed", False),
                in_conflict           = field_meta.get("in_conflict", False),
            )
        return results

    def overall_session_confidence(self, scores: Dict[str, ConfidenceScore]) -> float:
        """Return average confidence across all scored fields."""
        if not scores:
            return 0.0
        return round(sum(s.score for s in scores.values()) / len(scores), 3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _source_quality(value: Any, source_text: str) -> float:
        """
        How much of the source text supports this value?
        More context around the value = higher confidence.
        """
        if not source_text:
            return 0.50   # no source text available

        value_str  = str(value).strip().lower()
        source_str = source_text.strip().lower()

        if not value_str:
            return 0.20

        if value_str in source_str:
            word_count = len(source_str.split())
            if word_count >= 6:
                return 0.90
            elif word_count >= 3:
                return 0.75
            else:
                return 0.60

        return 0.50