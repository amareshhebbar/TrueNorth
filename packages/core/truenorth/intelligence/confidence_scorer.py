"""

Per-field confidence scoring. Every extracted value gets a 0.0–1.0 score
composed from 8 independent factors. The Reasoner uses this score to decide
whether to accept a value, ask for confirmation, or flag it.

Factors (8 total):
  1. extraction_confidence  — LLM's own certainty at extraction time      (25%)
  2. type_validation        — does the value pass regex / range checks     (15%)
  3. source_quality         — how much context surrounded the extraction   (15%)
  4. user_confirmation      — did the user explicitly confirm the value    (15%)
  5. conflict_history       — has this field ever been in a conflict       (10%)
  6. consistency            — does the value agree with related fields     (10%)
  7. extraction_method      — direct quote > LLM inference > rule-based   ( 5%)
  8. temporal_stability     — consistent across multiple extractions       ( 5%)

Compared to v1 (5 factors, fixed weights):
  + conflict_history now tracks past conflicts, not just current
  + consistency cross-checks related fields (age vs work experience)
  + extraction_method distinguishes quote / inference / rule-based
  + temporal_stability rewards fields confirmed consistently over turns
  + range validation added for numeric fields with min/max
  + allowed_values validation added for categorical fields
  + score_all now accepts full session history for temporal scoring
  + confidence_band() classifies score into HIGH/MEDIUM/LOW/UNCONFIDENT
  + session_health() returns a ConfidenceReport for the whole session
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Enums and result types
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionMethod(str, Enum):
    DIRECT_QUOTE = "direct_quote"   # user literally said the value verbatim
    LLM_EXTRACT  = "llm_extract"    # LLM inferred the value from natural language
    RULE_BASED   = "rule_based"     # regex / heuristic extraction
    USER_CONFIRM = "user_confirm"   # user said "yes" to a clarification question
    MANUAL       = "manual"         # set programmatically (e.g. from prior session)


class ConfidenceBand(str, Enum):
    HIGH        = "high"        # ≥ 0.80  — accept, no clarification needed
    MEDIUM      = "medium"      # 0.60–0.79 — accept, soft note in output
    LOW         = "low"         # 0.40–0.59 — ask for confirmation next turn
    UNCONFIDENT = "unconfident" # < 0.40  — re-ask immediately


@dataclass
class ConfidenceScore:
    """Per-field confidence result."""
    field:         str
    value:         Any
    score:         float           # 0.0–1.0 composite
    factors:       Dict[str, float]# factor_name → contribution (summing to score)
    band:          ConfidenceBand
    needs_confirm: bool            # True when score < CONFIRM_THRESHOLD
    issues:        List[str]       # human-readable warnings (for dry-run / dashboard)

    def to_dict(self) -> dict:
        return {
            "field":         self.field,
            "value":         self.value,
            "score":         round(self.score, 3),
            "band":          self.band.value,
            "factors":       {k: round(v, 4) for k, v in self.factors.items()},
            "needs_confirm": self.needs_confirm,
            "issues":        self.issues,
        }


@dataclass
class ConfidenceReport:
    """Session-level confidence health summary."""
    session_id:         str
    overall_score:      float
    overall_band:       ConfidenceBand
    field_scores:       Dict[str, ConfidenceScore]
    needs_confirm:      List[str]   # field names needing confirmation
    high_confidence:    List[str]
    low_confidence:     List[str]
    unconfident:        List[str]
    ready_for_output:   bool        # True if all required fields are HIGH/MEDIUM

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "overall_score":    round(self.overall_score, 3),
            "overall_band":     self.overall_band.value,
            "needs_confirm":    self.needs_confirm,
            "high_confidence":  self.high_confidence,
            "low_confidence":   self.low_confidence,
            "unconfident":      self.unconfident,
            "ready_for_output": self.ready_for_output,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Type validators (expanded from v1)
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE   = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")
_PHONE_IN   = re.compile(r"^[\+]?[6-9]\d{9}$")           # Indian mobile
_PHONE_INTL = re.compile(r"^[\+]?[\d\s\-\(\)]{7,15}$")
_DATE_RE    = re.compile(
    r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-]\d{2}[\/\-]\d{2}"
)
_NUMBER_RE  = re.compile(r"^-?\d+(?:\.\d+)?$")
_INT_RE     = re.compile(r"^-?\d+$")
_BOOL_RE    = re.compile(r"^(yes|no|true|false|y|n|1|0)$", re.IGNORECASE)
_URL_RE     = re.compile(r"^https?://\S+$")


def _validate_type(value: Any, field_cfg: dict) -> Tuple[float, List[str]]:
    """
    Validate a value against its declared field type and constraints.
    Returns (score 0-1, list of issue strings).
    """
    issues: List[str] = []
    if value is None:
        return 0.0, ["Value is None"]

    ftype = field_cfg.get("type", "text")
    v     = str(value).strip()

    # ── Type-specific validation ───────────────────────────────────────────
    type_scores = {
        "email":   (_EMAIL_RE.match(v),                       0.95, 0.15),
        "phone":   (_PHONE_IN.match(v) or _PHONE_INTL.match(v), 0.90, 0.20),
        "date":    (_DATE_RE.search(v),                       0.85, 0.25),
        "boolean": (_BOOL_RE.match(v),                        0.90, 0.15),
        "url":     (_URL_RE.match(v),                         0.90, 0.20),
        "number":  (_NUMBER_RE.match(v),                      0.88, 0.10),
        "float":   (_NUMBER_RE.match(v),                      0.88, 0.10),
        "integer": (_INT_RE.match(v),                         0.90, 0.10),
        "int":     (_INT_RE.match(v),                         0.90, 0.10),
    }

    if ftype in type_scores:
        match, ok_score, fail_score = type_scores[ftype]
        type_score = ok_score if match else fail_score
        if not match:
            issues.append(f"Value {v!r} fails {ftype} format validation")
    else:
        type_score = 0.65  # text / unknown — no regex to apply

    # ── Range validation (integer / number) ───────────────────────────────
    if ftype in ("integer", "int", "number", "float"):
        try:
            num = float(v.replace(",", ""))
            mn  = field_cfg.get("min")
            mx  = field_cfg.get("max")
            if mn is not None and num < mn:
                type_score = max(type_score - 0.30, 0.05)
                issues.append(f"Value {num} below minimum {mn}")
            elif mx is not None and num > mx:
                type_score = max(type_score - 0.30, 0.05)
                issues.append(f"Value {num} above maximum {mx}")
        except (ValueError, TypeError):
            pass

    # ── Allowed values validation (categorical) ───────────────────────────
    allowed = field_cfg.get("allowed_values") or field_cfg.get("enum", [])
    if allowed:
        normalized = [str(a).strip().lower() for a in allowed]
        if v.lower() not in normalized:
            type_score = max(type_score - 0.25, 0.10)
            issues.append(
                f"Value {v!r} not in allowed values: {allowed[:5]}"
                + ("…" if len(allowed) > 5 else "")
            )

    return round(type_score, 4), issues


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-field consistency checks
# ─────────────────────────────────────────────────────────────────────────────

_CONSISTENCY_CHECKS = [
    ("age", "work_experience_years",
     lambda a, b: float(a) > float(b) + 10,
     "age should exceed work experience by at least 10 years"),
    ("weight_kg", "height_cm",
     lambda w, h: 10 < float(w) / ((float(h) / 100) ** 2) < 70,
     "BMI (weight/height²) should be between 10 and 70"),
    # Age boundaries
    ("age", "age",
     lambda a, _: 5 < float(a) < 120,
     "age should be between 5 and 120"),
]


def _check_consistency(
    field:            str,
    value:            Any,
    collected_fields: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """
    Cross-check a field value against related collected fields.
    Returns (score_multiplier 0.5-1.0, issues).
    """
    issues: List[str] = []
    score = 1.0

    for fa, fb, check_fn, description in _CONSISTENCY_CHECKS:
        if field not in (fa, fb):
            continue

        other_field = fb if field == fa else fa
        other_val   = collected_fields.get(other_field)
        if other_val is None:
            continue

        try:
            val_a = value     if field == fa else other_val
            val_b = other_val if field == fa else value
            if not check_fn(val_a, val_b):
                score   = min(score, 0.55)
                issues.append(f"Consistency check failed: {description}")
        except (TypeError, ValueError, ZeroDivisionError):
            pass   # can't compute — skip

    return round(score, 4), issues


# ─────────────────────────────────────────────────────────────────────────────
#  Extraction method scoring
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_SCORES: Dict[str, float] = {
    ExtractionMethod.DIRECT_QUOTE: 0.98,   # "I am 28" → verbatim
    ExtractionMethod.USER_CONFIRM: 0.97,   # user said "yes, that's right"
    ExtractionMethod.LLM_EXTRACT:  0.80,   # LLM inferred from natural language
    ExtractionMethod.RULE_BASED:   0.65,   # regex / heuristic
    ExtractionMethod.MANUAL:       0.85,   # programmatic set
}


# ─────────────────────────────────────────────────────────────────────────────
#  ConfidenceScorer (hardened)
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Hardened per-field confidence scorer with 8 factors.

    Thresholds:
      HIGH        ≥ 0.80  — accept without clarification
      MEDIUM      0.60–0.79 — accept, note in output
      LOW         0.40–0.59 — ask for confirmation next turn
      UNCONFIDENT < 0.40  — re-ask immediately

    Usage (single field):
        scorer = ConfidenceScorer()
        cs = scorer.score(
            field                = "age",
            value                = 28,
            field_config         = {"type": "integer", "min": 1, "max": 120},
            extraction_confidence = 0.90,
            source_text          = "I am 28 years old",
            method               = ExtractionMethod.LLM_EXTRACT,
        )
        print(cs.band)   # HIGH

    Usage (all fields):
        report = scorer.session_health(
            session_id       = "sess-123",
            collected_fields = state.collected_fields,
            fields_config    = state.fields_config,
            required_fields  = list(state.required_fields.keys()),
            extraction_meta  = {...},
            history          = state.turn_history,
        )
    """

    # Score thresholds for confidence bands
    BAND_HIGH        = 0.80
    BAND_MEDIUM      = 0.60
    BAND_LOW         = 0.40
    CONFIRM_THRESHOLD = 0.60   
    
    WEIGHTS = {
        "extraction":      0.25,
        "type_validation": 0.15,
        "source_quality":  0.15,
        "confirmation":    0.15,
        "conflict":        0.10,
        "consistency":     0.10,
        "method":          0.05,
        "stability":       0.05,
    }

    def score(
        self,
        field:                 str,
        value:                 Any,
        field_config:          Optional[dict]            = None,
        extraction_confidence: float                     = 0.80,
        source_text:           str                       = "",
        user_confirmed:        bool                      = False,
        in_conflict:           bool                      = False,
        conflict_history:      int                       = 0,
        collected_fields:      Optional[Dict[str, Any]]  = None,
        method:                ExtractionMethod          = ExtractionMethod.LLM_EXTRACT,
        prior_extractions:     Optional[List[Any]]       = None,
    ) -> ConfidenceScore:
        """
        Compute composite confidence for one field value.

        Args:
            field:                 Field name
            value:                 Extracted value
            field_config:          Field spec from goal YAML
            extraction_confidence: LLM's self-reported confidence (0-1)
            source_text:           User message this was extracted from
            user_confirmed:        User explicitly confirmed ("yes, that's right")
            in_conflict:           Field currently has an unresolved conflict
            conflict_history:      How many times this field has been in conflict
            collected_fields:      All currently collected fields (for consistency)
            method:                How the value was extracted
            prior_extractions:     Previous extracted values for this field (for stability)
        """
        cfg    = field_config or {}
        issues: List[str] = []
        factors: Dict[str, float] = {}

        # ── 1. Extraction confidence (25%) ─────────────────────────────────
        factors["extraction"] = _clamp(extraction_confidence) * self.WEIGHTS["extraction"]

        # ── 2. Type validation (15%) ────────────────────────────────────────
        type_score, type_issues = _validate_type(value, cfg)
        issues.extend(type_issues)
        factors["type_validation"] = type_score * self.WEIGHTS["type_validation"]

        # ── 3. Source quality (15%) ─────────────────────────────────────────
        src_score = self._source_quality(value, source_text)
        factors["source_quality"] = src_score * self.WEIGHTS["source_quality"]

        # ── 4. User confirmation (15%) ──────────────────────────────────────
        if user_confirmed:
            confirm_score = 1.0
        elif method == ExtractionMethod.USER_CONFIRM:
            confirm_score = 0.95
        else:
            confirm_score = 0.40    # not confirmed — medium penalty
        factors["confirmation"] = confirm_score * self.WEIGHTS["confirmation"]

        # ── 5. Conflict history (10%) ───────────────────────────────────────
        if in_conflict:
            conflict_score = 0.05   # severe: active conflict
        elif conflict_history == 1:
            conflict_score = 0.55   # mild: one past conflict, now resolved
        elif conflict_history >= 2:
            conflict_score = 0.30   # bad: repeatedly conflicted field
            issues.append(f"Field has been in conflict {conflict_history} times")
        else:
            conflict_score = 1.0    # no conflict history
        factors["conflict"] = conflict_score * self.WEIGHTS["conflict"]

        # ── 6. Cross-field consistency (10%) ───────────────────────────────
        if collected_fields:
            consistency_score, cs_issues = _check_consistency(
                field, value, collected_fields
            )
            issues.extend(cs_issues)
        else:
            consistency_score = 0.80   
        factors["consistency"] = consistency_score * self.WEIGHTS["consistency"]

        # ── 7. Extraction method (5%) ───────────────────────────────────────
        method_score = _METHOD_SCORES.get(method, 0.75)
        if method == ExtractionMethod.DIRECT_QUOTE and source_text:
            if str(value).lower() not in source_text.lower():
                method_score = 0.70   
                issues.append("Direct quote claimed but value not found verbatim in source")
        factors["method"] = method_score * self.WEIGHTS["method"]

        # ── 8. Temporal stability (5%) ──────────────────────────────────────
        stability_score = self._stability(value, prior_extractions or [])
        factors["stability"] = stability_score * self.WEIGHTS["stability"]

        # ── Composite ───────────────────────────────────────────────────────
        raw    = sum(factors.values())
        final  = round(_clamp(raw), 4)
        band   = self._band(final)

        if final < self.CONFIRM_THRESHOLD:
            issues.append(
                f"Confidence {final:.2f} below threshold {self.CONFIRM_THRESHOLD}"
            )

        return ConfidenceScore(
            field         = field,
            value         = value,
            score         = final,
            factors       = factors,
            band          = band,
            needs_confirm = final < self.CONFIRM_THRESHOLD,
            issues        = issues,
        )

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_all(
        self,
        collected_fields:  Dict[str, Any],
        fields_config:     Dict[str, dict],
        extraction_meta:   Optional[Dict[str, dict]] = None,
        conflict_history:  Optional[Dict[str, int]]  = None,
    ) -> Dict[str, ConfidenceScore]:
        """
        Score all collected fields in one call.

        extraction_meta keys per field:
          confidence   : float (LLM confidence)
          source_text  : str
          confirmed    : bool
          in_conflict  : bool
          method       : ExtractionMethod value string
          prior_values : list of prior extracted values
        """
        meta      = extraction_meta or {}
        conflicts = conflict_history or {}
        results   = {}

        for field_name, value in collected_fields.items():
            fm = meta.get(field_name, {})
            method_str = fm.get("method", ExtractionMethod.LLM_EXTRACT)
            try:
                method = ExtractionMethod(method_str)
            except ValueError:
                method = ExtractionMethod.LLM_EXTRACT

            results[field_name] = self.score(
                field                 = field_name,
                value                 = value,
                field_config          = fields_config.get(field_name, {}),
                extraction_confidence = fm.get("confidence", 0.80),
                source_text           = fm.get("source_text", ""),
                user_confirmed        = fm.get("confirmed", False),
                in_conflict           = fm.get("in_conflict", False),
                conflict_history      = conflicts.get(field_name, 0),
                collected_fields      = collected_fields,
                method                = method,
                prior_extractions     = fm.get("prior_values", []),
            )
        return results

    def session_health(
        self,
        session_id:       str,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        required_fields:  List[str],
        extraction_meta:  Optional[Dict[str, dict]] = None,
        conflict_history: Optional[Dict[str, int]]  = None,
    ) -> ConfidenceReport:
        """
        Return a full confidence health report for the current session.
        Used by the Studio dashboard and dry-run reporter.
        """
        scores = self.score_all(
            collected_fields  = collected_fields,
            fields_config     = fields_config,
            extraction_meta   = extraction_meta,
            conflict_history  = conflict_history,
        )

        by_band: Dict[str, List[str]] = {
            "high": [], "medium": [], "low": [], "unconfident": []
        }
        needs_confirm: List[str] = []

        for fn, cs in scores.items():
            by_band[cs.band.value].append(fn)
            if cs.needs_confirm:
                needs_confirm.append(fn)

        overall = self.overall_score(scores)
        overall_band = self._band(overall)

        required_scores = [
            scores[f].band for f in required_fields if f in scores
        ]
        ready = all(
            b in (ConfidenceBand.HIGH, ConfidenceBand.MEDIUM)
            for b in required_scores
        ) and len(required_scores) == len(required_fields)

        return ConfidenceReport(
            session_id       = session_id,
            overall_score    = overall,
            overall_band     = overall_band,
            field_scores     = scores,
            needs_confirm    = needs_confirm,
            high_confidence  = by_band["high"],
            low_confidence   = by_band["low"],
            unconfident      = by_band["unconfident"],
            ready_for_output = ready,
        )

    # ------------------------------------------------------------------
    # Convenience aliases (backward compatibility with v1 tests)
    # ------------------------------------------------------------------

    def overall_session_confidence(
        self, scores: Dict[str, ConfidenceScore]
    ) -> float:
        return self.overall_score(scores)

    def overall_score(self, scores: Dict[str, ConfidenceScore]) -> float:
        if not scores:
            return 0.0
        return round(sum(s.score for s in scores.values()) / len(scores), 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _band(self, score: float) -> ConfidenceBand:
        if score >= self.BAND_HIGH:
            return ConfidenceBand.HIGH
        if score >= self.BAND_MEDIUM:
            return ConfidenceBand.MEDIUM
        if score >= self.BAND_LOW:
            return ConfidenceBand.LOW
        return ConfidenceBand.UNCONFIDENT

    @staticmethod
    def _source_quality(value: Any, source_text: str) -> float:
        """How much supporting context surrounds the value in the source text."""
        if not source_text:
            return 0.45

        value_str  = str(value).strip().lower()
        source_str = source_text.strip().lower()

        if not value_str:
            return 0.20

        if value_str in source_str:
            word_count = len(source_str.split())
            if word_count >= 8:
                return 0.95     # rich context
            elif word_count >= 4:
                return 0.80
            elif word_count >= 2:
                return 0.65
            else:
                return 0.50 
        try:
            num = float(str(value).replace(",", ""))
            if str(int(num)) in source_str or f"{num:.1f}" in source_str:
                return 0.70
        except (ValueError, TypeError):
            pass

        words = set(value_str.split())
        in_src = sum(1 for w in words if w in source_str and len(w) > 2)
        if words and in_src / len(words) >= 0.5:
            return 0.60

        return 0.40   

    @staticmethod
    def _stability(value: Any, prior_extractions: List[Any]) -> float:
        """
        How stable is this value across multiple extraction attempts?
        Consistent values across turns = higher confidence.
        """
        if not prior_extractions:
            return 0.75   # first extraction — neutral

        value_str = str(value).strip().lower()
        matches   = sum(
            1 for v in prior_extractions
            if str(v).strip().lower() == value_str
        )
        total = len(prior_extractions)
        consistency = matches / total

        if consistency >= 0.80:
            return 1.0   
        elif consistency >= 0.50:
            return 0.75
        elif consistency > 0:
            return 0.55   
        else:
            return 0.20   


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))