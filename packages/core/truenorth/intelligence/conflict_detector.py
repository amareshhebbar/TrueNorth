"""

Detects, classifies, scores, and resolves contradictions across conversation turns.

v2 improvements over v1:
  + 7 conflict types (NUMERIC_MISMATCH, BOOLEAN_FLIP, CATEGORICAL_FLIP,
                      TEXT_CONTRADICTION, UNIT_MISMATCH, CROSS_FIELD,
                      RANGE_VIOLATION, SEMANTIC_CONTRADICTION)
  + Severity scoring: CRITICAL / HIGH / MEDIUM / LOW per conflict
  + Confidence-weighted suppression — low-confidence new extraction does not
    override a high-confidence collected value
  + Semantic alias awareness — "gym" == "gym membership" → not a conflict
  + Unit normalisation — "70 kg" vs "154 lbs" → same value, no conflict
  + Cross-field consistency rules — age=22 + work_experience=25 → conflict
  + Auto-resolution — range violations and low-severity conflicts auto-resolved
  + ConflictStore — manages full lifecycle per session
  + Natural per-type clarification questions
  + resolve_from_user_input() — parse user message to win a conflict
  + ConflictReport — session-level statistics for Studio dashboard
  + v1 check() / resolve() API preserved for backward compatibility
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class ConflictType(str, Enum):
    NUMERIC_MISMATCH       = "numeric_mismatch"
    BOOLEAN_FLIP           = "boolean_flip"
    CATEGORICAL_FLIP       = "categorical_flip"
    TEXT_CONTRADICTION     = "text_contradiction"
    UNIT_MISMATCH          = "unit_mismatch"
    CROSS_FIELD            = "cross_field"
    RANGE_VIOLATION        = "range_violation"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"


class ConflictSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class ConflictStatus(str, Enum):
    OPEN          = "open"
    ESCALATED     = "escalated"
    RESOLVED      = "resolved"
    AUTO_RESOLVED = "auto_resolved"
    DISMISSED     = "dismissed"


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConflictEvidence:
    old_source_text:  str   = ""
    new_source_text:  str   = ""
    old_confidence:   float = 1.0
    new_confidence:   float = 1.0
    detection_method: str   = "rule"


@dataclass
class Conflict:
    field:           str
    old_value:       Any
    new_value:       Any
    conflict_type:   ConflictType
    turn_old:        int
    turn_new:        int
    id:              str             = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])
    severity:        ConflictSeverity = ConflictSeverity.MEDIUM
    status:          ConflictStatus  = ConflictStatus.OPEN
    resolution:      Optional[Any]   = None
    resolution_turn: Optional[int]   = None
    evidence:        ConflictEvidence = field(default_factory=ConflictEvidence)
    created_at:      float = field(default_factory=time.time)
    updated_at:      float = field(default_factory=time.time)
    related_field:   Optional[str]   = None
    # v1 compat attributes
    resolved:        bool = False

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "field":           self.field,
            "old_value":       self.old_value,
            "new_value":       self.new_value,
            "conflict_type":   self.conflict_type.value,
            "severity":        self.severity.value,
            "turn_old":        self.turn_old,
            "turn_new":        self.turn_new,
            "status":          self.status.value,
            "resolution":      self.resolution,
            "resolution_turn": self.resolution_turn,
            "related_field":   self.related_field,
            "resolved":        self.resolved,
            "evidence": {
                "old_confidence": self.evidence.old_confidence,
                "new_confidence": self.evidence.new_confidence,
                "old_source":     self.evidence.old_source_text[:100],
                "new_source":     self.evidence.new_source_text[:100],
            },
        }

    def clarification_question(self, field_config: Optional[dict] = None) -> str:
        cfg   = field_config or {}
        label = cfg.get("label", self.field.replace("_", " ").title())
        old, new = self.old_value, self.new_value
        templates = {
            ConflictType.BOOLEAN_FLIP: (
                f"Just to clarify — earlier you said {label} was {old!r}, "
                f"but now it sounds like {new!r}. Which is correct?"
            ),
            ConflictType.NUMERIC_MISMATCH: (
                f"I want to make sure I have the right {label}. "
                f"You mentioned {old!r} before, but now it sounds like {new!r}. "
                f"Which figure should I use?"
            ),
            ConflictType.CATEGORICAL_FLIP: (
                f"I noticed a change — your {label} was {old!r} earlier, "
                f"but now you're saying {new!r}. Has your answer changed?"
            ),
            ConflictType.UNIT_MISMATCH: (
                f"I want to confirm your {label}. Could you give me the value "
                f"in one unit so I get it right?"
            ),
            ConflictType.RANGE_VIOLATION: (
                f"The {label} you mentioned ({new!r}) seems outside the expected range. "
                f"Could you double-check that figure?"
            ),
            ConflictType.CROSS_FIELD: (
                f"Something doesn't quite add up — could you help me reconcile that?"
            ),
            ConflictType.SEMANTIC_CONTRADICTION: (
                f"I noticed something that might be a contradiction about {label}. "
                f"Earlier: {old!r}. Now: {new!r}. Could you clarify?"
            ),
            ConflictType.TEXT_CONTRADICTION: (
                f"I'm getting two different pictures of your {label}: "
                f"{old!r} vs {new!r}. Which one is accurate?"
            ),
        }
        return templates.get(
            self.conflict_type,
            f"I have two different answers for {label}: {old!r} and {new!r}. Which is correct?"
        )

    @property
    def is_open(self) -> bool:
        return self.status in (ConflictStatus.OPEN, ConflictStatus.ESCALATED)

    @property
    def is_resolved(self) -> bool:
        return self.status in (
            ConflictStatus.RESOLVED, ConflictStatus.AUTO_RESOLVED, ConflictStatus.DISMISSED
        )


@dataclass
class ConflictReport:
    session_id:             str
    total_detected:         int
    open_count:             int
    resolved_count:         int
    auto_resolved:          int
    dismissed:              int
    severity_counts:        Dict[str, int]
    most_conflicted_fields: List[str]
    conflict_rate:          float
    all_conflicts:          List[Conflict]

    def to_dict(self) -> dict:
        return {
            "session_id":             self.session_id,
            "total_detected":         self.total_detected,
            "open_count":             self.open_count,
            "resolved_count":         self.resolved_count,
            "auto_resolved":          self.auto_resolved,
            "dismissed":              self.dismissed,
            "severity_counts":        self.severity_counts,
            "most_conflicted_fields": self.most_conflicted_fields,
            "conflict_rate":          round(self.conflict_rate, 3),
            "conflicts":              [c.to_dict() for c in self.all_conflicts],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic aliases
# ─────────────────────────────────────────────────────────────────────────────

_SEMANTIC_ALIASES: List[frozenset] = [
    frozenset(["gym", "gym membership", "at the gym", "fitness center", "health club",
               "gym and dumbbells", "i go to the gym", "the gym"]),
    frozenset(["home", "at home", "home gym", "home workout",
               "working out at home", "dumbbells at home", "home equipment"]),
    frozenset(["no restrictions", "no dietary restrictions", "none",
               "nothing special", "no preference", "no specific diet"]),
    frozenset(["male", "man", "he", "m"]),
    frozenset(["female", "woman", "she", "f"]),
    frozenset(["sedentary", "not very active", "mostly sitting", "desk job"]),
    frozenset(["lightly active", "slightly active", "a little active", "some walking"]),
    frozenset(["moderately active", "moderate", "somewhat active"]),
    frozenset(["very active", "highly active", "athlete", "train a lot"]),
    frozenset(["lose weight", "weight loss", "lose fat", "cut weight", "get lean",
               "slim down", "reduce weight"]),
    frozenset(["build muscle", "gain muscle", "muscle gain", "bulk up",
               "get bigger", "hypertrophy"]),
    frozenset(["general fitness", "stay fit", "general health", "overall fitness"]),
    frozenset(["yes", "y", "true", "1", "yeah", "yep", "correct", "right"]),
    frozenset(["no", "n", "false", "0", "nope", "nah", "not really"]),
]


def _aliases_match(a: str, b: str) -> bool:
    a_l, b_l = a.strip().lower(), b.strip().lower()
    if a_l == b_l:
        return True
    for alias_set in _SEMANTIC_ALIASES:
        # Use exact equality — prevents "gym" matching "home gym" as a substring
        a_in = a_l in alias_set
        b_in = b_l in alias_set
        if a_in and b_in:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Unit normalisation
# ─────────────────────────────────────────────────────────────────────────────

_KG_TO_LBS = 2.20462
_LBS_TO_KG = 1 / _KG_TO_LBS
_IN_TO_CM  = 2.54

_UNIT_PATTERNS = {
    "weight_kg": [
        (re.compile(r"([\d.]+)\s*kg",      re.IGNORECASE), 1.0),
        (re.compile(r"([\d.]+)\s*lbs?",    re.IGNORECASE), _LBS_TO_KG),
        (re.compile(r"([\d.]+)\s*pounds?", re.IGNORECASE), _LBS_TO_KG),
    ],
    "height_cm": [
        (re.compile(r"([\d.]+)\s*cm",                     re.IGNORECASE), 1.0),
        (re.compile(r"([\d.]+)\s*m\b",                    re.IGNORECASE), 100.0),
        (re.compile(r"([\d.]+)\s*(?:in|inches?|\")",      re.IGNORECASE), _IN_TO_CM),
    ],
}


def _try_parse_with_unit(value_str: str, field_name: str) -> Optional[float]:
    for pattern, factor in _UNIT_PATTERNS.get(field_name, []):
        m = pattern.search(str(value_str))
        if m:
            try:
                return float(m.group(1)) * factor
            except (ValueError, IndexError):
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-field rules
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossFieldRule:
    field_a:     str
    field_b:     str
    description: str
    check:       Any
    severity:    ConflictSeverity = ConflictSeverity.MEDIUM


_CROSS_FIELD_RULES: List[CrossFieldRule] = [
    CrossFieldRule(
        "age", "work_experience_years",
        "age must exceed work_experience_years by at least 12",
        lambda a, b: float(a) > float(b) + 12,
        ConflictSeverity.HIGH,
    ),
    CrossFieldRule(
        "age", "years_training",
        "age must exceed years_training by at least 8",
        lambda a, b: float(a) > float(b) + 8,
        ConflictSeverity.MEDIUM,
    ),
    CrossFieldRule(
        "workout_days_per_week", "rest_days_per_week",
        "workout_days + rest_days should sum to 7",
        lambda wd, rd: abs(float(wd) + float(rd) - 7) <= 1,
        ConflictSeverity.MEDIUM,
    ),
    CrossFieldRule(
        "weight_kg", "height_cm",
        "BMI (weight/height^2) must be between 10 and 70",
        lambda w, h: 10 <= float(w) / ((float(h) / 100) ** 2) <= 70,
        ConflictSeverity.HIGH,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-resolution
# ─────────────────────────────────────────────────────────────────────────────

def _should_auto_resolve(conflict: Conflict) -> Tuple[bool, Optional[Any]]:
    ev = conflict.evidence
    if conflict.conflict_type == ConflictType.RANGE_VIOLATION:
        return True, conflict.old_value
    if conflict.severity == ConflictSeverity.LOW:
        return True, conflict.new_value
    if ev.new_confidence - ev.old_confidence >= 0.35:
        return True, conflict.new_value
    if ev.old_confidence < 0.40 and ev.new_confidence >= 0.70:
        return True, conflict.new_value
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
#  ConflictStore
# ─────────────────────────────────────────────────────────────────────────────

class ConflictStore:
    def __init__(self):
        self._conflicts: List[Conflict] = []

    def add(self, conflict: Conflict) -> bool:
        for ex in self._conflicts:
            if (ex.field == conflict.field
                    and str(ex.old_value) == str(conflict.old_value)
                    and str(ex.new_value) == str(conflict.new_value)
                    and ex.is_open):
                return False
        self._conflicts.append(conflict)
        logger.info("conflict_store: added id=%s field=%s type=%s severity=%s",
                    conflict.id, conflict.field,
                    conflict.conflict_type.value, conflict.severity.value)
        return True

    def resolve(self, conflict_id: str, resolution: Any, turn: int,
                collected: Dict[str, Any]) -> bool:
        c = self._get(conflict_id)
        if c is None:
            return False
        c.resolution = resolution
        c.resolution_turn = turn
        c.status = ConflictStatus.RESOLVED
        c.resolved = True
        c.updated_at = time.time()
        collected[c.field] = resolution
        logger.info("conflict_store: resolved id=%s field=%s resolution=%r",
                    c.id, c.field, resolution)
        return True

    def auto_resolve(self, conflict: Conflict, value: Any, turn: int,
                     collected: Dict[str, Any]) -> None:
        conflict.resolution = value
        conflict.resolution_turn = turn
        conflict.status = ConflictStatus.AUTO_RESOLVED
        conflict.resolved = True
        conflict.updated_at = time.time()
        collected[conflict.field] = value
        logger.info("conflict_store: auto-resolved id=%s field=%s value=%r",
                    conflict.id, conflict.field, value)

    def dismiss(self, conflict_id: str) -> bool:
        c = self._get(conflict_id)
        if c is None:
            return False
        c.status = ConflictStatus.DISMISSED
        c.updated_at = time.time()
        return True

    def escalate(self, conflict_id: str) -> bool:
        c = self._get(conflict_id)
        if c is None:
            return False
        if c.status == ConflictStatus.OPEN:
            c.status = ConflictStatus.ESCALATED
            c.updated_at = time.time()
        return True

    @property
    def open_conflicts(self) -> List[Conflict]:
        return [c for c in self._conflicts if c.is_open]

    @property
    def resolved_conflicts(self) -> List[Conflict]:
        return [c for c in self._conflicts if c.is_resolved]

    def open_for_field(self, field_name: str) -> List[Conflict]:
        return [c for c in self.open_conflicts if c.field == field_name]

    def has_open(self) -> bool:
        return bool(self.open_conflicts)

    def most_severe_open(self) -> Optional[Conflict]:
        order = {ConflictSeverity.CRITICAL: 4, ConflictSeverity.HIGH: 3,
                 ConflictSeverity.MEDIUM: 2, ConflictSeverity.LOW: 1}
        oc = self.open_conflicts
        return max(oc, key=lambda c: order.get(c.severity, 0)) if oc else None

    def to_dicts(self) -> List[dict]:
        return [c.to_dict() for c in self._conflicts]

    def report(self, session_id: str, total_turns: int) -> ConflictReport:
        sev: Dict[str, int] = {s.value: 0 for s in ConflictSeverity}
        fields: Dict[str, int] = {}
        for c in self._conflicts:
            sev[c.severity.value] += 1
            fields[c.field] = fields.get(c.field, 0) + 1
        top = sorted(fields, key=fields.get, reverse=True)[:5]
        return ConflictReport(
            session_id             = session_id,
            total_detected         = len(self._conflicts),
            open_count             = len(self.open_conflicts),
            resolved_count         = sum(1 for c in self._conflicts
                                         if c.status == ConflictStatus.RESOLVED),
            auto_resolved          = sum(1 for c in self._conflicts
                                         if c.status == ConflictStatus.AUTO_RESOLVED),
            dismissed              = sum(1 for c in self._conflicts
                                         if c.status == ConflictStatus.DISMISSED),
            severity_counts        = sev,
            most_conflicted_fields = top,
            conflict_rate          = len(self._conflicts) / max(total_turns, 1),
            all_conflicts          = list(self._conflicts),
        )

    def _get(self, conflict_id: str) -> Optional[Conflict]:
        return next((c for c in self._conflicts if c.id == conflict_id), None)


# ─────────────────────────────────────────────────────────────────────────────
#  ConflictDetector (hardened)
# ─────────────────────────────────────────────────────────────────────────────

class ConflictDetector:
    """
    Hardened conflict detector — 7 types, severity scoring, auto-resolution,
    semantic aliases, unit normalisation, cross-field checks.

    v1-compatible: check() and resolve() APIs preserved.
    """

    NUMERIC_CONFLICT_THRESHOLD:  float = 0.12
    CONFIDENCE_IGNORE_THRESHOLD: float = 0.30

    _id_counter: int = 0

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or {}

    # ------------------------------------------------------------------
    # v1-compatible API
    # ------------------------------------------------------------------

    def check(
        self,
        new_extractions:   Dict[str, Any],
        collected:         Dict[str, Any],
        fields_config:     Dict[str, dict],
        current_turn:      int  = 0,
        turn_map:          Optional[Dict[str, int]]   = None,
        field_confidences: Optional[Dict[str, float]] = None,
        source_texts:      Optional[Dict[str, str]]   = None,
    ) -> List[Conflict]:
        """v1-compatible. Returns detected conflicts without storing."""
        tm   = turn_map or {}
        conf = field_confidences or {}
        srcs = source_texts or {}
        conflicts: List[Conflict] = []
        for fn, new_val in new_extractions.items():
            if fn not in collected:
                continue
            c = self._compare_field(
                field_name     = fn,
                old_val        = collected[fn],
                new_val        = new_val,
                field_cfg      = fields_config.get(fn, {}),
                turn_old       = tm.get(fn, 0),
                turn_new       = current_turn,
                old_confidence = conf.get(fn, 1.0),
                new_confidence = 1.0,
                old_source     = srcs.get(fn, ""),
            )
            if c:
                conflicts.append(c)
                logger.info(
                    "conflict: field=%s type=%s severity=%s old=%r new=%r",
                    fn, c.conflict_type.value, c.severity.value, collected[fn], new_val,
                )
        return conflicts

    def resolve(
        self,
        conflict:         Conflict,
        resolution:       Any,
        collected:        Dict[str, Any],
        active_conflicts: List[dict],
    ) -> Dict[str, Any]:
        """v1-compatible resolve."""
        collected[conflict.field] = resolution
        conflict.resolution = resolution
        conflict.resolved   = True
        conflict.status     = ConflictStatus.RESOLVED
        updated = [c for c in active_conflicts
                   if not (c.get("field") == conflict.field and not c.get("resolved"))]
        active_conflicts.clear()
        active_conflicts.extend(updated)
        logger.info("conflict resolved: field=%s resolution=%r", conflict.field, resolution)
        return collected

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def check_and_store(
        self,
        new_extractions:   Dict[str, Any],
        collected:         Dict[str, Any],
        fields_config:     Dict[str, dict],
        current_turn:      int,
        store:             ConflictStore,
        turn_map:          Optional[Dict[str, int]]   = None,
        field_confidences: Optional[Dict[str, float]] = None,
        source_texts:      Optional[Dict[str, str]]   = None,
    ) -> List[Conflict]:
        """Full check with storage, dedup, and auto-resolution."""
        raw = self.check(new_extractions, collected, fields_config,
                         current_turn, turn_map, field_confidences, source_texts)
        new_open: List[Conflict] = []
        for conflict in raw:
            should_auto, auto_val = _should_auto_resolve(conflict)
            if should_auto:
                store.auto_resolve(conflict, auto_val, current_turn, collected)
                store.add(conflict)
            elif store.add(conflict):
                new_open.append(conflict)
        return new_open

    def check_cross_field(
        self,
        collected:     Dict[str, Any],
        fields_config: Dict[str, dict],
        current_turn:  int,
        store:         Optional[ConflictStore] = None,
    ) -> List[Conflict]:
        """Check collected fields against cross-field consistency rules."""
        conflicts: List[Conflict] = []
        for rule in _CROSS_FIELD_RULES:
            val_a = collected.get(rule.field_a)
            val_b = collected.get(rule.field_b)
            if val_a is None or val_b is None:
                continue
            try:
                if not rule.check(val_a, val_b):
                    c = self._make_conflict(
                        rule.field_a, val_a, val_b,
                        ConflictType.CROSS_FIELD, rule.severity,
                        current_turn, current_turn,
                    )
                    c.related_field = rule.field_b
                    c.evidence.new_source_text = rule.description
                    conflicts.append(c)
                    logger.info("cross-field conflict: %s <-> %s — %s",
                                rule.field_a, rule.field_b, rule.description)
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.debug("cross-field rule error: %s", e)
        if store:
            for c in conflicts:
                store.add(c)
        return conflicts

    def resolve_from_user_input(
        self,
        user_message:  str,
        store:         ConflictStore,
        collected:     Dict[str, Any],
        current_turn:  int,
        fields_config: Optional[Dict[str, dict]] = None,
    ) -> List[str]:
        """Parse user message to resolve open conflicts. Returns resolved IDs."""
        resolved: List[str] = []
        msg = user_message.strip().lower()
        for conflict in store.open_conflicts:
            val = self._parse_resolution(msg, conflict, fields_config or {})
            if val is not None:
                store.resolve(conflict.id, val, current_turn, collected)
                resolved.append(conflict.id)
        return resolved

    # ------------------------------------------------------------------
    # Core comparison
    # ------------------------------------------------------------------

    def _compare_field(
        self,
        field_name:     str,
        old_val:        Any,
        new_val:        Any,
        field_cfg:      dict,
        turn_old:       int,
        turn_new:       int,
        old_confidence: float = 1.0,
        new_confidence: float = 1.0,
        old_source:     str   = "",
        new_source:     str   = "",
    ) -> Optional[Conflict]:
        old_str = str(old_val).strip().lower()
        new_str = str(new_val).strip().lower()

        if old_str == new_str:
            return None

        # Suppress if new extraction is much less confident than old
        if new_confidence < old_confidence - self.CONFIDENCE_IGNORE_THRESHOLD:
            logger.debug("conflict suppressed (low new confidence): field=%s", field_name)
            return None

        # Semantic alias — same concept, different wording
        if _aliases_match(old_str, new_str):
            return None

        ftype    = field_cfg.get("type", "text")
        evidence = ConflictEvidence(
            old_source_text = old_source,
            new_source_text = new_source,
            old_confidence  = old_confidence,
            new_confidence  = new_confidence,
        )

        # Boolean
        if ftype in ("boolean", "bool") or (_is_bool_str(old_str) and _is_bool_str(new_str)):
            if _boolean_flip(old_str, new_str):
                return self._make_conflict(field_name, old_val, new_val,
                    ConflictType.BOOLEAN_FLIP, ConflictSeverity.HIGH,
                    turn_old, turn_new, evidence)

        if ftype in ("number", "integer", "float", "int"):
            num_result = self._check_numeric(field_name, old_val, new_val, field_cfg,
                                             turn_old, turn_new, evidence)
            if num_result is not None:
                return num_result
        um = self._check_unit_mismatch(field_name, old_val, new_val, field_cfg,
                                       turn_old, turn_new, evidence)
        if um:
            return um

        rv = self._check_range(field_name, new_val, field_cfg, old_val,
                               turn_old, turn_new, evidence)
        if rv:
            return rv

        allowed = field_cfg.get("allowed_values") or field_cfg.get("enum", [])
        if allowed:
            al = [str(v).lower() for v in allowed]
            if old_str in al and new_str in al:
                return self._make_conflict(field_name, old_val, new_val,
                    ConflictType.CATEGORICAL_FLIP, ConflictSeverity.HIGH,
                    turn_old, turn_new, evidence)

        if ftype == "text":
            sev, ctype = self._text_conflict_severity(old_str, new_str)
            if sev:
                return self._make_conflict(field_name, old_val, new_val,
                    ctype, sev, turn_old, turn_new, evidence)

        return None

    def _check_numeric(self, field, old_val, new_val, cfg,
                       turn_old, turn_new, evidence) -> Optional[Conflict]:
        try:
            o = float(str(old_val).replace(",", ""))
            n = float(str(new_val).replace(",", ""))
        except (ValueError, TypeError):
            return None

        mn, mx = cfg.get("min"), cfg.get("max")
        if mn is not None and n < mn:
            return self._make_conflict(field, old_val, new_val,
                ConflictType.RANGE_VIOLATION, ConflictSeverity.CRITICAL,
                turn_old, turn_new, evidence)
        if mx is not None and n > mx:
            return self._make_conflict(field, old_val, new_val,
                ConflictType.RANGE_VIOLATION, ConflictSeverity.CRITICAL,
                turn_old, turn_new, evidence)

        if o == 0:
            if n == 0:
                return None
            return self._make_conflict(field, old_val, new_val,
                ConflictType.NUMERIC_MISMATCH, ConflictSeverity.HIGH,
                turn_old, turn_new, evidence)

        diff = abs(o - n) / abs(o)
        if diff <= 0.02:
            return None
        if diff > 0.50:
            sev = ConflictSeverity.CRITICAL
        elif diff > 0.30:
            sev = ConflictSeverity.HIGH
        elif diff > self.NUMERIC_CONFLICT_THRESHOLD:
            sev = ConflictSeverity.MEDIUM
        else:
            return None
        return self._make_conflict(field, old_val, new_val,
            ConflictType.NUMERIC_MISMATCH, sev, turn_old, turn_new, evidence)

    def _check_unit_mismatch(self, field, old_val, new_val, cfg,
                              turn_old, turn_new, evidence) -> Optional[Conflict]:
        if field not in _UNIT_PATTERNS:
            return None
        old_n = _try_parse_with_unit(str(old_val), field)
        new_n = _try_parse_with_unit(str(new_val), field)
        if old_n is None or new_n is None:
            return None
        diff = abs(old_n - new_n) / max(abs(old_n), 1e-9)
        if diff <= 0.05:
            return None  # same value, different unit notation — not a conflict
        return self._make_conflict(field, old_val, new_val,
            ConflictType.UNIT_MISMATCH, ConflictSeverity.MEDIUM,
            turn_old, turn_new, evidence)

    def _check_range(self, field, new_val, cfg, old_val,
                     turn_old, turn_new, evidence) -> Optional[Conflict]:
        mn, mx = cfg.get("min"), cfg.get("max")
        if mn is None and mx is None:
            return None
        try:
            n = float(str(new_val).replace(",", ""))
        except (ValueError, TypeError):
            return None
        if (mn is not None and n < mn) or (mx is not None and n > mx):
            return self._make_conflict(field, old_val, new_val,
                ConflictType.RANGE_VIOLATION, ConflictSeverity.CRITICAL,
                turn_old, turn_new, evidence)
        return None

    def _text_conflict_severity(
        self, old_str: str, new_str: str
    ) -> Tuple[Optional[ConflictSeverity], ConflictType]:
        if old_str in new_str or new_str in old_str:
            return None, ConflictType.TEXT_CONTRADICTION

        antonyms = [
            ({"yes","true","always"}, {"no","false","never"}),
            ({"like","love","enjoy","prefer"}, {"hate","dislike","avoid"}),
            ({"can","able"}, {"cannot","can't","unable"}),
            ({"smoke","smoker"}, {"don't smoke","non-smoker","never smoked"}),
            ({"drink","drinker"}, {"don't drink","non-drinker","teetotal"}),
        ]
        for pos, neg in antonyms:
            if any(p in old_str for p in pos) and any(n in new_str for n in neg):
                return ConflictSeverity.HIGH, ConflictType.SEMANTIC_CONTRADICTION
            if any(p in new_str for p in pos) and any(n in old_str for n in neg):
                return ConflictSeverity.HIGH, ConflictType.SEMANTIC_CONTRADICTION

        if len(old_str) > 8 and len(new_str) > 8:
            ow = set(old_str.split())
            nw = set(new_str.split())
            if ow and nw:
                overlap = len(ow & nw) / max(len(ow), len(nw))
                if overlap < 0.25:
                    return ConflictSeverity.LOW, ConflictType.TEXT_CONTRADICTION
        return None, ConflictType.TEXT_CONTRADICTION

    _PREFER_OLD = re.compile(
        r"\b(the first|earlier|before|original|i said|keep the first|"
        r"my first answer|i already said|stick with)\b", re.IGNORECASE)
    _PREFER_NEW = re.compile(
        r"\b(actually|i meant|i mean|correction|sorry|correct that|"
        r"the second|i changed|new answer|let me correct|the latter)\b", re.IGNORECASE)

    def _parse_resolution(self, msg: str, conflict: Conflict,
                          fields_config: Dict[str, dict]) -> Optional[Any]:
        if self._PREFER_OLD.search(msg):
            return conflict.old_value
        if self._PREFER_NEW.search(msg):
            return conflict.new_value
        if str(conflict.old_value).lower() in msg:
            return conflict.old_value
        if str(conflict.new_value).lower() in msg:
            return conflict.new_value
        ftype = fields_config.get(conflict.field, {}).get("type", "text")
        if ftype in ("integer", "int", "number", "float"):
            nums = re.findall(r"-?\d+(?:\.\d+)?", msg)
            if nums:
                try:
                    v = float(nums[0])
                    return int(v) if ftype in ("integer", "int") else v
                except ValueError:
                    pass
        return None

    @classmethod
    def _make_conflict(
        cls,
        field:         str,
        old_val:       Any,
        new_val:       Any,
        conflict_type: ConflictType,
        severity:      ConflictSeverity,
        turn_old:      int,
        turn_new:      int,
        evidence:      Optional[ConflictEvidence] = None,
    ) -> Conflict:
        cls._id_counter += 1
        return Conflict(
            id            = f"c{cls._id_counter:05d}",
            field         = field,
            old_value     = old_val,
            new_value     = new_val,
            conflict_type = conflict_type,
            severity      = severity,
            turn_old      = turn_old,
            turn_new      = turn_new,
            evidence      = evidence or ConflictEvidence(),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_bool_str(s: str) -> bool:
    return s in ("yes", "no", "true", "false", "y", "n", "1", "0")


def _boolean_flip(a: str, b: str) -> bool:
    for x, y in [("yes","no"),("true","false"),("y","n"),("1","0")]:
        if (a==x and b==y) or (a==y and b==x):
            return True
    return False