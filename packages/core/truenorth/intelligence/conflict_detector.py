"""
truenorth/intelligence/conflict_detector.py

Detects contradictions between newly extracted field values and previously
collected values for the same fields.

Examples of conflicts caught:
  - Turn 3: age=25  →  Turn 7: age=30   (numeric mismatch)
  - Turn 2: goal=weight_loss  →  Turn 5: goal=muscle_gain (semantic mismatch)
  - Turn 4: smoker=no  →  Turn 8: smoker=yes (boolean flip)

Resolution strategy:
  - Flag the conflict in graph_state.active_conflicts
  - Reasoner picks it up → ConversationPlanner asks clarifying question
  - User response resolves conflict → winning value replaces old one
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conflict types
# ---------------------------------------------------------------------------

class ConflictType:
    NUMERIC_MISMATCH  = "numeric_mismatch"    # 25 vs 30
    BOOLEAN_FLIP      = "boolean_flip"        # yes vs no
    CATEGORICAL_FLIP  = "categorical_flip"    # weight_loss vs muscle_gain
    TEXT_CONTRADICTION = "text_contradiction" # semantically opposite text
    RANGE_VIOLATION   = "range_violation"     # value outside declared min/max


@dataclass
class Conflict:
    field:        str
    old_value:    Any
    new_value:    Any
    conflict_type: str
    turn_old:     int          # turn when old value was collected
    turn_new:     int          # turn when new value was collected
    resolved:     bool = False
    resolution:   Optional[Any] = None   # the accepted final value

    def to_dict(self) -> dict:
        return {
            "field":         self.field,
            "old_value":     self.old_value,
            "new_value":     self.new_value,
            "conflict_type": self.conflict_type,
            "turn_old":      self.turn_old,
            "turn_new":      self.turn_new,
            "resolved":      self.resolved,
            "resolution":    self.resolution,
        }

    def clarification_question(self, field_config: Optional[dict] = None) -> str:
        """Generate a natural clarifying question for this conflict."""
        fname = (field_config or {}).get("label", self.field.replace("_", " "))
        return (
            f"I noticed you mentioned {fname} as {self.old_value!r} earlier, "
            f"but now it seems like {self.new_value!r}. "
            f"Which one is correct?"
        )


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """
    Compares newly extracted values against the current collection to find conflicts.

    Usage:
        detector = ConflictDetector()
        conflicts = detector.check(
            new_extractions={"age": 30, "goal": "muscle_gain"},
            collected={"age": 25, "goal": "weight_loss"},
            fields_config=goal_config["fields"],
            current_turn=7,
        )
    """

    NUMERIC_CONFLICT_THRESHOLD: float = 0.15 

    def check(
        self,
        new_extractions: Dict[str, Any],
        collected:       Dict[str, Any],
        fields_config:   Dict[str, dict],
        current_turn:    int = 0,
        turn_map:        Optional[Dict[str, int]] = None,
    ) -> List[Conflict]:
        """
        Compare new extractions against already-collected values.

        Args:
            new_extractions: Freshly extracted {field: value} this turn
            collected:       Already collected {field: value}
            fields_config:   Goal YAML field specs {field_name: spec}
            current_turn:    Current turn number
            turn_map:        {field_name: turn_number} when each field was collected

        Returns:
            List of Conflict objects (may be empty)
        """
        turn_map  = turn_map or {}
        conflicts: List[Conflict] = []

        for field_name, new_val in new_extractions.items():
            if field_name not in collected:
                continue  # first time we've seen this field — no conflict possible

            old_val   = collected[field_name]
            field_cfg = fields_config.get(field_name, {})
            ftype     = field_cfg.get("type", "text")
            turn_old  = turn_map.get(field_name, 0)

            conflict = self._compare(
                field      = field_name,
                old_val    = old_val,
                new_val    = new_val,
                ftype      = ftype,
                field_cfg  = field_cfg,
                turn_old   = turn_old,
                turn_new   = current_turn,
            )
            if conflict:
                conflicts.append(conflict)
                logger.info(
                    "conflict detected: field=%s old=%r new=%r type=%s",
                    field_name, old_val, new_val, conflict.conflict_type,
                )

        return conflicts

    def resolve(
        self,
        conflict:   Conflict,
        resolution: Any,
        collected:  Dict[str, Any],
        active_conflicts: List[dict],
    ) -> Dict[str, Any]:
        """
        Apply a resolved conflict — update collected fields and remove from active list.
        Returns updated collected_fields dict.
        """
        collected[conflict.field] = resolution
        conflict.resolved   = True
        conflict.resolution = resolution

        # Remove from active_conflicts list (which stores dicts)
        updated = [
            c for c in active_conflicts
            if not (c.get("field") == conflict.field and not c.get("resolved"))
        ]
        active_conflicts.clear()
        active_conflicts.extend(updated)

        logger.info(
            "conflict resolved: field=%s resolution=%r", conflict.field, resolution
        )
        return collected

    # ------------------------------------------------------------------
    # Internal comparison logic
    # ------------------------------------------------------------------

    def _compare(
        self,
        field:     str,
        old_val:   Any,
        new_val:   Any,
        ftype:     str,
        field_cfg: dict,
        turn_old:  int,
        turn_new:  int,
    ) -> Optional[Conflict]:
        """Return a Conflict if old_val and new_val are genuinely different, else None."""

        old_str = str(old_val).strip().lower()
        new_str = str(new_val).strip().lower()

        if old_str == new_str:
            return None

        if ftype in ("boolean", "bool") or old_str in ("yes", "no", "true", "false"):
            if self._is_boolean_flip(old_str, new_str):
                return Conflict(
                    field=field, old_value=old_val, new_value=new_val,
                    conflict_type=ConflictType.BOOLEAN_FLIP,
                    turn_old=turn_old, turn_new=turn_new,
                )

        if ftype in ("number", "integer", "float"):
            conflict = self._check_numeric(
                field, old_val, new_val, field_cfg, turn_old, turn_new
            )
            if conflict:
                return conflict

        range_conflict = self._check_range(
            field, new_val, field_cfg, old_val, turn_old, turn_new
        )
        if range_conflict:
            return range_conflict

        allowed = field_cfg.get("allowed_values") or field_cfg.get("enum", [])
        if allowed and old_str != new_str:
            if old_str in [v.lower() for v in allowed] and new_str in [v.lower() for v in allowed]:
                return Conflict(
                    field=field, old_value=old_val, new_value=new_val,
                    conflict_type=ConflictType.CATEGORICAL_FLIP,
                    turn_old=turn_old, turn_new=turn_new,
                )

        if ftype == "text" and self._is_text_contradiction(old_str, new_str):
            return Conflict(
                field=field, old_value=old_val, new_value=new_val,
                conflict_type=ConflictType.TEXT_CONTRADICTION,
                turn_old=turn_old, turn_new=turn_new,
            )

        return None

    @staticmethod
    def _is_boolean_flip(a: str, b: str) -> bool:
        opposites = [("yes", "no"), ("true", "false"), ("y", "n"), ("1", "0")]
        for x, y in opposites:
            if (a == x and b == y) or (a == y and b == x):
                return True
        return False

    def _check_numeric(
        self, field, old_val, new_val, cfg, turn_old, turn_new
    ) -> Optional[Conflict]:
        try:
            o = float(str(old_val).replace(",", ""))
            n = float(str(new_val).replace(",", ""))
        except (ValueError, TypeError):
            return None

        if o == 0:
            return None  

        diff = abs(o - n) / abs(o)
        if diff > self.NUMERIC_CONFLICT_THRESHOLD:
            return Conflict(
                field=field, old_value=old_val, new_value=new_val,
                conflict_type=ConflictType.NUMERIC_MISMATCH,
                turn_old=turn_old, turn_new=turn_new,
            )
        return None

    @staticmethod
    def _check_range(field, new_val, cfg, old_val, turn_old, turn_new) -> Optional[Conflict]:
        mn = cfg.get("min")
        mx = cfg.get("max")
        if mn is None and mx is None:
            return None
        try:
            n = float(str(new_val).replace(",", ""))
        except (ValueError, TypeError):
            return None
        if (mn is not None and n < mn) or (mx is not None and n > mx):
            return Conflict(
                field=field, old_value=old_val, new_value=new_val,
                conflict_type=ConflictType.RANGE_VIOLATION,
                turn_old=turn_old, turn_new=turn_new,
            )
        return None

    @staticmethod
    def _is_text_contradiction(a: str, b: str) -> bool:
        """Simple heuristic: flag if neither string is a substring of the other."""
        return a not in b and b not in a and len(a) > 8 and len(b) > 8