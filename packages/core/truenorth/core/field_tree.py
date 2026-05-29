"""
truenorth/core/field_tree.py

Conditional field tree evaluator.

Allows goal YAML to declare fields that are only asked when certain
conditions are met based on previously collected values.

Supported condition types in YAML:
  if_true:           "field_name"            # ask if collected field is truthy
  if_false:          "field_name"            # ask if collected field is falsy
  if_value_is:       {field: "x", value: "y"}# ask if field == value (exact)
  if_value_in:       {field: "x", values: [y,z]} # ask if field in list
  if_value_not:      {field: "x", value: "y"}# ask if field != value
  if_numeric_gt:     {field: "x", value: N}  # ask if field > N
  if_numeric_lt:     {field: "x", value: N}  # ask if field < N
  if_all_of:         [{condition}, {condition}] # ALL conditions must pass
  if_any_of:         [{condition}, {condition}] # ANY condition must pass

Example YAML:
  fields:
    - name: smoker
      type: boolean
      required: true
      question: "Do you smoke?"

    - name: cigarettes_per_day
      type: integer
      required: true
      question: "How many cigarettes a day?"
      if_true: smoker             # only asked when smoker=true/yes

    - name: quit_date
      type: date
      required: false
      question: "When did you quit?"
      if_value_is:
        field: smoker
        value: "no"

    - name: heavy_lifting_program
      type: text
      required: false
      question: "What lifting program are you on?"
      if_value_in:
        field: primary_goal
        values: ["build muscle", "powerlifting", "strength"]

    - name: injury_detail
      type: text
      required: true
      question: "What type of injury?"
      if_any_of:
        - if_true: has_injury
        - if_value_not: {field: injury_level, value: "none"}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Condition evaluators
# ─────────────────────────────────────────────────────────────────────────────

def _truthy(val: Any) -> bool:
    """True if value is set and not a falsy sentinel."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s not in ("", "false", "no", "n", "0", "none", "null", "skip", "[skip]")


def _falsy(val: Any) -> bool:
    return not _truthy(val)


def _normalize(val: Any) -> str:
    """Normalize a value for comparison (lowercase, stripped)."""
    return str(val).strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
#  FieldTree
# ─────────────────────────────────────────────────────────────────────────────

class FieldTree:
    """
    Evaluates conditional field visibility rules.

    Usage:
        ft = FieldTree(fields_config)

        # Check if a field should be shown
        if ft.is_visible("cigarettes_per_day", collected_fields):
            # ask the question

        # Get all currently visible fields
        visible = ft.visible_fields(collected_fields)

        # Get the next field to ask (first visible uncollected required field)
        next_field = ft.next_required(collected_fields, skipped_fields)
    """

    def __init__(self, fields_config: Dict[str, dict]):
        self._cfg = fields_config
        # Build ordered list preserving YAML declaration order
        self._ordered = list(fields_config.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_visible(
        self,
        field_name:       str,
        collected_fields: Dict[str, Any],
    ) -> bool:
        """
        Return True if this field's conditions are satisfied and it should be asked.

        A field with no conditions is always visible.
        A field whose condition references an uncollected field is NOT visible
        (the gating field must be answered first).
        """
        cfg = self._cfg.get(field_name, {})
        return self._eval_all_conditions(cfg, collected_fields)

    def visible_fields(self, collected_fields: Dict[str, Any]) -> List[str]:
        """Return all field names that are currently visible."""
        return [f for f in self._ordered if self.is_visible(f, collected_fields)]

    def next_required(
        self,
        collected_fields: Dict[str, Any],
        skipped_fields:   Optional[set] = None,
    ) -> Optional[str]:
        """
        Return the name of the next required, uncollected, visible field.
        Returns None when all required visible fields are collected.
        """
        skipped = skipped_fields or set()
        for field_name in self._ordered:
            cfg = self._cfg.get(field_name, {})
            if not cfg.get("required", True):
                continue
            if field_name in collected_fields:
                continue
            if field_name in skipped:
                continue
            if self.is_visible(field_name, collected_fields):
                return field_name
        return None

    def next_optional(
        self,
        collected_fields: Dict[str, Any],
        skipped_fields:   Optional[set] = None,
        asked_optional:   Optional[set] = None,
        max_optional:     int = 3,
    ) -> Optional[str]:
        """
        Return the next optional field to ask (if any).
        Stops after max_optional have been asked.
        """
        skipped       = skipped_fields or set()
        asked         = asked_optional or set()
        if len(asked) >= max_optional:
            return None

        for field_name in self._ordered:
            cfg = self._cfg.get(field_name, {})
            if cfg.get("required", True):
                continue
            if field_name in collected_fields:
                continue
            if field_name in skipped:
                continue
            if field_name in asked:
                continue
            if self.is_visible(field_name, collected_fields):
                return field_name
        return None

    def all_required_collected(self, collected_fields: Dict[str, Any]) -> bool:
        """True when every required visible field has been collected."""
        return self.next_required(collected_fields) is None

    def dependency_summary(self) -> Dict[str, List[str]]:
        """
        Return a dict of {field_name: [gating_fields...]} for documentation.
        Used by dry-run and Studio to show the field dependency graph.
        """
        result: Dict[str, List[str]] = {}
        for fn, cfg in self._cfg.items():
            gates = self._extract_gate_fields(cfg)
            if gates:
                result[fn] = gates
        return result

    # ------------------------------------------------------------------
    # Condition evaluators
    # ------------------------------------------------------------------

    def _eval_all_conditions(
        self,
        cfg:              dict,
        collected_fields: Dict[str, Any],
    ) -> bool:
        """
        Evaluate ALL condition types present in a field config.
        All present conditions must pass (implicit AND).
        """
        # if_true: field_name
        if_true = cfg.get("if_true")
        if if_true:
            if not _truthy(collected_fields.get(if_true)):
                return False

        # if_false: field_name
        if_false = cfg.get("if_false")
        if if_false:
            if not _falsy(collected_fields.get(if_false)):
                return False

        # if_value_is: {field: x, value: y}
        if_value_is = cfg.get("if_value_is")
        if if_value_is:
            if not self._eval_value_is(if_value_is, collected_fields):
                return False

        # if_value_in: {field: x, values: [y, z]}
        if_value_in = cfg.get("if_value_in")
        if if_value_in:
            if not self._eval_value_in(if_value_in, collected_fields):
                return False

        # if_value_not: {field: x, value: y}
        if_value_not = cfg.get("if_value_not")
        if if_value_not:
            if not self._eval_value_not(if_value_not, collected_fields):
                return False

        # if_numeric_gt: {field: x, value: N}
        if_gt = cfg.get("if_numeric_gt")
        if if_gt:
            if not self._eval_numeric(if_gt, collected_fields, ">"):
                return False

        # if_numeric_lt: {field: x, value: N}
        if_lt = cfg.get("if_numeric_lt")
        if if_lt:
            if not self._eval_numeric(if_lt, collected_fields, "<"):
                return False

        # if_all_of: [{condition}, ...]  — all must pass
        if_all = cfg.get("if_all_of")
        if if_all:
            if not all(
                self._eval_all_conditions(c, collected_fields)
                for c in if_all
            ):
                return False

        # if_any_of: [{condition}, ...]  — at least one must pass
        if_any = cfg.get("if_any_of")
        if if_any:
            if not any(
                self._eval_all_conditions(c, collected_fields)
                for c in if_any
            ):
                return False

        return True

    @staticmethod
    def _eval_value_is(cond: dict, collected: Dict[str, Any]) -> bool:
        gate_field = cond.get("field")
        gate_value = cond.get("value")
        if gate_field is None:
            return True
        collected_val = collected.get(gate_field)
        if collected_val is None:
            return False    # gate field not yet collected
        return _normalize(collected_val) == _normalize(gate_value)

    @staticmethod
    def _eval_value_in(cond: dict, collected: Dict[str, Any]) -> bool:
        gate_field  = cond.get("field")
        gate_values = cond.get("values", [])
        if gate_field is None:
            return True
        collected_val = collected.get(gate_field)
        if collected_val is None:
            return False
        normalized    = _normalize(collected_val)
        return any(normalized == _normalize(v) for v in gate_values)

    @staticmethod
    def _eval_value_not(cond: dict, collected: Dict[str, Any]) -> bool:
        gate_field = cond.get("field")
        gate_value = cond.get("value")
        if gate_field is None:
            return True
        collected_val = collected.get(gate_field)
        if collected_val is None:
            return True    # not set → not equal to gate_value → condition passes
        return _normalize(collected_val) != _normalize(gate_value)

    @staticmethod
    def _eval_numeric(cond: dict, collected: Dict[str, Any], op: str) -> bool:
        gate_field = cond.get("field")
        threshold  = cond.get("value")
        if gate_field is None or threshold is None:
            return True
        collected_val = collected.get(gate_field)
        if collected_val is None:
            return False
        try:
            num = float(str(collected_val).replace(",", ""))
            thr = float(threshold)
            if op == ">":
                return num > thr
            if op == "<":
                return num < thr
            if op == ">=":
                return num >= thr
            if op == "<=":
                return num <= thr
        except (ValueError, TypeError):
            pass
        return False

    @staticmethod
    def _extract_gate_fields(cfg: dict) -> List[str]:
        """Extract all field names that gate this field (for dependency graph)."""
        gates: List[str] = []
        for key in ("if_true", "if_false"):
            if val := cfg.get(key):
                gates.append(val)
        for key in ("if_value_is", "if_value_in", "if_value_not",
                    "if_numeric_gt", "if_numeric_lt"):
            if cond := cfg.get(key):
                if f := cond.get("field"):
                    gates.append(f)
        for cond_list in (cfg.get("if_all_of", []), cfg.get("if_any_of", [])):
            for sub in cond_list:
                for key in ("if_true", "if_false"):
                    if val := sub.get(key):
                        gates.append(val)
                for key in ("if_value_is", "if_value_in", "if_value_not",
                            "if_numeric_gt", "if_numeric_lt"):
                    if cond := sub.get(key):
                        if f := cond.get("field"):
                            gates.append(f)
        return list(dict.fromkeys(gates))   # deduplicated, order-preserved