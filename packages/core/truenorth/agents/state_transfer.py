"""
Cross-goal state transfer for TrueNorth's multi-goal chains.

When a user completes a fitness goal and moves to a nutrition goal,
they shouldn't be asked for their age, weight, and activity level again —
TrueNorth already has that. StateTransfer carries the right fields forward.

Core concepts:

  FieldMap       — declares which fields in Goal A correspond to fields in Goal B.
                   Example: fitness_plan.age → nutrition_plan.user_age
                   Can be explicit or inferred by field name similarity.

  StateTransfer  — extracts collected fields from a completed session and
                   seeds them into a new session, skipping fields that are
                   already known.

  GoalChain      — a sequence of goals to run in order, each seeded by
                   the previous one. Declared in YAML under `chain:`.

  GoalRouter     — given completed state, picks the NEXT goal to run.
                   Can be rule-based (if goal=X, next=Y) or LLM-driven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    """One field-to-field mapping between source and target goal."""
    source_field:  str
    target_field:  str
    transform:     Optional[Callable[[Any], Any]] = None
    condition:     Optional[Callable[[Any], bool]] = None

    def apply(self, value: Any) -> Tuple[bool, Any]:
        """
        Apply the mapping. Returns (should_carry, transformed_value).
        """
        if self.condition and not self.condition(value):
            return False, None
        if self.transform:
            try:
                value = self.transform(value)
            except Exception as e:
                logger.warning(
                    "state_transfer: transform failed %s→%s: %s",
                    self.source_field, self.target_field, e,
                )
                return False, None
        return True, value

class FieldMap:
    """
    Collection of FieldMappings between two goals.
    """

    def __init__(self):
        self._mappings: List[FieldMapping] = []

    def add(
        self,
        source: str,
        target: str,
        transform:  Optional[Callable] = None,
        condition:  Optional[Callable] = None,
    ) -> "FieldMap":
        self._mappings.append(FieldMapping(source, target, transform, condition))
        return self

    def add_direct(self, *field_names: str) -> "FieldMap":
        """Add same-name mappings (field carries with identical name)."""
        for name in field_names:
            self._mappings.append(FieldMapping(name, name))
        return self

    @classmethod
    def from_yaml(cls, carry_fields: List[Any]) -> "FieldMap":
        """
        Parse the YAML carry_fields list.
        Accepts: "age" (direct) or {"age": "user_age"} (rename).
        """
        fm = cls()
        for item in carry_fields:
            if isinstance(item, str):
                fm.add_direct(item)
            elif isinstance(item, dict):
                for src, tgt in item.items():
                    if tgt is None or tgt == src:
                        fm.add_direct(src)
                    else:
                        fm.add(src, tgt)
        return fm

    @classmethod
    def auto_infer(
        cls,
        source_fields: List[str],
        target_fields: List[str],
    ) -> "FieldMap":
        """
        Auto-infer mappings: any field that exists in BOTH source and target
        with the same name carries directly.
        """
        fm = cls()
        common = set(source_fields) & set(target_fields)
        for f in sorted(common):
            fm.add_direct(f)
        return fm

    @property
    def mappings(self) -> List[FieldMapping]:
        return list(self._mappings)

@dataclass
class TransferResult:
    """Result of one state transfer."""
    source_goal_id:  str
    target_goal_id:  str
    carried_fields:  Dict[str, Any]
    skipped_fields:  List[str]
    missing_fields:  List[str]

    @property
    def coverage_pct(self) -> float:
        total = len(self.carried_fields) + len(self.missing_fields)
        return round(len(self.carried_fields) / max(total, 1) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "source_goal_id": self.source_goal_id,
            "target_goal_id": self.target_goal_id,
            "carried_count":  len(self.carried_fields),
            "skipped_count":  len(self.skipped_fields),
            "missing_count":  len(self.missing_fields),
            "coverage_pct":   self.coverage_pct,
            "carried_fields": list(self.carried_fields.keys()),
        }

class StateTransfer:
    """
    Extracts collected fields from a completed session and seeds them
    into a new session for a different goal.
    """

    def __init__(
        self,
        field_map:           Optional[FieldMap] = None,
        auto_infer:          bool               = True,
        confidence_threshold: float             = 0.70,
    ):
        self._field_map   = field_map
        self._auto_infer  = auto_infer
        self._min_conf    = confidence_threshold

    def extract(
        self,
        source_state:           dict,
        source_goal_id:         str,
        target_goal_id:         str,
        target_required_fields: Optional[List[str]] = None,
        target_fields_config:   Optional[dict]      = None,
    ) -> TransferResult:
        """
        Extract fields from completed source state and map them to target goal.

        Args:
            source_state:            to_dict() output of completed engine state
            source_goal_id:          id of the source goal
            target_goal_id:          id of the target goal
            target_required_fields:  list of required field names in target goal
            target_fields_config:    {field_name: config} of target goal fields

        Returns:
            TransferResult with carried_fields ready to seed the new engine.
        """
        collected   = source_state.get("collected_fields", {})
        confidences = source_state.get("field_confidences", {})

        field_map = self._field_map
        if field_map is None and self._auto_infer:
            target_fields = list(target_required_fields or [])
            if target_fields_config:
                target_fields = list(target_fields_config.keys())
            field_map = FieldMap.auto_infer(
                source_fields = list(collected.keys()),
                target_fields = target_fields,
            )

        if field_map is None:
            field_map = FieldMap()

        carried:  Dict[str, Any] = {}
        skipped:  List[str]      = []

        for mapping in field_map.mappings:
            if mapping.source_field not in collected:
                continue
            value = collected[mapping.source_field]
            conf  = confidences.get(mapping.source_field, 1.0)

            if conf < self._min_conf:
                logger.debug(
                    "state_transfer: skipping %s (confidence %.2f < threshold %.2f)",
                    mapping.source_field, conf, self._min_conf,
                )
                skipped.append(mapping.source_field)
                continue

            ok, transformed = mapping.apply(value)
            if ok:
                carried[mapping.target_field] = transformed
            else:
                skipped.append(mapping.source_field)

        target_required = list(target_required_fields or [])
        if target_fields_config:
            target_required = [
                f for f, cfg in target_fields_config.items()
                if cfg.get("required", True)
            ]
        missing = [f for f in target_required if f not in carried]

        result = TransferResult(
            source_goal_id = source_goal_id,
            target_goal_id = target_goal_id,
            carried_fields = carried,
            skipped_fields = skipped,
            missing_fields = missing,
        )
        logger.info(
            "state_transfer: %s → %s: carried=%d skipped=%d missing=%d coverage=%.1f%%",
            source_goal_id, target_goal_id,
            len(carried), len(skipped), len(missing), result.coverage_pct,
        )
        return result

    def seed_engine(
        self,
        engine:         Any,
        carried_fields: Dict[str, Any],
        confidences:    Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Pre-fill an engine's collected_fields with transferred values.
        Returns the number of fields successfully seeded.

        Call this AFTER engine.start() and BEFORE the first user message.
        """
        count = 0
        confs = confidences or {}
        for field_name, value in carried_fields.items():
            if field_name in engine.state.fields_config:
                conf = confs.get(field_name, 0.90)
                engine.state.set_field(field_name, value, confidence=conf)
                count += 1
                logger.debug(
                    "state_transfer: seeded field=%s value=%r conf=%.2f",
                    field_name, value, conf,
                )
            else:
                logger.debug(
                    "state_transfer: skipped unknown field=%s in target", field_name
                )
        return count

@dataclass
class ChainStep:
    """One step in a goal chain."""
    goal_id:      str
    carry_fields: List[Any] = field(default_factory=list)
    condition:    Optional[Dict[str, Any]] = None
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def condition_met(self, collected_fields: dict) -> bool:
        """Return True if this step's condition is satisfied."""
        if not self.condition:
            return True
        for field_name, expected in self.condition.items():
            actual = collected_fields.get(field_name)
            if actual != expected:
                return False
        return True

class GoalChain:
    """
    Declarative sequence of goals, each seeded by the previous one.

    Build from YAML `chain:` section or programmatically:

        chain = GoalChain([
            ChainStep("fitness_plan",   carry_fields=["age", "weight_kg"]),
            ChainStep("nutrition_plan", carry_fields=[{"age": "user_age"}],
                      condition={"primary_goal": "lose weight"}),
            ChainStep("exercise_plan",  carry_fields=[{"age": "user_age"}]),
        ])

        next_step = chain.next(
            current_goal_id  = "fitness_plan",
            collected_fields = state.collected_fields,
        )
    """

    def __init__(self, steps: List[ChainStep]):
        self._steps = steps

    @classmethod
    def from_yaml(cls, chain_config: dict) -> "GoalChain":
        """
        Parse the YAML `chain:` block.
        Expected format:
            chain:
              on_complete:
                - if:   {primary_goal: "lose weight"}
                  then: nutrition_plan
                  carry_fields: [{age: user_age}]
                - else: exercise_plan
        """
        steps: List[ChainStep] = []
        for rule in chain_config.get("on_complete", []):
            if "if" in rule:
                goal_id      = rule.get("then", "")
                carry_fields = rule.get("carry_fields", [])
                steps.append(ChainStep(
                    goal_id      = goal_id,
                    carry_fields = carry_fields,
                    condition    = rule["if"],
                ))
            elif "else" in rule:
                goal_id      = rule["else"]
                carry_fields = rule.get("carry_fields", [])
                steps.append(ChainStep(goal_id=goal_id, carry_fields=carry_fields))
        return cls(steps)

    def next(
        self,
        current_goal_id:  str,
        collected_fields: dict,
    ) -> Optional[ChainStep]:
        """
        Find the next step to run, given the current goal and collected state.
        Evaluates conditions in order; returns the first matching step.
        """
        for step in self._steps:
            if step.goal_id == current_goal_id:
                continue
            if step.condition_met(collected_fields):
                logger.info(
                    "goal_chain: %s → %s (condition_met)",
                    current_goal_id, step.goal_id,
                )
                return step

        logger.info("goal_chain: no next step found after %s", current_goal_id)
        return None

    def field_map_for(self, step: ChainStep) -> FieldMap:
        """Build the FieldMap for a specific chain step."""
        return FieldMap.from_yaml(step.carry_fields)

    def all_goals(self) -> List[str]:
        return [s.goal_id for s in self._steps]
