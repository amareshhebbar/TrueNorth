"""

Turn-level decision engine. Given the current graph state, the reasoner decides:
  - Which field to target next
  - Whether to acknowledge / clarify / redirect before asking
  - Whether the session is complete
  - Whether to escalate (emotion, conflict, budget)

The reasoner is intentionally rule-based + lightweight. It does NOT call an LLM.
LLM calls happen in conversation_planner (question phrasing) and field_extractor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from truenorth.core.graph_state import GraphState

logger = logging.getLogger(__name__)

class ReasonerAction(str, Enum):
    ASK_FIELD         = "ask_field"
    ASK_OPTIONAL      = "ask_optional"
    CLARIFY           = "clarify"
    ACKNOWLEDGE       = "acknowledge"
    RESOLVE_CONFLICT  = "resolve_conflict"
    HANDLE_EMOTION    = "handle_emotion"
    GENERATE_OUTPUT   = "generate_output"
    WAIT              = "wait"
    END               = "end"
    BUDGET_EXCEEDED   = "budget_exceeded"

@dataclass
class ReasonerDecision:
    action: ReasonerAction
    target_field: Optional[str] = None
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ReasonerDecision(action={self.action.value}, field={self.target_field}, reason={self.reason!r})"

class Reasoner:
    """
    Stateless decision engine. Call decide() once per conversation turn.

    Priority order (highest → lowest):
      1. Budget exceeded
      2. Critical emotion (distress / anger)
      3. Unresolved conflict
      4. Required field missing
      5. Clarification needed (low-confidence last extraction)
      6. Optional fields
      7. Generate output (all required fields collected)
      8. End
    """

    CLARIFY_THRESHOLD: float = 0.45

    DISTRESS_THRESHOLD: float = 0.70

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.clarify_threshold = self.config.get("clarify_threshold", self.CLARIFY_THRESHOLD)
        self.distress_threshold = self.config.get("distress_threshold", self.DISTRESS_THRESHOLD)

    def decide(self, state: "GraphState") -> ReasonerDecision:
        """
        Examine the graph state and return the next action.

        Args:
            state: Current session graph state (fields, emotion, conflicts, budget, …)

        Returns:
            ReasonerDecision describing what the engine should do next.
        """

        if self._budget_exceeded(state):
            logger.warning("session=%s budget exceeded — ending session", state.session_id)
            return ReasonerDecision(
                action=ReasonerAction.BUDGET_EXCEEDED,
                reason="Cost budget exceeded for this session",
            )

        emotion_decision = self._check_emotion(state)
        if emotion_decision:
            return emotion_decision

        conflict_decision = self._check_conflicts(state)
        if conflict_decision:
            return conflict_decision

        required_decision = self._next_required_field(state)
        if required_decision:
            return required_decision

        clarify_decision = self._check_clarification(state)
        if clarify_decision:
            return clarify_decision

        if self._all_required_collected(state):

            optional_decision = self._next_optional_field(state)
            if optional_decision:
                return optional_decision

            logger.info("session=%s all required fields collected → generate output", state.session_id)
            return ReasonerDecision(
                action=ReasonerAction.GENERATE_OUTPUT,
                reason="All required fields collected",
            )

        logger.error("session=%s reasoner fell through all checks — ending", state.session_id)
        return ReasonerDecision(action=ReasonerAction.END, reason="No further action determined")

    def _budget_exceeded(self, state: "GraphState") -> bool:
        budget = getattr(state, "cost_budget_usd", None)
        spent = getattr(state, "total_cost_usd", 0.0)
        if budget is not None and spent >= budget:
            return True
        return False

    def _check_emotion(self, state: "GraphState") -> Optional[ReasonerDecision]:
        emotion = getattr(state, "current_emotion", None)
        if emotion is None:
            return None
        score = emotion.get("score", 0.0)
        label = emotion.get("label", "neutral")
        if label in ("distress", "anger", "frustration") and score >= self.distress_threshold:
            return ReasonerDecision(
                action=ReasonerAction.HANDLE_EMOTION,
                reason=f"Detected {label} with score {score:.2f}",
                metadata={"emotion": emotion},
            )
        return None

    def _check_conflicts(self, state: "GraphState") -> Optional[ReasonerDecision]:
        conflicts = getattr(state, "active_conflicts", [])
        if not conflicts:
            return None

        conflict = conflicts[-1]
        return ReasonerDecision(
            action=ReasonerAction.RESOLVE_CONFLICT,
            reason=f"Conflict on field '{conflict.get('field')}'",
            metadata={"conflict": conflict},
        )

    def _next_required_field(self, state: "GraphState") -> Optional[ReasonerDecision]:
        from truenorth.core.field_tree import FieldTree
        fields_config: dict = getattr(state, "fields_config", {})
        collected: dict = getattr(state, "collected_fields", {})
        skipped: set = getattr(state, "skipped_fields", set())

        ft = FieldTree(fields_config)
        next_field = ft.next_required(collected, skipped)
        if next_field:
            return ReasonerDecision(
                action=ReasonerAction.ASK_FIELD,
                target_field=next_field,
                reason=f"Required field '{next_field}' not yet collected",
            )
        return None

    def _next_optional_field(self, state: "GraphState") -> Optional[ReasonerDecision]:
        from truenorth.core.field_tree import FieldTree
        fields_config: dict = getattr(state, "fields_config", {})
        collected: dict = getattr(state, "collected_fields", {})
        skipped: set = getattr(state, "skipped_fields", set())
        asked_optional: set = getattr(state, "asked_optional_fields", set())

        max_optional = self.config.get("max_optional_fields", 3)
        ft = FieldTree(fields_config)
        next_field = ft.next_optional(
            collected_fields = collected,
            skipped_fields   = skipped,
            asked_optional   = asked_optional,
            max_optional     = max_optional,
        )
        if next_field:
            return ReasonerDecision(
                action=ReasonerAction.ASK_OPTIONAL,
                target_field=next_field,
                reason=f"Optional field '{next_field}' not yet collected",
            )
        return None

    def _check_clarification(self, state: "GraphState") -> Optional[ReasonerDecision]:
        last_extraction = getattr(state, "last_extraction", None)
        if last_extraction is None:
            return None
        confidence = last_extraction.get("confidence", 1.0)
        field_name = last_extraction.get("field")
        if confidence < self.clarify_threshold and field_name:
            return ReasonerDecision(
                action=ReasonerAction.CLARIFY,
                target_field=field_name,
                reason=f"Low confidence ({confidence:.2f}) on last extraction for '{field_name}'",
                metadata={"last_extraction": last_extraction},
            )
        return None

    def _all_required_collected(self, state: "GraphState") -> bool:
        from truenorth.core.field_tree import FieldTree
        fields_config: dict = getattr(state, "fields_config", {})
        collected: dict = getattr(state, "collected_fields", {})
        skipped: set = getattr(state, "skipped_fields", set())
        ft = FieldTree(fields_config)
        return ft.all_required_collected(collected)

    def _field_condition_met(self, field_name: str, field_cfg: dict, collected: dict) -> bool:
        """
        Evaluate if_true / if_value_is gates on a field.
        Returns True if the field should be asked (condition satisfied or no condition).
        """
        if_true = field_cfg.get("if_true")
        if_value_is = field_cfg.get("if_value_is")

        if if_true:
            gate_val = collected.get(if_true)
            if not gate_val:
                return False

        if if_value_is:
            gate_field = if_value_is.get("field")
            gate_value = if_value_is.get("value")
            if collected.get(gate_field) != gate_value:
                return False

        return True

    def explain(self, state: "GraphState") -> str:
        """Return a human-readable explanation of the current decision (for dry-run / debug)."""
        decision = self.decide(state)
        return (
            f"Action     : {decision.action.value}\n"
            f"Target     : {decision.target_field or '—'}\n"
            f"Reason     : {decision.reason}\n"
            f"Metadata   : {decision.metadata or '—'}"
        )
