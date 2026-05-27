"""
Detect contradictions in collected profile data.

soft_conflict: plausible but inconsistent (vegetarian vs ate chicken)
hard_conflict: logically impossible (age 15 + worked 30 years)
"""

from __future__ import annotations
from dataclasses import dataclass
from truenorth.core.graph_state import GraphState, FieldValue
from truenorth.llm.router import LLMRouter

SYSTEM = "You are a data consistency checker for a structured data collection system."

PROMPT = """
Existing profile:
{profile}

New value being added:
  Field: {field_name}
  Value: {new_value}
  From user text: "{raw_text}"

Does this new value contradict any existing profile data?

Rules:
- soft_conflict: values are inconsistent but possible (lifestyle changes, errors)
- hard_conflict: mathematically or logically impossible
- no_conflict: no contradiction

Respond in JSON:
{{
  "conflict_type": "no_conflict | soft_conflict | hard_conflict",
  "conflicting_field": "<field_name or null>",
  "explanation": "<brief explanation or null>",
  "clarification_message": "<what to ask user, or null>"
}}
"""


@dataclass
class ConflictResult:
    conflict_type: str          # no_conflict | soft_conflict | hard_conflict
    conflicting_field: str | None = None
    explanation: str | None = None
    clarification_message: str | None = None

    @property
    def has_conflict(self) -> bool:
        return self.conflict_type != "no_conflict"


class ConflictDetector:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def check(self, field_name: str, new_value: FieldValue,
                    state: GraphState) -> ConflictResult:
        if len(state.profile) < 2:
            return ConflictResult(conflict_type="no_conflict")

        profile_text = "\n".join(
            f"  {k}: {v.value} (confidence: {v.confidence:.1f})"
            for k, v in state.profile.items()
            if k != field_name
        )

        prompt = PROMPT.format(
            profile=profile_text,
            field_name=field_name,
            new_value=new_value.value,
            raw_text=new_value.raw_text,
        )

        try:
            data, _ = await self.router.complete_json(
                task="conflict_detection", prompt=prompt,
                system=SYSTEM, temperature=0.1, max_tokens=250
            )
            return ConflictResult(
                conflict_type=data.get("conflict_type", "no_conflict"),
                conflicting_field=data.get("conflicting_field"),
                explanation=data.get("explanation"),
                clarification_message=data.get("clarification_message"),
            )
        except Exception:
            return ConflictResult(conflict_type="no_conflict")
