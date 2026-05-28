"""
truenorth/core/conversation_planner.py

Given a Reasoner decision, generates the actual agent response text.
This is the only module that writes the words the user reads.

Responsibilities:
  - Phrase questions naturally (not robotic)
  - Adapt tone to detected emotion (empathetic when frustrated)
  - Adapt language to detected locale (respond in Hindi if user writes Hindi)
  - Handle conflict resolution requests gracefully
  - Acknowledge context before asking next question
  - Keep responses to ONE question at a time (strict)
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from truenorth.core.reasoner import ReasonerAction, ReasonerDecision
from truenorth.core.graph_state import GraphState

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConversationPlanner
# ---------------------------------------------------------------------------

class ConversationPlanner:
    """
    Generates the next agent response given a ReasonerDecision.

    Usage:
        planner = ConversationPlanner(router=llm_router)
        response = await planner.plan(decision=decision, state=state)
    """

    _SYSTEM_TEMPLATE = """\
You are {name}, a {tone} conversational AI assistant.
Your goal: collect information from the user through natural conversation.

Rules you MUST follow:
1. Ask EXACTLY ONE question per response. Never two.
2. Keep responses SHORT — 1-3 sentences maximum.
3. Acknowledge what the user just said before asking the next question (when relevant).
4. Never use jargon. Speak like a real person.
5. If the user seems frustrated, acknowledge it and offer to help differently.
6. Respond in the same language the user is writing in.
7. Do not prefix with "Sure!" or "Great!" every time — vary your phrasing.
"""

    def __init__(self, router: Optional["LLMRouter"] = None):
        self._router = router

    async def plan(
        self,
        decision: ReasonerDecision,
        state:    GraphState,
    ) -> str:
        """
        Generate the agent's next response.

        Args:
            decision: What the Reasoner decided to do next
            state:    Current session state

        Returns:
            String — the agent's response to send to the user
        """
        action = decision.action

        # Route to the appropriate response generator
        if action == ReasonerAction.ASK_FIELD:
            return await self._ask_field(decision, state)

        if action == ReasonerAction.ASK_OPTIONAL:
            return await self._ask_field(decision, state, is_optional=True)

        if action == ReasonerAction.CLARIFY:
            return await self._clarify(decision, state)

        if action == ReasonerAction.RESOLVE_CONFLICT:
            return self._resolve_conflict(decision, state)

        if action == ReasonerAction.HANDLE_EMOTION:
            return await self._handle_emotion(state)

        if action == ReasonerAction.GENERATE_OUTPUT:
            return "I have all the information I need. Let me prepare your report..."

        if action == ReasonerAction.BUDGET_EXCEEDED:
            return (
                "I'm sorry, but we've reached the session limit. "
                "I'll save your progress — you can resume later with the session ID."
            )

        if action == ReasonerAction.END:
            return "Thank you for completing this. Your information has been recorded."

        # Fallback
        return await self._ask_field(decision, state)

    # ------------------------------------------------------------------
    # Response generators
    # ------------------------------------------------------------------

    async def _ask_field(
        self,
        decision:    ReasonerDecision,
        state:       GraphState,
        is_optional: bool = False,
    ) -> str:
        field_name = decision.target_field
        if not field_name:
            return "Can you tell me a bit more?"

        field_cfg  = state.fields_config.get(field_name, {})
        question   = field_cfg.get("question") or self._default_question(field_name, field_cfg)
        follow_up  = field_cfg.get("follow_up_hint", "")

        if self._router is None or not state.turn_history:
            return question

        return await self._llm_rephrase(
            question    = question,
            state       = state,
            is_optional = is_optional,
            follow_up   = follow_up,
        )

    async def _clarify(self, decision: ReasonerDecision, state: GraphState) -> str:
        field_name  = decision.target_field or ""
        field_cfg   = state.fields_config.get(field_name, {})
        last_val    = decision.metadata.get("last_extraction", {}).get("value", "")
        question    = field_cfg.get("question", f"What is your {field_name.replace('_',' ')}?")

        if self._router is None:
            return f"I didn't quite get that — {question}"

        prompt = (
            f"The user gave an unclear answer about '{field_name}'. "
            f"The extracted value was {last_val!r}, but confidence was low. "
            f"Write a gentle one-sentence clarification request that restates the question: "
            f"'{question}'. Do NOT ask anything else."
        )
        return await self._llm_short(prompt, state)

    def _resolve_conflict(self, decision: ReasonerDecision, state: GraphState) -> str:
        conflict = decision.metadata.get("conflict", {})
        field    = conflict.get("field", "")
        old_val  = conflict.get("old_value", "")
        new_val  = conflict.get("new_value", "")
        label    = state.fields_config.get(field, {}).get("label", field.replace("_", " "))

        return (
            f"I noticed a discrepancy — earlier you mentioned {label} as {old_val!r}, "
            f"but now it seems like {new_val!r}. Which one is correct?"
        )

    async def _handle_emotion(self, state: GraphState) -> str:
        emotion = state.current_emotion or {}
        label   = emotion.get("label", "frustrated")

        if self._router is None:
            acks = {
                "frustrated": "I understand this can feel tedious. Let me know if you'd like to take a break.",
                "distressed":  "I can hear that you're going through a lot. Take your time.",
                "confused":    "No worries — let me explain that differently.",
                "anxious":     "It's okay to take your time. There's no rush here.",
            }
            return acks.get(label, "I understand. Let me know how I can help better.")

        prompt = (
            f"The user appears {label}. Write a short (1-2 sentence) empathetic response "
            f"that acknowledges their feelings and gently offers to continue or take a break. "
            f"Do NOT ask a question. Do NOT be overly effusive."
        )
        return await self._llm_short(prompt, state)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _llm_rephrase(
        self,
        question:    str,
        state:       GraphState,
        is_optional: bool = False,
        follow_up:   str = "",
    ) -> str:
        """Ask LLM to rephrase a YAML question into natural conversational language."""
        from truenorth.llm.base import Message as LLMMessage
        from truenorth.llm.router import TASK_CONVERSE

        persona = state.persona
        system  = self._SYSTEM_TEMPLATE.format(
            name = persona.get("name", "TrueNorth"),
            tone = persona.get("tone", "friendly"),
        )

        last_user_msg = state.user_messages[-1] if state.user_messages else ""
        recent_exchange = ""
        if state.turn_history:
            recent = state.turn_history[-4:]
            recent_exchange = "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in recent
            )

        optional_note = " (this question is optional — the user can skip it)" if is_optional else ""
        follow_note   = f" Hint: {follow_up}" if follow_up else ""

        prompt = (
            f"Recent conversation:\n{recent_exchange}\n\n"
            f"Now ask this question in a natural, conversational way{optional_note}:\n"
            f"{question!r}{follow_note}\n\n"
            f"Write ONLY the response. Do not explain what you're doing. "
            f"Language: {state.detected_language}."
        )

        try:
            resp = await self._router.generate(
                task      = TASK_CONVERSE,
                messages  = [LLMMessage(role="user", content=prompt)],
                system    = system,
                max_tokens = 120,
                temperature = 0.8,
            )
            return resp.content.strip()
        except Exception as e:
            logger.warning("conversation_planner LLM failed: %s — using raw question", e)
            return question

    async def _llm_short(self, prompt: str, state: GraphState) -> str:
        """Send a short prompt, return the LLM response."""
        from truenorth.llm.base import Message as LLMMessage
        from truenorth.llm.router import TASK_CONVERSE

        persona = state.persona
        system  = self._SYSTEM_TEMPLATE.format(
            name = persona.get("name", "TrueNorth"),
            tone = persona.get("tone", "friendly"),
        )
        try:
            resp = await self._router.generate(
                task       = TASK_CONVERSE,
                messages   = [LLMMessage(role="user", content=prompt)],
                system     = system,
                max_tokens  = 80,
                temperature = 0.7,
            )
            return resp.content.strip()
        except Exception as e:
            logger.warning("conversation_planner short LLM failed: %s", e)
            return "Could you clarify that for me?"

    @staticmethod
    def _default_question(field_name: str, field_cfg: dict) -> str:
        label = field_cfg.get("label", field_name.replace("_", " ").title())
        ftype = field_cfg.get("type", "text")

        if ftype == "boolean":
            return f"Would you say {label}? (yes or no)"
        if ftype in ("integer", "number"):
            return f"What is your {label}?"
        return f"Can you tell me your {label}?"