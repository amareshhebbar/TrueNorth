"""
Decide what the agent should say next.
Picks the next field to ask about, generates a natural question.
Adapts based on emotion state, conversation quality, and config.
"""

from __future__ import annotations
from truenorth.core.graph_state import GraphState
from truenorth.core.yaml_loader import GoalConfig, FieldConfig
from truenorth.llm.router import LLMRouter

SYSTEM_TEMPLATE = """You are {persona}.

Your job: collect specific information from the user through natural conversation.
Rules:
- Ask ONE question at a time (maximum 2 if very related)
- Be conversational, not clinical
- Never say "field" or "data" — just ask naturally
- Adapt your tone to the user's emotional state
- If user seems frustrated: shorter, kinder questions
- If all required data is collected: say you have everything and are generating their plan
- Language: respond in the same language the user is writing in"""

PLANNING_PROMPT = """
Collected so far:
{collected}

Still need (required):
{missing_required}

Still need (optional, ask if conversation is going well):
{missing_optional}

User's current emotion: {emotion} (confidence: {emotion_confidence:.0%})
Conversation quality score: {quality:.0%}

Last user message: "{last_message}"

Conversation history (last 3 turns):
{history}

Instructions:
1. Decide the NEXT field to ask about (from missing_required first)
2. Generate a natural, conversational question to collect it
3. If emotion is frustrated/anxious: acknowledge their feeling first (1 sentence)
4. If all required fields are collected: return a completion message

Respond in JSON:
{{
  "next_field": "<field_name or null if complete>",
  "message": "<your response to the user>",
  "is_complete": false,
  "reasoning": "<brief>"
}}
"""


class ConversationPlanner:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def plan_next_turn(self, state: GraphState,
                              config: GoalConfig) -> tuple[str, str | None]:
        """
        Returns (agent_message, next_field_name_or_None).
        """
        if state.is_ready_for_output:
            return await self._generate_completion_message(state, config), None

        # Check escalation triggers
        if self._should_escalate(state, config):
            return await self._generate_escalation_message(config), "__escalate__"

        history = "\n".join(
            f"{t.role.upper()}: {t.content}"
            for t in state.conversation[-3:]
        )

        last_message = ""
        if state.conversation:
            user_turns = [t for t in state.conversation if t.role == "user"]
            if user_turns:
                last_message = user_turns[-1].content

        persona = self._get_persona(config)
        system = SYSTEM_TEMPLATE.format(persona=persona)

        collected_text = "\n".join(
            f"  {k}: {v.value} (confidence: {v.confidence:.0%})"
            for k, v in state.profile.items()
        ) or "  (nothing yet)"

        prompt = PLANNING_PROMPT.format(
            collected=collected_text,
            missing_required=", ".join(state.missing_required) or "none",
            missing_optional=", ".join(state.missing_optional) or "none",
            emotion=state.emotion_state,
            emotion_confidence=state.emotion_confidence,
            quality=state.conversation_quality_score,
            last_message=last_message,
            history=history or "(first turn)",
        )

        try:
            data, _ = await self.router.complete_json(
                task="conversation", prompt=prompt, system=system,
                temperature=0.7, max_tokens=400
            )
            message = data.get("message", "Could you tell me more?")
            next_field = data.get("next_field")

            if data.get("is_complete"):
                return message, None

            return message, next_field
        except Exception as e:
            return "Could you tell me a bit more about yourself?", state.missing_required[0] if state.missing_required else None

    def _should_escalate(self, state: GraphState, config: GoalConfig) -> bool:
        triggers = config.escalation.get("triggers", [])
        for trigger in triggers:
            condition = trigger.get("condition", "")
            if "distressed" in condition and state.emotion_state == "distressed":
                return True
            if "consecutive_confusion" in condition:
                confusion_turns = sum(
                    1 for t in state.conversation[-6:]
                    if t.role == "user" and len(t.content) < 10
                )
                threshold = int(condition.split(">=")[-1].strip()) if ">=" in condition else 3
                if confusion_turns >= threshold:
                    return True
        return False

    def _get_persona(self, config: GoalConfig) -> str:
        return getattr(config.persona, 'base', 'a helpful assistant')

    async def _generate_completion_message(self, state: GraphState,
                                            config: GoalConfig) -> str:
        system = SYSTEM_TEMPLATE.format(persona=self._get_persona(config))
        profile_summary = ", ".join(
            f"{k}={v.value}" for k, v in list(state.profile.items())[:6]
        )
        prompt = f"""
The user has provided all required information: {profile_summary}
Generate a brief, warm message telling them you have everything you need
and are now generating their personalized plan. Be encouraging. 1-2 sentences.
Respond ONLY with the message text, no JSON.
"""
        resp = await self.router.complete(
            task="conversation", prompt=prompt, system=system,
            temperature=0.7, max_tokens=150
        )
        return resp.content.strip()

    async def _generate_escalation_message(self, config: GoalConfig) -> str:
        triggers = config.escalation.get("triggers", [])
        for t in triggers:
            if "distressed" in t.get("condition", ""):
                return t.get("escalation_message",
                    "I want to make sure you get the right support. "
                    "Let me connect you with someone who can help.")
        return "Let me connect you with a human advisor who can better assist you."
