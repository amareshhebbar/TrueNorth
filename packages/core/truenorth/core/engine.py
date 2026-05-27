"""
TrueNorthEngine — the central orchestrator.

Processes one conversation turn:
1. Extract field values from user message
2. Detect emotion state
3. Check for conflicts in collected data
4. Update state + missing fields
5. Plan next question
6. Return response
"""

from __future__ import annotations
import copy
from datetime import datetime
from truenorth.core.graph_state import GraphState, FieldValue
from truenorth.core.yaml_loader import GoalConfig, YamlLoader
from truenorth.core.field_extractor import FieldExtractor
from truenorth.core.conversation_planner import ConversationPlanner
from truenorth.intelligence.emotion_detector import EmotionDetector
from truenorth.intelligence.conflict_detector import ConflictDetector
from truenorth.llm.router import LLMRouter
from truenorth.output.generator import OutputGenerator
from truenorth.privacy.pii_detector import PIIDetector


class TrueNorthEngine:
    """
    The main engine. One instance per session.
    All processing is async and stateless — pass state in, get new state out.
    """

    def __init__(self, config: GoalConfig, router: LLMRouter):
        self.config = config
        self.router = router
        self.extractor = FieldExtractor(router)
        self.planner = ConversationPlanner(router)
        self.emotion = EmotionDetector(router)
        self.conflict = ConflictDetector(router)
        self.output_gen = OutputGenerator(router)
        self.pii = PIIDetector()

    @classmethod
    def from_yaml(cls, source, router: LLMRouter | None = None) -> "TrueNorthEngine":
        config = YamlLoader().load(source)
        if router is None:
            router = LLMRouter()
        return cls(config, router)

    def create_initial_state(self, session_id: str, user_id: str | None = None) -> GraphState:
        """Create a fresh GraphState for a new session."""
        state = GraphState(
            session_id=session_id,
            goal_id=self.config.goal_id,
            config=self.config.model_dump(),
            user_id=user_id,
        )
        state.missing_required = [f.name for f in self.config.required_fields]
        state.missing_optional = [f.name for f in self.config.optional_fields]
        return state

    async def process_turn(self, state: GraphState, user_message: str) -> tuple[GraphState, str]:
        """
        Process one user message. Returns (new_state, agent_response).
        Core pipeline: extract → emotion → conflict → plan → respond.
        """
        # 1. Scan for PII in message
        self.pii.scan(user_message)  # logs warnings, doesn't block

        # 2. Add user turn to history
        state = state.add_turn("user", user_message)

        # 3. Extract field values from message
        extracted = await self.extractor.extract(user_message, state, self.config)

        # 4. Detect emotion state
        history_dicts = [{"role": t.role, "content": t.content} for t in state.conversation]
        emotion_result = await self.emotion.detect(user_message, history_dicts)

        # 5. Update state with emotion
        new_state = copy.deepcopy(state)
        new_state.emotion_state = emotion_result.state
        new_state.emotion_confidence = emotion_result.confidence

        # 6. Process extracted fields (conflict check + set)
        for field_name, field_value in extracted.items():
            # Check for conflicts with existing data
            if field_name in new_state.profile:
                conflict = await self.conflict.check(field_name, field_value, new_state)
                if conflict.has_conflict and conflict.clarification_message:
                    # Ask for clarification instead of setting
                    new_state = new_state.add_turn("assistant", conflict.clarification_message)
                    return new_state, conflict.clarification_message

            new_state = new_state.set_field(field_name, field_value)

        # 7. Update cost tracking
        new_state.cost_usd = self.router.session_cost
        new_state.tokens_used = self.router.session_tokens

        # 8. Check if emotion says skip optional fields
        if emotion_result.skip_optional:
            new_state.missing_optional = []

        # 9. Plan the next agent response
        agent_message, next_field = await self.planner.plan_next_turn(new_state, self.config)

        if next_field == "__escalate__":
            new_state.escalated = True
            new_state.escalation_reason = f"emotion:{new_state.emotion_state}"
        elif next_field is None and new_state.is_ready_for_output:
            # Generate output
            output = await self.output_gen.generate(new_state, self.config)
            new_state.output = output
            new_state.completed = True
        else:
            new_state.current_field_target = next_field

        # 10. Add agent turn to history
        new_state = new_state.add_turn("assistant", agent_message)

        return new_state, agent_message

    async def generate_welcome_message(self, state: GraphState) -> str:
        """Generate the opening message for a new session."""
        first_field = self.config.required_fields[0].name if self.config.required_fields else None
        system = f"You are {getattr(self.config.persona, 'base', 'a helpful assistant')}."
        prompt = f"""
Generate a warm, brief welcome message to start collecting information.
First field you'll ask about: {first_field or 'general information'}
Keep it to 1-2 sentences. Be friendly and natural.
Respond ONLY with the message text.
"""
        resp = await self.router.complete(
            task="conversation", prompt=prompt, system=system,
            temperature=0.8, max_tokens=100
        )
        return resp.content.strip()

    async def generate_resume_message(self, state: GraphState) -> str:
        """Generate a resume message when user returns to an existing session."""
        collected = ", ".join(
            f"{k}: {v.value}" for k, v in list(state.profile.items())[:5]
        )
        missing = ", ".join(state.missing_required[:3])

        prompt = f"""
The user is returning to a session where we already collected: {collected}
We still need: {missing}

Generate a warm 2-3 sentence message:
1. Welcome them back
2. Briefly summarize what they told us
3. Ask to continue

Be natural and friendly. Respond ONLY with the message.
"""
        resp = await self.router.complete(
            task="conversation", prompt=prompt,
            temperature=0.7, max_tokens=150
        )
        return resp.content.strip()
