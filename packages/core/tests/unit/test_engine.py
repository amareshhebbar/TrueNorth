"""
tests/unit/test_engine.py

Unit tests for TrueNorthEngine and its pipeline components.
All tests use MockLLMClient — zero API calls, zero cost.

Run:
    make test-unit
    cd packages/core && PYTHONPATH=. pytest tests/unit/test_engine.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.core.engine        import TrueNorthEngine, EngineResponse
from truenorth.core.graph_state   import GraphState
from truenorth.intelligence.emotion_detector   import EmotionDetector, Emotion
from truenorth.intelligence.conflict_detector  import ConflictDetector, ConflictType
from truenorth.intelligence.confidence_scorer  import ConfidenceScorer
from truenorth.intelligence.language_detector  import LanguageDetector
from truenorth.intelligence.conversation_quality import ConversationQualityMonitor
from truenorth.llm.cost_tracker   import CostTracker, BudgetExceededError
from truenorth.llm.router         import LLMRouter
from truenorth.privacy.pii_detector import PIIDetector
from truenorth.testing.mock_llm   import MockLLMClient

MINIMAL_GOAL = {
    "id": "test_goal",
    "name": "Test Goal",
    "fields": [
        {"name": "name",   "type": "text",    "required": True,  "question": "What's your name?"},
        {"name": "age",    "type": "integer",  "required": True,  "question": "How old are you?"},
        {"name": "goal",   "type": "text",     "required": True,  "question": "What's your goal?"},
        {"name": "notes",  "type": "text",     "required": False, "question": "Any notes?"},
    ],
    "persona": {"name": "TestBot", "tone": "friendly"},
    "output": {"format": "text"},
}

FITNESS_YAML = Path(__file__).parent.parent / "examples" / "goals" / "fitness_plan.yaml"

def make_mock_router(extraction_json: str = '{"extractions": []}') -> LLMRouter:
    """Build a router backed by a mock LLM that returns predictable JSON."""
    mock = MockLLMClient(
        responses={
            "extract":   extraction_json,
            "classify":  '{"label": "neutral", "score": 0.6}',
        },
        default="Sounds good! What else can you tell me?",
    )
    router = LLMRouter()
    for model in ["gemini-3.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
        router.register_client(model, mock)
    return router

@pytest.fixture
def router():
    return make_mock_router()

@pytest.fixture
async def engine(router):
    return TrueNorthEngine(goal_config=MINIMAL_GOAL, router=router)

class TestEngineLifecycle:

    @pytest.mark.asyncio
    async def test_engine_creates_with_goal_config(self):
        e = TrueNorthEngine(goal_config=MINIMAL_GOAL)
        assert e.state.goal_id == "test_goal"
        assert len(e.state.fields_config) == 4
        assert e.state.session_id != ""

    @pytest.mark.asyncio
    async def test_engine_starts_and_returns_first_question(self, engine):
        resp = await engine.start()
        assert isinstance(resp, EngineResponse)
        assert resp.text != ""
        assert resp.turn == 0
        assert not resp.is_complete

    @pytest.mark.asyncio
    async def test_engine_from_yaml(self):
        if not FITNESS_YAML.exists():
            pytest.skip("fitness_plan.yaml not found")
        e = await TrueNorthEngine.from_yaml(str(FITNESS_YAML), router=make_mock_router())
        assert e.state.goal_id == "fitness_plan"
        assert len(e.state.fields_config) >= 9

    @pytest.mark.asyncio
    async def test_repr(self, engine):
        r = repr(engine)
        assert "TrueNorthEngine" in r
        assert "test_goal" in r

class TestMessageProcessing:

    @pytest.mark.asyncio
    async def test_process_message_returns_response(self, engine):
        await engine.start()
        resp = await engine.process_message("My name is Priya")
        assert isinstance(resp, EngineResponse)
        assert resp.text != ""
        assert resp.turn == 1

    @pytest.mark.asyncio
    async def test_turn_counter_increments(self, engine):
        await engine.start()
        for i in range(1, 4):
            resp = await engine.process_message(f"answer {i}")
            assert resp.turn == i

    @pytest.mark.asyncio
    async def test_user_message_added_to_history(self, engine):
        await engine.start()
        await engine.process_message("Hello there")
        user_msgs = engine.state.user_messages
        assert "Hello there" in user_msgs

    @pytest.mark.asyncio
    async def test_agent_message_added_to_history(self, engine):
        await engine.start()
        await engine.process_message("Hi")
        agent_msgs = engine.state.agent_messages
        assert len(agent_msgs) >= 2

class TestLanguageDetection:

    def test_detects_english(self):
        ld = LanguageDetector()
        r  = ld.detect("I want to lose weight and feel healthier")
        assert r.language_code == "en"
        assert r.is_indian is False

    def test_detects_hindi_devanagari(self):
        ld = LanguageDetector()
        r  = ld.detect("मैं थोड़ा वजन कम करना चाहता हूं")
        assert r.language_code == "hi"
        assert r.is_indian is True
        assert r.script == "Devanagari"

    def test_detects_hinglish_romanized(self):
        ld = LanguageDetector()
        r  = ld.detect("main gym jaana chahta hoon lekin time nahi hai")

        assert r.language_code in ("hi", "en")

    def test_detects_tamil_script(self):
        ld = LanguageDetector()
        r  = ld.detect("என்னுடைய வயது 28 ஆகும்")
        assert r.language_code == "ta"
        assert r.is_indian is True

    @pytest.mark.asyncio
    async def test_engine_updates_state_language(self, router):
        e = TrueNorthEngine(goal_config=MINIMAL_GOAL, router=router)
        await e.start()
        await e.process_message("My name is Alex and I am 30 years old")
        assert e.state.detected_language in ("en", "hi")

class TestEmotionDetection:

    @pytest.mark.asyncio
    async def test_detects_frustration(self):
        ed = EmotionDetector()
        r  = await ed.detect("This is annoying, I've already told you this twice!")
        assert r.label in (Emotion.FRUSTRATED, Emotion.ANGRY)
        assert r.is_negative is True

    @pytest.mark.asyncio
    async def test_detects_neutral(self):
        ed = EmotionDetector()
        r  = await ed.detect("I am 28 years old")
        assert r.label == Emotion.NEUTRAL

    @pytest.mark.asyncio
    async def test_detects_confusion(self):
        ed = EmotionDetector()
        r  = await ed.detect("Wait, what do you mean by activity level?")
        assert r.label in (Emotion.CONFUSED, Emotion.NEUTRAL)

    @pytest.mark.asyncio
    async def test_engine_stores_emotion_in_state(self, engine):
        await engine.start()
        await engine.process_message("This is great, I love working out!")
        assert engine.state.current_emotion is not None
        assert "label" in engine.state.current_emotion

class TestConflictDetection:

    def test_detects_numeric_conflict(self):
        cd = ConflictDetector()
        conflicts = cd.check(
            new_extractions = {"age": 35},
            collected       = {"age": 25},
            fields_config   = {"age": {"type": "integer"}},
            current_turn    = 5,
        )
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.NUMERIC_MISMATCH

    def test_detects_boolean_flip(self):
        cd = ConflictDetector()
        conflicts = cd.check(
            new_extractions = {"is_smoker": "yes"},
            collected       = {"is_smoker": "no"},
            fields_config   = {"is_smoker": {"type": "boolean"}},
            current_turn    = 3,
        )
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.BOOLEAN_FLIP

    def test_no_conflict_for_same_value(self):
        cd = ConflictDetector()
        conflicts = cd.check(
            new_extractions = {"name": "Priya"},
            collected       = {"name": "Priya"},
            fields_config   = {"name": {"type": "text"}},
            current_turn    = 2,
        )
        assert len(conflicts) == 0

    def test_no_conflict_for_new_field(self):
        cd = ConflictDetector()
        conflicts = cd.check(
            new_extractions = {"goal": "lose weight"},
            collected       = {},
            fields_config   = {"goal": {"type": "text"}},
            current_turn    = 1,
        )
        assert len(conflicts) == 0

class TestPIIDetection:

    def test_detects_email(self):
        pii = PIIDetector()
        r   = pii.scan("Contact me at priya@example.com please")
        assert r.has_pii
        assert any(m.type == "email" for m in r.matches)
        assert "priya@example.com" not in r.redacted

    def test_detects_indian_phone(self):
        pii = PIIDetector()
        r   = pii.scan("Call me on 9876543210")
        assert r.has_pii
        assert "9876543210" not in r.redacted
        assert r.has_high_risk

    def test_detects_aadhaar(self):
        pii = PIIDetector()
        r   = pii.scan("My Aadhaar is 1234 5678 9012")
        assert r.has_pii
        assert any(m.type == "aadhaar" for m in r.matches)

    def test_clean_text_has_no_pii(self):
        pii = PIIDetector()
        r   = pii.scan("I want to lose 5 kg in 3 months")
        assert not r.has_pii
        assert r.redacted == r.original

    @pytest.mark.asyncio
    async def test_engine_redacts_before_llm(self, router):
        e = TrueNorthEngine(goal_config=MINIMAL_GOAL, router=router)
        await e.start()
        await e.process_message("I am Priya, email priya@secret.com, age 28")
        history = e.state.turn_history

        assert any("Priya" in t.get("content", "") for t in history if t["role"] == "user")

class TestConfidenceScoring:

    def test_high_confidence_for_explicit_value(self):
        cs = ConfidenceScorer()
        score = cs.score(
            field                = "age",
            value                = 28,
            field_config         = {"type": "integer"},
            extraction_confidence = 0.95,
            source_text          = "I am 28 years old",
        )
        assert score.score >= 0.70

    def test_low_confidence_triggers_confirm_flag(self):
        cs = ConfidenceScorer()
        score = cs.score(
            field                = "age",
            value                = 28,
            field_config         = {"type": "integer"},
            extraction_confidence = 0.20,
            source_text          = "",
        )
        assert score.needs_confirm is True

    def test_confirmed_value_gets_bonus(self):
        cs = ConfidenceScorer()
        unconfirmed = cs.score("goal", "lose weight", extraction_confidence=0.7, user_confirmed=False)
        confirmed   = cs.score("goal", "lose weight", extraction_confidence=0.7, user_confirmed=True)
        assert confirmed.score > unconfirmed.score

class TestCostTracking:

    def test_records_call_cost(self):
        ct  = CostTracker()
        rec = ct.record("s1", "claude-haiku-4-5-20251001", "converse", 100, 50, 300)
        assert rec.cost_usd > 0
        assert rec.latency_ms == 300

    def test_accumulates_session_cost(self):
        ct = CostTracker()
        ct.record("s1", "gemini-3.5-flash", "extract",  200, 100, 100)
        ct.record("s1", "claude-haiku-4-5-20251001",    "converse", 150, 80, 200)
        cost = ct.get_session_cost("s1")
        assert cost.call_count == 2
        assert cost.total_cost_usd > 0

    def test_budget_cap_raises_error(self):
        ct = CostTracker()
        ct.set_budget("s2", 0.000001)
        ct.record("s2", "claude-haiku-4-5-20251001", "converse", 1000, 500, 0)
        with pytest.raises(BudgetExceededError):
            ct.check_budget("s2")

    def test_local_model_is_free(self):
        ct  = CostTracker()
        rec = ct.record("s3", "ollama", "converse", 1000, 1000, 50)
        assert rec.cost_usd == 0.0

class TestConversationQuality:

    def test_short_filler_lowers_clarity_and_flags(self):
        qm = ConversationQualityMonitor()
        r  = qm.check(turn_number=3, user_message="ok", turn_history=[], fields_collected=1, total_required_fields=8)

        assert r.clarity_score < 0.50
        assert "FILLER_ANSWER" in r.flags

    def test_frustration_detected_in_message(self):
        qm = ConversationQualityMonitor()
        r  = qm.check(turn_number=5, user_message="This is annoying, forget it", turn_history=[], fields_collected=2, total_required_fields=8)
        assert r.frustration_signal > 0.40

    def test_healthy_message_passes(self):
        qm = ConversationQualityMonitor()
        r  = qm.check(turn_number=2, user_message="I work out 3 times a week at the gym", turn_history=[], fields_collected=3, total_required_fields=8)
        assert r.is_healthy

class TestGraphState:

    def test_serialise_deserialise(self):
        state = GraphState.from_goal_config(MINIMAL_GOAL, session_id="test-123")
        state.set_field("name", "Priya", confidence=0.95)
        state.set_field("age", 28, confidence=0.90)
        restored = GraphState.from_dict(state.to_dict())
        assert restored.collected_fields["name"] == "Priya"
        assert restored.collected_fields["age"] == 28
        assert restored.field_confidences["name"] == pytest.approx(0.95)

    def test_missing_required_tracking(self):
        state = GraphState.from_goal_config(MINIMAL_GOAL, session_id="s")
        assert "name" in state.missing_required
        state.set_field("name", "Alex")
        assert "name" not in state.missing_required

    def test_completion_percentage(self):
        state = GraphState.from_goal_config(MINIMAL_GOAL, session_id="s")
        assert state.completion_pct == 0.0
        state.set_field("name", "Alex")
        assert state.completion_pct == pytest.approx(1/3, abs=0.01)
        state.set_field("age", 28)
        state.set_field("goal", "fitness")
        assert state.completion_pct == pytest.approx(1.0)

    def test_add_turn_increments_history(self):
        state = GraphState.from_goal_config(MINIMAL_GOAL, session_id="s")
        state.add_turn("user", "Hello")
        state.add_turn("assistant", "Hi!")
        assert len(state.turn_history) == 2
        assert state.user_messages == ["Hello"]
        assert state.agent_messages == ["Hi!"]
