"""
No network calls. No APScheduler. No real SMTP/WhatsApp/Twilio.
All external calls mocked.

Classes:
  1.  FollowUpRule         — YAML parsing, trigger parsing, condition_met
  2.  ScheduledReminder    — is_due, to_dict, status transitions
  3.  ReminderEngine_Schedule — schedule_all, conditions, cancel
  4.  ReminderEngine_Due   — get_due, mark_delivered, stats
  5.  ReminderEngine_Loop  — background _tick (mock delivery)
  6.  DeliveryResult       — success/failure structure
  7.  MultiChannelDelivery — routes to correct adapter
  8.  ConsoleAdapter       — always succeeds, prints
  9.  LongTermMemory_Store — store, confidence threshold
  10. LongTermMemory_Read  — get, get_all, get_by_goal
  11. LongTermMemory_Seed  — seed_engine, forget/GDPR
  12. SessionResume        — check resumable, re-engagement message
  13. VectorStore          — add, search, cosine similarity
  14. DPDPManager          — consent, rights, audit log
  15. GDPRManager          — consent, legal basis, subject rights
  16. WhatsAppMessage      — webhook parsing, normalisation
  17. WhatsAppChannel      — session management, send (mocked)
  18. SectorCompliance     — DPDP + GDPR across 5 sectors
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.scheduler.reminder_engine import (
    FollowUpRule, ScheduledReminder, ReminderEngine,
    ReminderStatus,
)
from truenorth.scheduler.delivery import (
    DeliveryResult, DeliveryChannel, ConsoleAdapter, MultiChannelDelivery,
)
from truenorth.memory.long_term      import LongTermMemory, UserFact

from truenorth.compliance.dpdp import (
    DPDPManager, ConsentStatus, DataPrincipalRight,
)

from truenorth.memory.session_resume import SessionResume
from truenorth.memory.vector_store import VectorStore
from truenorth.compliance.gdpr import GDPRManager, GDPRLegalBasis, DataSubjectRight
from truenorth.channel.whatsapp import WhatsAppMessage, WhatsAppChannel

NOW_UTC = datetime.now(timezone.utc)

def _rule(
    rule_id:   str   = "r1",
    trigger:   str   = "after 1 day",
    channel:   str   = "email",
    prompt:    str   = "Check in with user.",
    check_field: Optional[str] = None,
    condition: Optional[dict]  = None,
) -> FollowUpRule:
    return FollowUpRule(
        rule_id        = rule_id,
        trigger        = trigger,
        channel        = channel,
        message_prompt = prompt,
        check_field    = check_field,
        condition      = condition,
    )

def _engine_with_console() -> ReminderEngine:
    delivery = MultiChannelDelivery({"email": ConsoleAdapter(), "whatsapp": ConsoleAdapter()})
    return ReminderEngine(delivery=delivery)

class TestFollowUpRule:

    def test_from_yaml_basic(self):
        raw  = {"trigger": "after 2 days", "channel": "whatsapp", "message_prompt": "Check in"}
        rule = FollowUpRule.from_yaml(raw, 0)
        assert rule.trigger  == "after 2 days"
        assert rule.channel  == "whatsapp"
        assert rule.rule_id  == "rule_0"

    def test_from_yaml_with_id(self):
        raw  = {"id": "my_rule", "trigger": "after 3 days", "channel": "email", "message_prompt": ""}
        rule = FollowUpRule.from_yaml(raw)
        assert rule.rule_id == "my_rule"

    def test_parse_trigger_after_days(self):
        rule    = _rule(trigger="after 3 days")
        base    = NOW_UTC
        fire_at = rule.parse_trigger(base)
        assert fire_at is not None
        diff = (fire_at - base).total_seconds()
        assert 3 * 86400 - 10 < diff < 3 * 86400 + 10

    def test_parse_trigger_after_hours(self):
        rule    = _rule(trigger="after 6 hours")
        fire_at = rule.parse_trigger(NOW_UTC)
        diff    = (fire_at - NOW_UTC).total_seconds()
        assert 6 * 3600 - 5 < diff < 6 * 3600 + 5

    def test_parse_trigger_after_weeks(self):
        rule    = _rule(trigger="after 2 weeks")
        fire_at = rule.parse_trigger(NOW_UTC)
        diff    = (fire_at - NOW_UTC).total_seconds()
        assert 14 * 86400 - 10 < diff < 14 * 86400 + 10

    def test_parse_trigger_weekly(self):
        rule    = _rule(trigger="weekly")
        fire_at = rule.parse_trigger(NOW_UTC)
        diff    = (fire_at - NOW_UTC).total_seconds()
        assert 7 * 86400 - 5 < diff < 7 * 86400 + 5

    def test_parse_trigger_monthly(self):
        rule    = _rule(trigger="monthly")
        fire_at = rule.parse_trigger(NOW_UTC)
        diff    = (fire_at - NOW_UTC).total_seconds()
        assert 29 * 86400 < diff < 31 * 86400

    def test_parse_trigger_on_date(self):
        rule    = _rule(trigger="on 2030-01-15")
        fire_at = rule.parse_trigger(NOW_UTC)
        assert fire_at is not None
        assert fire_at.year == 2030
        assert fire_at.month == 1

    def test_parse_trigger_invalid_returns_none(self):
        rule    = _rule(trigger="gobbledygook")
        fire_at = rule.parse_trigger(NOW_UTC)
        assert fire_at is None

    def test_condition_met_no_condition(self):
        rule = _rule()
        assert rule.condition_met({"anything": "here"}) is True
        assert rule.condition_met({}) is True

    def test_condition_met_check_field_absent(self):
        rule = FollowUpRule(
            rule_id="r", trigger="after 1 day", channel="email",
            message_prompt="", check_field="exercise_done", check_value=None,
        )
        assert rule.condition_met({})                            is True
        assert rule.condition_met({"exercise_done": "yes"})     is False

    def test_condition_met_check_value_matches(self):
        rule = FollowUpRule(
            rule_id="r", trigger="after 1 day", channel="email",
            message_prompt="", check_field="goal", check_value="lose weight",
        )
        assert rule.condition_met({"goal": "lose weight"})  is True
        assert rule.condition_met({"goal": "build muscle"}) is False

    def test_condition_met_condition_dict(self):
        rule = _rule(condition={"primary_goal": "lose weight"})
        assert rule.condition_met({"primary_goal": "lose weight"}) is True
        assert rule.condition_met({"primary_goal": "other"})       is False

class TestScheduledReminder:

    def _reminder(self, fire_at: datetime, status: ReminderStatus = ReminderStatus.PENDING):
        return ScheduledReminder(
            reminder_id = "rem-1",
            rule_id     = "r1",
            session_id  = "s1",
            user_id     = "u1",
            goal_id     = "fitness_plan",
            channel     = "email",
            fire_at     = fire_at,
            status      = status,
        )

    def test_is_due_past(self):
        past = NOW_UTC - timedelta(hours=1)
        r    = self._reminder(past)
        assert r.is_due() is True

    def test_is_due_future(self):
        future = NOW_UTC + timedelta(hours=1)
        r      = self._reminder(future)
        assert r.is_due() is False

    def test_not_due_if_not_pending(self):
        past = NOW_UTC - timedelta(hours=1)
        r    = self._reminder(past, ReminderStatus.DELIVERED)
        assert r.is_due() is False

    def test_to_dict_has_required_keys(self):
        r = self._reminder(NOW_UTC)
        d = r.to_dict()
        for k in ["reminder_id", "rule_id", "session_id", "channel", "fire_at", "status"]:
            assert k in d

class TestReminderEngineSchedule:

    def test_schedule_all_creates_reminders(self):
        engine = _engine_with_console()
        rules  = [_rule("r1", "after 1 day"), _rule("r2", "after 3 days")]
        result = engine.schedule_all(
            rules=rules, session_id="s1", user_id="u1",
            goal_id="fitness_plan", collected_fields={},
        )
        assert len(result) == 2

    def test_schedule_respects_condition(self):
        engine = _engine_with_console()
        rules  = [
            _rule("r1", condition={"goal": "lose weight"}),
            _rule("r2"),
        ]
        result = engine.schedule_all(
            rules=rules, session_id="s1", user_id="u1",
            goal_id="g", collected_fields={"goal": "build muscle"},
        )
        rule_ids = [r.rule_id for r in result]
        assert "r2" in rule_ids
        assert "r1" not in rule_ids

    def test_schedule_all_with_check_field(self):
        engine = _engine_with_console()
        rule = FollowUpRule(
            rule_id="chk", trigger="after 2 days", channel="email",
            message_prompt="", check_field="exercise_done", check_value=None,
        )
        result = engine.schedule_all(
            rules=[rule], session_id="s1", user_id="u1",
            goal_id="g", collected_fields={},
        )
        assert len(result) == 1

        result2 = engine.schedule_all(
            rules=[rule], session_id="s2", user_id="u1",
            goal_id="g", collected_fields={"exercise_done": "yes"},
        )
        assert len(result2) == 0

    def test_cancel_reminder(self):
        engine = _engine_with_console()
        [rem]  = engine.schedule_all(
            rules=[_rule()], session_id="s1", user_id="u1",
            goal_id="g", collected_fields={},
        )
        assert engine.cancel(rem.reminder_id) is True
        assert engine._reminders[rem.reminder_id].status == ReminderStatus.CANCELLED

    def test_cancel_all_for_session(self):
        engine = _engine_with_console()
        engine.schedule_all(
            rules=[_rule("r1"), _rule("r2")], session_id="s1",
            user_id="u1", goal_id="g", collected_fields={},
        )
        engine.schedule_all(
            rules=[_rule("r3")], session_id="s2",
            user_id="u2", goal_id="g", collected_fields={},
        )
        count = engine.cancel_all_for_session("s1")
        assert count == 2
        s2_rems = engine.get_pending("s2")
        assert len(s2_rems) == 1

class TestReminderEngineDue:

    def _past_reminder(self, engine: ReminderEngine) -> ScheduledReminder:
        rule  = _rule()
        past  = NOW_UTC - timedelta(seconds=5)
        return engine.schedule_one(rule, "s1", "u1", "g", fire_at=past)

    def test_get_due_returns_past_reminders(self):
        engine = _engine_with_console()
        self._past_reminder(engine)
        due = engine.get_due()
        assert len(due) == 1

    def test_get_due_excludes_future(self):
        engine = _engine_with_console()
        rule   = _rule()
        future = NOW_UTC + timedelta(days=1)
        engine.schedule_one(rule, "s1", "u1", "g", fire_at=future)
        assert engine.get_due() == []

    def test_mark_triggered(self):
        engine = _engine_with_console()
        rem    = self._past_reminder(engine)
        engine.mark_triggered(rem.reminder_id)
        assert engine._reminders[rem.reminder_id].status == ReminderStatus.TRIGGERED
        assert engine._reminders[rem.reminder_id].fire_count == 1

    def test_mark_delivered(self):
        engine = _engine_with_console()
        rem    = self._past_reminder(engine)
        engine.mark_triggered(rem.reminder_id)
        engine.mark_delivered(rem.reminder_id, {"message_id": "msg-1"}, "Hello!")
        r = engine._reminders[rem.reminder_id]
        assert r.status       == ReminderStatus.DELIVERED
        assert r.message_text == "Hello!"

    def test_mark_failed(self):
        engine = _engine_with_console()
        rem    = self._past_reminder(engine)
        engine.mark_failed(rem.reminder_id, "connection timeout")
        assert engine._reminders[rem.reminder_id].status == ReminderStatus.FAILED

    def test_get_pending_filter(self):
        engine = _engine_with_console()
        self._past_reminder(engine)
        rule2 = _rule("r2")
        engine.schedule_one(rule2, "s2", "u2", "g", fire_at=NOW_UTC - timedelta(seconds=1))
        assert len(engine.get_pending("s1")) == 1
        assert len(engine.get_pending("s2")) == 1

    def test_stats(self):
        engine = _engine_with_console()
        self._past_reminder(engine)
        s = engine.stats()
        assert s["total"] == 1
        assert "pending" in s["by_status"]

class TestReminderEngineLoop:

    @pytest.mark.asyncio
    async def test_tick_delivers_due_reminder(self):
        delivered = []

        class _MockDelivery:
            async def send(self, reminder):
                delivered.append(reminder.reminder_id)
                return DeliveryResult(success=True, channel="email")

        class _MockPlanner:
            async def compose(self, reminder):
                return "Test follow-up message"

        engine = ReminderEngine(delivery=_MockDelivery(), planner=_MockPlanner())
        rule   = _rule()
        past   = NOW_UTC - timedelta(seconds=1)
        rem    = engine.schedule_one(rule, "s1", "u1", "g", fire_at=past)
        await engine._tick()

        assert rem.reminder_id in delivered
        assert engine._reminders[rem.reminder_id].status == ReminderStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_tick_handles_delivery_failure(self):
        class _FailDelivery:
            async def send(self, reminder):
                raise RuntimeError("network error")

        engine = ReminderEngine(delivery=_FailDelivery(), planner=AsyncMock())
        rule   = _rule()
        rem    = engine.schedule_one(rule, "s1", "u1", "g",
                                     fire_at=NOW_UTC - timedelta(seconds=1))
        await engine._tick()
        assert engine._reminders[rem.reminder_id].status == ReminderStatus.FAILED

class TestDeliveryResult:

    def test_success_result(self):
        r = DeliveryResult(success=True, channel="email", message_id="msg-1", latency_ms=120)
        assert r.success
        assert r.message_id == "msg-1"

    def test_failure_result(self):
        r = DeliveryResult(success=False, channel="whatsapp", error="timeout")
        assert not r.success
        assert r.error == "timeout"

    def test_to_dict_has_required_keys(self):
        d = DeliveryResult(success=True, channel="sms").to_dict()
        for k in ["success", "channel", "message_id", "error", "latency_ms"]:
            assert k in d

class TestMultiChannelDelivery:

    def _reminder_for(self, channel: str) -> ScheduledReminder:
        return ScheduledReminder(
            reminder_id="rem", rule_id="r", session_id="s",
            user_id="u", goal_id="g", channel=channel,
            fire_at=NOW_UTC,
        )

    @pytest.mark.asyncio
    async def test_routes_to_correct_adapter(self):
        results = []
        class _TrackingAdapter(DeliveryChannel):
            def __init__(self, name):
                self._name = name
            async def send(self, reminder):
                results.append(self._name)
                return DeliveryResult(success=True, channel=self._name)

        delivery = MultiChannelDelivery({
            "email":     _TrackingAdapter("email"),
            "whatsapp":  _TrackingAdapter("whatsapp"),
        })
        await delivery.send(self._reminder_for("email"))
        await delivery.send(self._reminder_for("whatsapp"))
        assert results == ["email", "whatsapp"]

    @pytest.mark.asyncio
    async def test_fallback_on_unknown_channel(self):
        delivery = MultiChannelDelivery(
            {"console": ConsoleAdapter()},
            fallback_channel="console",
        )
        result = await delivery.send(self._reminder_for("telegram"))
        assert result.success

    @pytest.mark.asyncio
    async def test_no_fallback_no_adapter_fails(self):
        delivery = MultiChannelDelivery({}, fallback_channel=None)
        result   = await delivery.send(self._reminder_for("email"))
        assert not result.success

class TestConsoleAdapter:

    @pytest.mark.asyncio
    async def test_always_succeeds(self):
        adapter = ConsoleAdapter()
        rem     = ScheduledReminder(
            reminder_id="r1", rule_id="rule", session_id="s",
            user_id="u", goal_id="g", channel="console",
            fire_at=NOW_UTC, message_text="Hello!",
        )
        result = await adapter.send(rem)
        assert result.success
        assert result.channel == "console"

class TestLongTermMemoryStore:

    def test_store_high_confidence(self):
        mem  = LongTermMemory()
        fact = UserFact("u1", "age", 28, "fitness_plan", "s1", confidence=0.90)
        assert mem.store(fact) is True
        assert mem.get("u1", "age") == 28

    def test_store_below_threshold_rejected(self):
        mem  = LongTermMemory(min_confidence=0.70)
        fact = UserFact("u1", "age", 28, "fitness_plan", "s1", confidence=0.50)
        assert mem.store(fact) is False
        assert mem.get("u1", "age") is None

    def test_store_from_session(self):
        mem   = LongTermMemory()
        count = mem.store_from_session(
            user_id          = "u1",
            session_id       = "s1",
            goal_id          = "fitness_plan",
            collected_fields = {"age": 28, "weight_kg": 65.0, "name": "Priya"},
            field_confidences = {"age": 0.95, "weight_kg": 0.90, "name": 0.99},
        )
        assert count == 3
        assert mem.get("u1", "age")  == 28
        assert mem.get("u1", "name") == "Priya"

    def test_lower_confidence_existing_not_overwritten(self):
        mem = LongTermMemory()
        mem.store(UserFact("u1", "age", 28, "g", "s1", confidence=0.95))
        mem.store(UserFact("u1", "age", 30, "g", "s2", confidence=0.60))
        assert mem.get("u1", "age") == 28

    def test_higher_confidence_overwrites(self):
        mem = LongTermMemory()
        mem.store(UserFact("u1", "age", 28, "g", "s1", confidence=0.70))
        mem.store(UserFact("u1", "age", 29, "g", "s2", confidence=0.95))
        assert mem.get("u1", "age") == 29

class TestLongTermMemoryRead:

    def _mem(self) -> LongTermMemory:
        mem = LongTermMemory()
        mem.store_from_session("u1", "s1", "fitness_plan",
                                {"age": 28, "weight_kg": 65.0, "goal": "lose weight"},
                                {"age": 0.9, "weight_kg": 0.85, "goal": 0.95})
        mem.store_from_session("u1", "s2", "nutrition_plan",
                                {"calories": 1800},
                                {"calories": 0.80})
        return mem

    def test_get_returns_value(self):
        assert self._mem().get("u1", "age") == 28

    def test_get_unknown_returns_none(self):
        assert self._mem().get("u1", "nonexistent") is None

    def test_get_all(self):
        all_facts = self._mem().get_all("u1")
        assert "age"       in all_facts
        assert "weight_kg" in all_facts
        assert "calories"  in all_facts

    def test_get_by_goal(self):
        fitness_facts = self._mem().get_by_goal("u1", "fitness_plan")
        assert "age"       in fitness_facts
        assert "calories"  not in fitness_facts

    def test_get_all_facts_returns_user_fact_objects(self):
        facts = self._mem().get_all_facts("u1")
        assert all(isinstance(f, UserFact) for f in facts)

    def test_unknown_user_returns_empty(self):
        mem = self._mem()
        assert mem.get_all("unknown_user") == {}
        assert mem.get_all_facts("unknown_user") == []

class TestLongTermMemorySeed:

    def test_forget_one_field(self):
        mem = LongTermMemory()
        mem.store(UserFact("u1", "age",    28, "g", "s1"))
        mem.store(UserFact("u1", "weight", 65, "g", "s1"))
        mem.forget("u1", "age")
        assert mem.get("u1", "age")    is None
        assert mem.get("u1", "weight") == 65

    def test_forget_all_user_data(self):
        mem = LongTermMemory()
        mem.store(UserFact("u1", "age", 28, "g", "s1"))
        mem.store(UserFact("u1", "weight", 65, "g", "s1"))
        count = mem.forget("u1")
        assert count == 2
        assert mem.get_all("u1") == {}

    def test_stats_single_user(self):
        mem = LongTermMemory()
        mem.store(UserFact("u1", "age", 28, "g", "s1"))
        s = mem.stats("u1")
        assert s["fact_count"] == 1

    def test_stats_global(self):
        mem = LongTermMemory()
        mem.store(UserFact("u1", "age", 28, "g", "s1"))
        mem.store(UserFact("u2", "age", 35, "g", "s2"))
        s = mem.stats()
        assert s["total_users"] == 2
        assert s["total_facts"] == 2

class TestSessionResume:

    def _mock_sm(self, state_data: Optional[dict]) -> Any:
        sm = AsyncMock()
        sm.load = AsyncMock(return_value=state_data)
        return sm

    @pytest.mark.asyncio
    async def test_resumable_session(self):
        state = {
            "collected_fields": {"age": 28, "weight_kg": 65},
            "missing_required": ["primary_goal"],
            "completion_pct":   40.0,
            "current_turn":     4,
            "final_output":     None,
        }
        resume = SessionResume(session_manager=self._mock_sm(state))
        result = await resume.check("sess-abc")
        assert result.resumable       is True
        assert result.completion_pct  == pytest.approx(40.0)
        assert result.turns_completed == 4
        assert result.re_engagement_msg is not None

    @pytest.mark.asyncio
    async def test_not_resumable_not_found(self):
        resume = SessionResume(session_manager=self._mock_sm(None))
        result = await resume.check("nonexistent")
        assert result.resumable is False
        assert result.error     is not None

    @pytest.mark.asyncio
    async def test_not_resumable_completed(self):
        state = {
            "collected_fields": {"age": 28},
            "missing_required": [],
            "completion_pct":   100.0,
            "current_turn":     8,
            "final_output":     {"format": "json", "content": "{}"},
        }
        resume = SessionResume(session_manager=self._mock_sm(state))
        result = await resume.check("completed-sess")
        assert result.resumable is False

    def test_re_engagement_message_has_name(self):
        msg = SessionResume._re_engagement_message(
            collected={"name": "Priya", "age": 28},
            missing=["weight_kg"],
            turns=3,
        )
        assert "Priya" in msg
        assert "1" in msg or "weight" in msg.lower() or "1 more" in msg

class TestVectorStore:

    @pytest.mark.asyncio
    async def test_add_and_count(self):
        store = VectorStore()
        await store.add("s1", "patient has lower back pain", {"goal_id": "medical"})
        await store.add("s2", "financial planning for retirement",  {"goal_id": "financial"})
        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_search_returns_relevant_result(self):
        store = VectorStore()
        await store.add("s1", "lower back pain moderate severity")
        await store.add("s2", "financial retirement savings")
        results = await store.search("back pain")
        assert len(results) >= 1
        assert results[0].session_id == "s1"

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        store   = VectorStore()
        results = await store.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_top_k(self):
        store = VectorStore()
        for i in range(10):
            await store.add(f"s{i}", f"session {i} content health medical")
        results = await store.search("health medical", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self):
        store = VectorStore()
        await store.add("s1", "back pain", {"goal_id": "medical", "user_id": "u1"})
        await store.add("s2", "back pain", {"goal_id": "medical", "user_id": "u2"})
        results = await store.search("back pain", filter_meta={"user_id": "u1"})
        assert len(results) == 1
        assert results[0].session_id == "s1"

    @pytest.mark.asyncio
    async def test_delete_removes_session(self):
        store = VectorStore()
        await store.add("s1", "test content")
        deleted = await store.delete("s1")
        assert deleted is True
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_result_has_required_keys(self):
        store = VectorStore()
        await store.add("s1", "medical intake")
        results = await store.search("medical")
        d = results[0].to_dict()
        assert "session_id" in d
        assert "score"      in d
        assert "snippet"    in d

class TestDPDPManager:

    def _dpdp(self) -> DPDPManager:
        return DPDPManager(
            data_fiduciary = "HealthCo Pvt Ltd",
            purpose        = "Medical intake for personalised advice",
            retention_days = 90,
        )

    def test_consent_notice_has_required_text(self):
        dpdp   = self._dpdp()
        notice = dpdp.consent_notice(["health", "contact"])
        assert "DPDP" in notice or "Digital Personal Data" in notice
        assert "HealthCo Pvt Ltd" in notice
        assert "90 days"          in notice

    def test_grant_consent_creates_record(self):
        dpdp   = self._dpdp()
        record = dpdp.grant_consent("u1", "s1", "consent text")
        assert record.user_id  == "u1"
        assert record.status   == ConsentStatus.GRANTED
        assert record.is_valid is True

    def test_has_valid_consent_true_after_grant(self):
        dpdp = self._dpdp()
        dpdp.grant_consent("u1", "s1")
        assert dpdp.has_valid_consent("u1") is True

    def test_has_valid_consent_false_without_grant(self):
        dpdp = self._dpdp()
        assert dpdp.has_valid_consent("u_unknown") is False

    def test_withdraw_consent(self):
        dpdp = self._dpdp()
        dpdp.grant_consent("u1", "s1")
        assert dpdp.withdraw_consent("u1") is True
        assert dpdp.has_valid_consent("u1") is False

    def test_request_erasure(self):
        dpdp = self._dpdp()
        req  = dpdp.request_erasure("u1")
        assert req.right   == DataPrincipalRight.ERASE
        assert req.user_id == "u1"

    def test_request_access(self):
        dpdp = self._dpdp()
        req  = dpdp.request_access("u1")
        assert req.right == DataPrincipalRight.ACCESS

    def test_audit_log_records_actions(self):
        dpdp = self._dpdp()
        dpdp.grant_consent("u1", "s1")
        dpdp.withdraw_consent("u1")
        log = dpdp.audit_log("u1")
        actions = [e["action"] for e in log]
        assert "consent_granted"   in actions
        assert "consent_withdrawn" in actions

    def test_consent_record_to_dict(self):
        dpdp   = self._dpdp()
        record = dpdp.grant_consent("u1", "s1")
        d      = record.to_dict()
        for k in ["record_id", "user_id", "session_id", "purpose",
                   "data_fiduciary", "status", "is_valid"]:
            assert k in d

class TestGDPRManager:

    def _gdpr(self) -> GDPRManager:
        return GDPRManager(
            controller  = "MedTech GmbH",
            dpo_email   = "dpo@medtech.example.com",
            legal_basis = GDPRLegalBasis.CONSENT,
        )

    def test_privacy_notice_has_required_fields(self):
        gdpr   = self._gdpr()
        notice = gdpr.privacy_notice("Medical intake")
        assert "MedTech GmbH"   in notice
        assert "GDPR"           in notice or "Privacy" in notice
        assert "consent"        in notice.lower()

    def test_grant_consent_creates_record(self):
        gdpr   = self._gdpr()
        record = gdpr.grant_consent("u1", "s1", purpose="Medical intake")
        assert record.user_id    == "u1"
        assert record.is_active  is True
        assert record.legal_basis == GDPRLegalBasis.CONSENT

    def test_withdraw_consent(self):
        gdpr = self._gdpr()
        gdpr.grant_consent("u1", "s1")
        assert gdpr.withdraw_consent("u1") is True
        assert gdpr.has_valid_consent("u1") is False

    def test_has_valid_consent_false_for_unknown(self):
        gdpr = self._gdpr()
        assert gdpr.has_valid_consent("unknown") is False

    def test_data_subject_rights_request(self):
        gdpr   = self._gdpr()
        req_id = gdpr.request_right("u1", DataSubjectRight.ERASURE, "Please delete my data")
        assert isinstance(req_id, str) and len(req_id) > 0
        assert any(r["right"] == "erasure" for r in gdpr._rights_log)

    def test_audit_log_populated(self):
        gdpr = self._gdpr()
        gdpr.grant_consent("u1", "s1")
        gdpr.request_right("u1", DataSubjectRight.ACCESS)
        log = gdpr.audit_log("u1")
        assert len(log) >= 2

    def test_gdpr_record_to_dict(self):
        gdpr   = self._gdpr()
        record = gdpr.grant_consent("u1", "s1")
        d      = record.to_dict()
        for k in ["record_id", "user_id", "legal_basis", "is_active"]:
            assert k in d

class TestWhatsAppMessage:

    def _webhook_payload(self, text: str = "Hello", from_num: str = "+919876543210") -> dict:
        return {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id":        "wamid.123",
                            "from":      from_num,
                            "type":      "text",
                            "timestamp": str(int(time.time())),
                            "text":      {"body": text},
                        }]
                    }
                }]
            }]
        }

    def test_parse_text_message(self):
        payload = self._webhook_payload("I am 28 years old")
        msg     = WhatsAppMessage.from_webhook(payload)
        assert msg is not None
        assert msg.text         == "I am 28 years old"
        assert msg.from_number  == "+919876543210"
        assert msg.message_type == "text"

    def test_parse_returns_none_for_empty(self):
        payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
        assert WhatsAppMessage.from_webhook(payload) is None

    def test_parse_returns_none_for_malformed(self):
        assert WhatsAppMessage.from_webhook({}) is None
        assert WhatsAppMessage.from_webhook({"entry": []}) is None

    def test_to_dict_has_required_keys(self):
        payload = self._webhook_payload()
        msg     = WhatsAppMessage.from_webhook(payload)
        d       = msg.to_dict()
        for k in ["message_id", "from_number", "text", "type", "timestamp"]:
            assert k in d

    def test_interactive_button_reply_parsed(self):
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": "wamid.456",
                            "from": "+919876543210",
                            "type": "interactive",
                            "timestamp": str(int(time.time())),
                            "interactive": {
                                "type": "button_reply",
                                "button_reply": {"id": "0", "title": "Yes"},
                            },
                        }]
                    }
                }]
            }]
        }
        msg = WhatsAppMessage.from_webhook(payload)
        assert msg is not None
        assert msg.text         == "Yes"
        assert msg.message_type == "interactive"

class TestWhatsAppChannel:

    def _make_channel(self) -> WhatsAppChannel:
        from truenorth.testing.mock_llm import MockLLMClient
        from truenorth.llm.router       import LLMRouter
        from truenorth.core.engine      import TrueNorthEngine

        goal = {
            "id": "wa_test",
            "fields": [{"name": "age", "type": "integer", "required": True, "question": "Age?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output": {"format": "json"},
        }
        mock   = MockLLMClient(default="Got it.")
        router = LLMRouter()
        for m in ["gemini-3.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        def factory(session_id):
            return TrueNorthEngine(goal_config=goal, router=router, session_id=session_id)

        channel = WhatsAppChannel(
            engine_factory = factory,
            verify_token   = "test_token",
            access_token   = "test_access",
            phone_id       = "123456789",
        )
        channel._send_text = AsyncMock(return_value={"messages": [{"id": "msg-1"}]})
        return channel

    def test_verify_webhook_success(self):
        channel = self._make_channel()
        result  = channel.verify_webhook("subscribe", "test_token", "challenge_xyz")
        assert result == "challenge_xyz"

    def test_verify_webhook_wrong_token(self):
        channel = self._make_channel()
        result  = channel.verify_webhook("subscribe", "wrong_token", "challenge_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_webhook_processes_message(self):
        channel = self._make_channel()
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": "msg-1", "from": "+919876543210",
                            "type": "text", "timestamp": str(int(time.time())),
                            "text": {"body": "I am 28 years old"},
                        }]
                    }
                }]
            }]
        }
        result = await channel.handle_webhook(payload)
        assert result["status"]    == "ok"
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_handle_webhook_no_message(self):
        channel = self._make_channel()
        payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
        result  = await channel.handle_webhook(payload)
        assert result["status"]    == "ok"
        assert result["processed"] is False

    @pytest.mark.asyncio
    async def test_session_persisted_across_calls(self):
        channel = self._make_channel()
        payload = lambda text: {
            "entry": [{"changes": [{"value": {"messages": [{
                "id": f"msg-{text[:3]}", "from": "+911111111111",
                "type": "text", "timestamp": str(int(time.time())),
                "text": {"body": text},
            }]}}]}]
        }
        await channel.handle_webhook(payload("first message"))
        await channel.handle_webhook(payload("second message"))
        assert len(channel._sessions) == 1

class TestSectorCompliance:

    SECTORS = [
        ("healthcare",     "HealthCo Pvt Ltd",  "Patient intake for personalised treatment"),
        ("legal",          "LexFirm LLP",        "Case intake for legal representation"),
        ("hr_recruitment", "TechCo HR Dept",     "Candidate screening for employment"),
        ("financial",      "FinServ Ltd",         "KYC and financial planning advice"),
        ("fitness",        "FitApp Pvt Ltd",      "Fitness assessment and plan creation"),
    ]

    @pytest.mark.parametrize("sector,fiduciary,purpose", SECTORS)
    def test_dpdp_consent_cycle_all_sectors(self, sector, fiduciary, purpose):
        """Full DPDP consent cycle works for every sector."""
        dpdp   = DPDPManager(data_fiduciary=fiduciary, purpose=purpose)
        notice = dpdp.consent_notice(["personal_data"])
        assert fiduciary in notice

        record = dpdp.grant_consent(f"user_{sector}", f"sess_{sector}", notice)
        assert record.is_valid is True
        assert dpdp.has_valid_consent(f"user_{sector}") is True

        req = dpdp.request_erasure(f"user_{sector}")
        assert req.right == DataPrincipalRight.ERASE

        log = dpdp.audit_log(f"user_{sector}")
        assert len(log) >= 2

    @pytest.mark.parametrize("sector,fiduciary,purpose", SECTORS)
    def test_gdpr_consent_cycle_all_sectors(self, sector, fiduciary, purpose):
        """Full GDPR consent cycle works for every sector."""
        gdpr   = GDPRManager(controller=fiduciary, dpo_email=f"dpo@{sector}.example.com")
        notice = gdpr.privacy_notice(purpose)
        assert fiduciary in notice

        record = gdpr.grant_consent(f"eu_user_{sector}", f"sess_{sector}", purpose=purpose)
        assert record.is_active is True

        req_id = gdpr.request_right(f"eu_user_{sector}", DataSubjectRight.PORTABILITY)
        assert req_id is not None
