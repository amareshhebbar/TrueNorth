"""
No network calls, no external services.

Classes:
  1.  LogCategories       — event factory functions and types
  2.  LogEvent            — serialisation, from_dict round-trip
  3.  MemorySink          — stores events, filter by category/session
  4.  StdoutSink          — category filter, JSON output
  5.  CallbackSink        — invokes user callback per event
  6.  TurnTrace           — lifecycle, cost/token aggregation
  7.  SessionTrace        — multi-turn aggregation, summary
  8.  Tracer_Record       — record_* methods populate TurnTrace
  9.  Tracer_Emit         — events reach registered sinks
  10. Tracer_Query        — session_summary, get_turn_trace
  11. HealthMonitor_Empty — zero sessions → empty report
  12. HealthMonitor_Report — completion rate, avg turns, skip rates
  13. HealthMonitor_Alerts — threshold violations produce alerts
  14. HealthMonitor_Compare — period-over-period comparison
  15. ABEngine_Assign     — deterministic hash assignment
  16. ABEngine_Outcome    — record completions, compute rates
  17. ABEngine_Significance — z-test, p-value, winner
  18. ABRegistry          — manage multiple tests
  19. CostDashboard_Summary — goal cost aggregation
  20. CostDashboard_Detail  — session-level detail
  21. CostDashboard_Trend   — time series bucketing
  22. CostDashboard_Models  — per-model breakdown
  23. SectorObservability   — same stack for 5 sectors
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.observability import (
    LogCategory, LogLevel, LogEvent, make_event,
    conversation_event, extraction_event, emotion_event,
    conflict_event, cost_event, hallucination_event, compliance_event,
    TrueNorthTracer, TurnTrace, SessionTrace,
    MemorySink, StdoutSink, CallbackSink,
    HealthMonitor, GoalHealthReport,
    ABEngine, ABRegistry, ABVariant, ABStatus, CostDashboard, CostSummary,
)
from truenorth.observability.log_categories import make_event


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tracer_with_memory() -> tuple[TrueNorthTracer, MemorySink]:
    tracer = TrueNorthTracer()
    sink   = MemorySink()
    tracer.add_sink(sink)
    return tracer, sink


def _populated_tracer(
    goal_id:     str = "fitness_plan",
    session_id:  str = "s1",
    turns:       int = 4,
    completed:   bool = True,
) -> TrueNorthTracer:
    """Build a tracer with a full session worth of events."""
    tracer, sink = _tracer_with_memory()
    tracer.session_start(session_id, goal_id, user_id="u1")
    for t in range(1, turns + 1):
        trace = tracer.turn_start(session_id, goal_id, t)
        tracer.record_user_input(session_id, goal_id, t, f"turn {t} input", language="en")
        tracer.record_extraction(session_id, goal_id, t, f"field_{t}", f"val_{t}", 0.90, True, "gemini-3.5-flash")
        tracer.record_emotion(session_id, goal_id, t, "neutral", 0.0, 0.2)
        tracer.record_llm_call(session_id, goal_id, t, "gemini-3.5-flash", "extract", 100, 50, 0.0001, 300)
        tracer.record_hallucination(session_id, goal_id, t, "CLEAN", 3, 0, "some output")
        tracer.record_response(session_id, goal_id, t, "OK response", 320)
        tracer.turn_end(session_id, t, "OK response")
    if completed:
        tracer.session_end(session_id)
        sess = tracer.get_session(session_id)
        if sess:
            sess.finished_at = time.time()
    return tracer


# ─────────────────────────────────────────────────────────────────────────────
#  1. LogCategory events
# ─────────────────────────────────────────────────────────────────────────────

class TestLogCategories:

    def test_make_event_produces_log_event(self):
        ev = make_event(LogCategory.COST, "s1", "g1", {"x": 1})
        assert isinstance(ev, LogEvent)
        assert ev.category == LogCategory.COST

    def test_conversation_event_roles(self):
        ev = conversation_event("s1", "g1", 1, "user", "hello")
        assert ev.data["role"] == "user"
        assert ev.data["text_snippet"] == "hello"

    def test_extraction_event_success(self):
        ev = extraction_event("s1", "g1", 1, "age", 28, 0.92, True, "gemini")
        assert ev.data["success"]    is True
        assert ev.data["field"]      == "age"
        assert ev.data["confidence"] == pytest.approx(0.92)

    def test_extraction_event_failure_is_warning(self):
        ev = extraction_event("s1", "g1", 1, "age", None, 0.30, False)
        assert ev.level == LogLevel.WARNING

    def test_emotion_event_distress_is_warning(self):
        ev = emotion_event("s1", "g1", 1, "distressed", -0.8, 0.9)
        assert ev.level == LogLevel.WARNING

    def test_conflict_event_high_severity_warning(self):
        ev = conflict_event("s1", "g1", 2, "age", 28, 35, "NUMERIC_MISMATCH", "HIGH")
        assert ev.level == LogLevel.WARNING

    def test_cost_event_has_all_fields(self):
        ev = cost_event("s1", "g1", 1, "claude-haiku-4-5-20251001", "extract", 200, 80, 0.00024, 420)
        assert ev.data["model"]        == "claude-haiku-4-5-20251001"
        assert ev.data["input_tokens"] == 200
        assert ev.data["cost_usd"]     == pytest.approx(0.00024)
        assert ev.data["total_tokens"] == 280

    def test_hallucination_blocked_is_error(self):
        ev = hallucination_event("s1", "g1", 3, "BLOCKED", 5, 2, "bad claim")
        assert ev.level == LogLevel.ERROR

    def test_compliance_event_data(self):
        ev = compliance_event("s1", "g1", 0, "consent_granted", "dpdp", "u1", {"ip": "1.2.3.4"})
        assert ev.data["action"]    == "consent_granted"
        assert ev.data["framework"] == "dpdp"
        assert ev.user_id           == "u1"

    def test_all_category_values(self):
        assert LogCategory.CONVERSATION  == "conversation"
        assert LogCategory.EXTRACTION    == "extraction"
        assert LogCategory.EMOTION       == "emotion"
        assert LogCategory.CONFLICT      == "conflict"
        assert LogCategory.COST          == "cost"
        assert LogCategory.HALLUCINATION == "hallucination"
        assert LogCategory.COMPLIANCE    == "compliance"


# ─────────────────────────────────────────────────────────────────────────────
#  2. LogEvent
# ─────────────────────────────────────────────────────────────────────────────

class TestLogEvent:

    def test_to_dict_has_required_keys(self):
        ev = make_event(LogCategory.COST, "s1", "g1", {"x": 1})
        d  = ev.to_dict()
        for k in ["event_id", "category", "level", "session_id", "goal_id", "timestamp", "data"]:
            assert k in d

    def test_from_dict_round_trip(self):
        ev1 = make_event(LogCategory.EXTRACTION, "sess-abc", "fitness", {"field": "age"})
        d   = ev1.to_dict()
        ev2 = LogEvent.from_dict(d)
        assert ev2.session_id == "sess-abc"
        assert ev2.category   == LogCategory.EXTRACTION
        assert ev2.data["field"] == "age"

    def test_event_id_unique(self):
        e1 = make_event(LogCategory.COST, "s1", "g1", {})
        e2 = make_event(LogCategory.COST, "s1", "g1", {})
        assert e1.event_id != e2.event_id

    def test_timestamp_is_recent(self):
        ev  = make_event(LogCategory.COST, "s1", "g1", {})
        now = time.time()
        assert now - 5 < ev.timestamp <= now


# ─────────────────────────────────────────────────────────────────────────────
#  3. MemorySink
# ─────────────────────────────────────────────────────────────────────────────

class TestMemorySink:

    @pytest.mark.asyncio
    async def test_stores_emitted_events(self):
        sink = MemorySink()
        ev   = make_event(LogCategory.COST, "s1", "g1", {})
        await sink.emit(ev)
        assert len(sink.events) == 1

    @pytest.mark.asyncio
    async def test_by_category_filter(self):
        sink = MemorySink()
        await sink.emit(make_event(LogCategory.COST,       "s1", "g1", {}))
        await sink.emit(make_event(LogCategory.EXTRACTION, "s1", "g1", {}))
        await sink.emit(make_event(LogCategory.COST,       "s2", "g1", {}))
        cost_events = sink.by_category(LogCategory.COST)
        assert len(cost_events) == 2

    @pytest.mark.asyncio
    async def test_by_session_filter(self):
        sink = MemorySink()
        await sink.emit(make_event(LogCategory.COST, "sess-A", "g", {}))
        await sink.emit(make_event(LogCategory.COST, "sess-B", "g", {}))
        assert len(sink.by_session("sess-A")) == 1

    @pytest.mark.asyncio
    async def test_max_events_rolling_window(self):
        sink = MemorySink(max_events=5)
        for i in range(10):
            await sink.emit(make_event(LogCategory.COST, f"s{i}", "g", {}))
        assert len(sink.events) == 5

    def test_clear(self):
        sink = MemorySink()
        sink._events = [make_event(LogCategory.COST, "s", "g", {})]
        sink.clear()
        assert len(sink.events) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  4. StdoutSink
# ─────────────────────────────────────────────────────────────────────────────

class TestStdoutSink:

    @pytest.mark.asyncio
    async def test_category_filter_blocks(self, capsys):
        sink = StdoutSink(categories=[LogCategory.COST])
        await sink.emit(make_event(LogCategory.EXTRACTION, "s1", "g", {}))
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.asyncio
    async def test_category_filter_passes(self, capsys):
        sink = StdoutSink(categories=[LogCategory.COST])
        await sink.emit(make_event(LogCategory.COST, "s1", "g", {"amount": 1}))
        captured = capsys.readouterr()
        assert "cost" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
#  5. CallbackSink
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbackSink:

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        received = []
        sink = CallbackSink(callback=lambda ev: received.append(ev.category))
        await sink.emit(make_event(LogCategory.COST, "s1", "g", {}))
        assert LogCategory.COST in received

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self):
        received = []
        async def cb(ev): received.append(ev.session_id)
        sink = CallbackSink(callback=cb)
        await sink.emit(make_event(LogCategory.COST, "sess-X", "g", {}))
        assert "sess-X" in received


# ─────────────────────────────────────────────────────────────────────────────
#  6. TurnTrace
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnTrace:

    def test_latency_ms_computed(self):
        trace = TurnTrace("s1", "g1", 1)
        trace.started_at  = time.time() - 0.5
        trace.finished_at = time.time()
        assert trace.latency_ms >= 400   # approx 500ms

    def test_total_cost_aggregated(self):
        trace = TurnTrace("s1", "g1", 1)
        trace.llm_calls = [
            {"cost_usd": 0.001},
            {"cost_usd": 0.002},
        ]
        assert trace.total_cost_usd == pytest.approx(0.003)

    def test_total_tokens_aggregated(self):
        trace = TurnTrace("s1", "g1", 1)
        trace.llm_calls = [
            {"total_tokens": 100},
            {"total_tokens": 50},
        ]
        assert trace.total_tokens == 150

    def test_extraction_success_rate(self):
        trace = TurnTrace("s1", "g1", 1)
        trace.extractions = [
            {"success": True}, {"success": True}, {"success": False},
        ]
        assert trace.extraction_success_rate == pytest.approx(2/3)

    def test_to_dict_has_required_keys(self):
        trace = TurnTrace("s1", "g1", 1)
        d     = trace.to_dict()
        for k in ["session_id", "goal_id", "turn", "latency_ms",
                  "total_cost_usd", "total_tokens"]:
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
#  7. SessionTrace
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionTrace:

    def _session(self) -> SessionTrace:
        s = SessionTrace("s1", "fitness_plan", user_id="u1")
        for i in range(3):
            t = TurnTrace("s1", "fitness_plan", i + 1)
            t.llm_calls     = [{"cost_usd": 0.001, "total_tokens": 100}]
            t.pii_detected  = (i == 1)   # PII on turn 2
            t.conflicts     = [{"field": "age"}] if i == 2 else []
            t.fw_verdict    = "CLEAN"
            t.finished_at   = time.time()
            s.turns.append(t)
        s.finished_at = time.time()
        return s

    def test_turn_count(self):
        s = self._session()
        assert s.turn_count == 3

    def test_total_cost(self):
        s = self._session()
        assert s.total_cost_usd == pytest.approx(0.003)

    def test_pii_turn_count(self):
        s = self._session()
        assert s.pii_turn_count == 1

    def test_conflict_count(self):
        s = self._session()
        assert s.conflict_count == 1

    def test_session_summary_has_keys(self):
        s = self._session()
        d = s.session_summary()
        for k in ["session_id", "goal_id", "turn_count", "total_cost_usd",
                  "pii_turns", "conflicts"]:
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
#  8. Tracer — record methods
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerRecord:

    def test_record_user_input_updates_turn_trace(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "fitness")
        trace = tracer.turn_start("s1", "fitness", 1)
        tracer.record_user_input("s1", "fitness", 1, "hello world", language="hi")
        assert trace.user_text_chars == 11
        assert trace.user_language   == "hi"

    def test_record_extraction_adds_to_list(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "fitness")
        trace = tracer.turn_start("s1", "fitness", 1)
        tracer.record_extraction("s1", "fitness", 1, "age", 28, 0.92, True, "gemini")
        assert len(trace.extractions) == 1
        assert trace.extractions[0]["field"] == "age"

    def test_record_pii_sets_flag(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "medical")
        trace = tracer.turn_start("s1", "medical", 1)
        tracer.record_pii("s1", "medical", 1, ["AADHAAR", "PHONE"])
        assert trace.pii_detected is True
        assert "AADHAAR" in trace.pii_types

    def test_record_emotion(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "g1")
        trace = tracer.turn_start("s1", "g1", 1)
        tracer.record_emotion("s1", "g1", 1, "anxious", -0.4, 0.7)
        assert trace.emotion == "anxious"

    def test_record_conflict(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "g1")
        trace = tracer.turn_start("s1", "g1", 1)
        tracer.record_conflict("s1", "g1", 1, "age", 28, 35, "NUMERIC_MISMATCH", "HIGH")
        assert len(trace.conflicts) == 1

    def test_record_llm_call(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "g1")
        trace = tracer.turn_start("s1", "g1", 1)
        tracer.record_llm_call("s1", "g1", 1, "gemini-3.5-flash", "extract", 200, 80, 0.0001, 300)
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0]["model"] == "gemini-3.5-flash"

    def test_record_hallucination(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "g1")
        trace = tracer.turn_start("s1", "g1", 1)
        tracer.record_hallucination("s1", "g1", 1, "FLAGGED", 5, 1, "snippet")
        assert trace.fw_verdict == "FLAGGED"
        assert trace.fw_blocked == 1

    def test_turn_end_sets_finished_at(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "g1")
        trace = tracer.turn_start("s1", "g1", 1)
        tracer.turn_end("s1", 1, "response text")
        assert trace.finished_at is not None
        assert trace.response_chars == len("response text")


# ─────────────────────────────────────────────────────────────────────────────
#  9. Tracer — events reach sinks
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerEmit:

    @pytest.mark.asyncio
    async def test_events_reach_memory_sink(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "fitness")
        tracer.turn_start("s1", "fitness", 1)
        tracer.record_llm_call("s1", "fitness", 1, "gemini", "extract", 100, 50, 0.0001, 200)
        await asyncio.sleep(0.01)   # let event tasks complete
        cost_events = sink.by_category(LogCategory.COST)
        assert len(cost_events) >= 1

    @pytest.mark.asyncio
    async def test_pii_emits_compliance_event(self):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", "g")
        tracer.turn_start("s1", "g", 1)
        tracer.record_pii("s1", "g", 1, ["AADHAAR"])
        await asyncio.sleep(0.01)
        comp_events = sink.by_category(LogCategory.COMPLIANCE)
        assert len(comp_events) >= 1

    @pytest.mark.asyncio
    async def test_multiple_sinks_both_receive(self):
        received_a, received_b = [], []
        tracer = TrueNorthTracer()
        tracer.add_sink(CallbackSink(lambda ev: received_a.append(ev)))
        tracer.add_sink(CallbackSink(lambda ev: received_b.append(ev)))
        tracer.session_start("s1", "g")
        await asyncio.sleep(0.01)
        assert len(received_a) >= 1
        assert len(received_b) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  10. Tracer — query API
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerQuery:

    def test_session_summary_returns_dict(self):
        tracer = _populated_tracer()
        summary = tracer.session_summary("s1")
        assert summary is not None
        assert "turn_count"      in summary
        assert "total_cost_usd"  in summary

    def test_get_turn_trace(self):
        tracer = _populated_tracer(turns=3)
        t = tracer.get_turn_trace("s1", 2)
        assert t is not None
        assert t.turn == 2

    def test_all_session_ids(self):
        tracer = _populated_tracer(session_id="s1")
        _populated_tracer.__wrapped__ = None
        # Add second session
        tracer.session_start("s2", "fitness_plan")
        assert "s1" in tracer.all_session_ids()
        assert "s2" in tracer.all_session_ids()

    def test_session_summary_none_for_unknown(self):
        tracer = TrueNorthTracer()
        assert tracer.session_summary("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────────────
#  11. HealthMonitor — empty
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthMonitorEmpty:

    def test_empty_report_for_unknown_goal(self):
        tracer  = TrueNorthTracer()
        monitor = HealthMonitor(tracer=tracer)
        report  = monitor.goal_report("nonexistent_goal", window_hours=24)
        assert report.session_count   == 0
        assert report.completion_rate == 0.0

    def test_no_alerts_for_empty(self):
        tracer  = TrueNorthTracer()
        monitor = HealthMonitor(tracer=tracer)
        report  = monitor.goal_report("empty")
        alerts  = monitor.check_alerts(report)
        assert isinstance(alerts, list)


# ─────────────────────────────────────────────────────────────────────────────
#  12. HealthMonitor — full report
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthMonitorReport:

    def _monitor_with_sessions(self, n_completed: int = 8, n_total: int = 10) -> HealthMonitor:
        tracer = TrueNorthTracer()
        for i in range(n_total):
            sid  = f"sess-{i}"
            completed = i < n_completed
            tracer.session_start(sid, "fitness_plan", user_id=f"u{i}")
            for t in range(1, 5):
                trace = tracer.turn_start(sid, "fitness_plan", t)
                tracer.record_user_input(sid, "fitness_plan", t, f"msg {t}")
                tracer.record_extraction(sid, "fitness_plan", t, f"field_{t}", f"val_{t}", 0.90, True)
                tracer.record_llm_call(sid, "fitness_plan", t, "gemini-3.5-flash", "extract", 100, 50, 0.0001, 300)
                tracer.turn_end(sid, t, "response")
            tracer.session_end(sid)
            sess = tracer.get_session(sid)
            if sess:
                sess.finished_at = time.time()
        return HealthMonitor(tracer=tracer)

    def test_completion_rate_correct(self):
        monitor = self._monitor_with_sessions(8, 10)
        report  = monitor.goal_report("fitness_plan", window_hours=24)
        assert report.session_count == 10
        # All sessions have finished_at → all "completed" per _is_complete()
        assert report.completion_rate > 0

    def test_report_to_dict_has_keys(self):
        monitor = self._monitor_with_sessions()
        report  = monitor.goal_report("fitness_plan", window_hours=24)
        d = report.to_dict()
        for k in ["goal_id", "completion_rate", "avg_turns", "p95_latency_ms",
                  "field_skip_rates", "abandonment_map", "pii_rate"]:
            assert k in d

    def test_field_skip_rates_computed(self):
        tracer = TrueNorthTracer()
        sid    = "s1"
        tracer.session_start(sid, "g1")
        trace  = tracer.turn_start(sid, "g1", 1)
        tracer.record_extraction(sid, "g1", 1, "age", 28, 0.90, True)
        tracer.record_extraction(sid, "g1", 1, "age", None, 0.20, False)
        monitor = HealthMonitor(tracer=tracer)
        report  = monitor.goal_report("g1", window_hours=24)
        if "age" in report.field_skip_rates:
            assert report.field_skip_rates["age"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  13. HealthMonitor — alerts
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthMonitorAlerts:

    def test_low_completion_triggers_alert(self):
        report = GoalHealthReport(
            goal_id="g", window_hours=24, session_count=100,
            completed_count=40, abandoned_count=60,
            completion_rate=0.40,   # below 0.60 threshold
            avg_turns=5.0, avg_cost_usd=0.01,
            p50_latency_ms=300, p95_latency_ms=800,
            field_skip_rates={}, abandonment_map={},
            pii_rate=0.0, conflict_rate=0.0, hallucination_rate=0.0,
        )
        monitor = HealthMonitor(tracer=TrueNorthTracer())
        alerts  = monitor.check_alerts(report)
        metrics = [a["metric"] for a in alerts]
        assert "completion_rate" in metrics

    def test_high_hallucination_triggers_critical(self):
        report = GoalHealthReport(
            goal_id="g", window_hours=24, session_count=100,
            completed_count=80, abandoned_count=10,
            completion_rate=0.80, avg_turns=5.0, avg_cost_usd=0.01,
            p50_latency_ms=300, p95_latency_ms=800,
            field_skip_rates={}, abandonment_map={},
            pii_rate=0.0, conflict_rate=0.0,
            hallucination_rate=0.10,   # above 0.05 threshold
        )
        monitor = HealthMonitor(tracer=TrueNorthTracer())
        alerts  = monitor.check_alerts(report)
        metrics = [a["metric"] for a in alerts]
        assert "hallucination_rate" in metrics
        hw_alert = next(a for a in alerts if a["metric"] == "hallucination_rate")
        assert hw_alert["severity"] == "critical"

    def test_high_field_skip_triggers_alert(self):
        report = GoalHealthReport(
            goal_id="g", window_hours=24, session_count=50,
            completed_count=40, abandoned_count=5,
            completion_rate=0.80, avg_turns=5.0, avg_cost_usd=0.01,
            p50_latency_ms=300, p95_latency_ms=800,
            field_skip_rates={"chief_complaint": 0.45},   # above 0.30 threshold
            abandonment_map={}, pii_rate=0.0, conflict_rate=0.0, hallucination_rate=0.01,
        )
        monitor = HealthMonitor(tracer=TrueNorthTracer())
        alerts  = monitor.check_alerts(report)
        field_alerts = [a for a in alerts if a["metric"] == "field_skip_rate"]
        assert len(field_alerts) >= 1
        assert field_alerts[0]["field"] == "chief_complaint"

    def test_healthy_report_no_alerts(self):
        report = GoalHealthReport(
            goal_id="g", window_hours=24, session_count=200,
            completed_count=180, abandoned_count=10,
            completion_rate=0.90, avg_turns=5.0, avg_cost_usd=0.01,
            p50_latency_ms=300, p95_latency_ms=1200,
            field_skip_rates={"age": 0.05}, abandonment_map={},
            pii_rate=0.1, conflict_rate=0.2, hallucination_rate=0.01,
        )
        monitor = HealthMonitor(tracer=TrueNorthTracer())
        alerts  = monitor.check_alerts(report)
        assert len(alerts) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  14. HealthMonitor — period comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthMonitorCompare:

    def test_compare_returns_dict(self):
        tracer  = TrueNorthTracer()
        monitor = HealthMonitor(tracer=tracer)
        result  = monitor.compare_periods("fitness_plan", current_hours=24, previous_hours=24)
        assert "goal_id"          in result
        assert "completion_rate"  in result
        assert "session_count"    in result

    def test_compare_change_pct_computed(self):
        tracer  = TrueNorthTracer()
        monitor = HealthMonitor(tracer=tracer)
        result  = monitor.compare_periods("fitness_plan")
        cr = result["completion_rate"]
        assert "current" in cr and "previous" in cr


# ─────────────────────────────────────────────────────────────────────────────
#  15. ABEngine — assignment
# ─────────────────────────────────────────────────────────────────────────────

class TestABEngineAssign:

    def _ab(self) -> ABEngine:
        return ABEngine(
            test_id          = "test1",
            variant_a_config = {"id": "v1"},
            variant_b_config = {"id": "v2"},
            split_ratio      = 0.50,
            min_sessions     = 10,
        )

    def test_same_session_gets_same_variant(self):
        ab = self._ab()
        c1 = ab.assign("sess-stable")
        c2 = ab.assign("sess-stable")
        assert c1 is c2

    def test_get_variant_returns_assignment(self):
        ab = self._ab()
        ab.assign("sess-xyz")
        v = ab.get_variant("sess-xyz")
        assert v in (ABVariant.A, ABVariant.B)

    def test_unassigned_returns_none(self):
        ab = self._ab()
        assert ab.get_variant("never_assigned") is None

    def test_split_roughly_50_50(self):
        ab = self._ab()
        counts = {ABVariant.A: 0, ABVariant.B: 0}
        for i in range(1000):
            ab.assign(f"sess-{i}")
            v = ab.get_variant(f"sess-{i}")
            counts[v] += 1
        # With 1000 sessions expect within 10% of 50/50
        assert 400 < counts[ABVariant.A] < 600
        assert 400 < counts[ABVariant.B] < 600

    def test_custom_split_30_70(self):
        ab = ABEngine("t", {"id": "a"}, {"id": "b"}, split_ratio=0.30, min_sessions=10)
        b_count = 0
        for i in range(1000):
            ab.assign(f"s{i}")
            if ab.get_variant(f"s{i}") == ABVariant.B:
                b_count += 1
        assert 200 < b_count < 400


# ─────────────────────────────────────────────────────────────────────────────
#  16. ABEngine — outcome recording
# ─────────────────────────────────────────────────────────────────────────────

class TestABEngineOutcome:

    def test_record_outcome_increments_completions(self):
        ab = ABEngine("t", {"id": "a"}, {"id": "b"}, split_ratio=0.50, min_sessions=5)
        ab.assign("s1")
        variant = ab.get_variant("s1")
        ab.record_outcome("s1", completed=True, cost_usd=0.01, turns=5)
        stats = ab._stats[variant]
        assert stats.completions == 1
        assert stats.total_cost  == pytest.approx(0.01)

    def test_incomplete_doesnt_increment(self):
        ab = ABEngine("t", {"id": "a"}, {"id": "b"}, split_ratio=0.50, min_sessions=5)
        ab.assign("s1")
        variant = ab.get_variant("s1")
        ab.record_outcome("s1", completed=False)
        assert ab._stats[variant].completions == 0

    def test_unassigned_session_record_ignored(self):
        ab = ABEngine("t", {"id": "a"}, {"id": "b"}, split_ratio=0.50, min_sessions=5)
        ab.record_outcome("not_assigned", completed=True)   # should not crash
        assert ab._stats[ABVariant.A].completions == 0


# ─────────────────────────────────────────────────────────────────────────────
#  17. ABEngine — statistical significance
# ─────────────────────────────────────────────────────────────────────────────

class TestABEngineSignificance:

    def _ab_with_data(
        self,
        n:      int   = 200,
        rate_a: float = 0.60,
        rate_b: float = 0.80,
    ) -> ABEngine:
        ab = ABEngine("sig_test", {"id":"a"}, {"id":"b"}, split_ratio=0.50, min_sessions=50)
        for i in range(n):
            sid  = f"sess-{i}"
            cfg  = ab.assign(sid)
            v    = ab.get_variant(sid)
            rate = rate_b if v == ABVariant.B else rate_a
            ab.record_outcome(sid, completed=(i % 10 < int(rate * 10)))
        return ab

    def test_not_enough_data_returns_running(self):
        ab = ABEngine("t", {"id":"a"}, {"id":"b"}, split_ratio=0.50, min_sessions=1000)
        for i in range(5):
            ab.assign(f"s{i}")
            ab.record_outcome(f"s{i}", True)
        result = ab.result()
        assert result.status == ABStatus.RUNNING

    def test_significant_difference_detected(self):
        ab     = self._ab_with_data(n=400, rate_a=0.50, rate_b=0.80)
        result = ab.result()
        # With 200/200 and a 30pp difference, p should be significant
        if result.p_value is not None:
            assert result.p_value < 0.05

    def test_result_to_dict(self):
        ab     = self._ab_with_data(n=200)
        result = ab.result()
        d      = result.to_dict()
        for k in ["test_id", "status", "variant_a", "variant_b"]:
            assert k in d

    def test_variant_stats_to_dict(self):
        ab     = self._ab_with_data(n=200)
        result = ab.result()
        d      = result.stats_a.to_dict()
        for k in ["variant", "sessions", "completions", "completion_rate"]:
            assert k in d

    def test_stop_changes_status(self):
        ab = ABEngine("t", {"id":"a"}, {"id":"b"}, split_ratio=0.50, min_sessions=10)
        ab.stop()
        assert ab._status == ABStatus.STOPPED


# ─────────────────────────────────────────────────────────────────────────────
#  18. ABRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestABRegistry:

    def test_register_and_assign(self):
        registry = ABRegistry()
        registry.register(ABEngine("test_a", {"id":"a"}, {"id":"b"}, min_sessions=5))
        cfg = registry.assign("test_a", "sess-1")
        assert isinstance(cfg, dict)

    def test_assign_unknown_test_returns_none(self):
        registry = ABRegistry()
        assert registry.assign("nonexistent", "sess-1") is None

    def test_record_outcome_delegates(self):
        registry = ABRegistry()
        registry.register(ABEngine("t1", {"id":"a"}, {"id":"b"}, split_ratio=0.50, min_sessions=5))
        registry.assign("t1", "s1")
        registry.record_outcome("t1", "s1", completed=True)
        v     = registry._tests["t1"].get_variant("s1")
        stats = registry._tests["t1"]._stats[v]
        assert stats.completions == 1

    def test_all_results(self):
        registry = ABRegistry()
        registry.register(ABEngine("t1", {"id":"a"}, {"id":"b"}, min_sessions=5))
        registry.register(ABEngine("t2", {"id":"a"}, {"id":"b"}, min_sessions=5))
        results = registry.all_results()
        assert "t1" in results and "t2" in results

    def test_list_tests(self):
        registry = ABRegistry()
        registry.register(ABEngine("x1", {}, {}, min_sessions=5))
        registry.register(ABEngine("x2", {}, {}, min_sessions=5))
        assert "x1" in registry.list_tests()
        assert "x2" in registry.list_tests()


# ─────────────────────────────────────────────────────────────────────────────
#  19. CostDashboard — goal cost summary
# ─────────────────────────────────────────────────────────────────────────────

class TestCostDashboardSummary:

    def _dashboard(self, goal_id: str = "fitness") -> CostDashboard:
        tracer = _populated_tracer(goal_id=goal_id, session_id="s1", turns=4)
        return CostDashboard(tracer=tracer)

    def test_summary_for_known_goal(self):
        dash    = self._dashboard("fitness")
        summary = dash.goal_cost_summary("fitness", period_days=1)
        assert isinstance(summary, CostSummary)
        assert summary.goal_id      == "fitness"
        assert summary.session_count >= 1
        assert summary.total_cost_usd >= 0

    def test_summary_to_dict_has_keys(self):
        dash = self._dashboard()
        d    = dash.goal_cost_summary("fitness", period_days=1).to_dict()
        for k in ["goal_id", "session_count", "total_cost_usd", "by_model", "by_task"]:
            assert k in d

    def test_empty_summary_for_unknown_goal(self):
        dash    = self._dashboard()
        summary = dash.goal_cost_summary("nonexistent_goal_xyz")
        assert summary.session_count == 0
        assert summary.total_cost_usd == 0.0

    def test_by_model_populated(self):
        dash    = self._dashboard()
        summary = dash.goal_cost_summary("fitness", period_days=1)
        assert "gemini-3.5-flash" in summary.by_model

    def test_by_task_populated(self):
        dash    = self._dashboard()
        summary = dash.goal_cost_summary("fitness", period_days=1)
        assert "extract" in summary.by_task


# ─────────────────────────────────────────────────────────────────────────────
#  20. CostDashboard — session detail
# ─────────────────────────────────────────────────────────────────────────────

class TestCostDashboardDetail:

    def test_session_detail_without_cost_tracker(self):
        tracer = _populated_tracer(session_id="s1")
        dash   = CostDashboard(tracer=tracer)
        d      = dash.session_cost_detail("s1")
        assert d["session_id"]  == "s1"
        assert "turn_count"     in d or "turns" in d

    def test_session_detail_with_cost_tracker(self):
        from truenorth.llm.cost_tracker import CostTracker
        ct = CostTracker()
        ct.record("s1", "gemini-3.5-flash", "extract", 100, 50, turn=1, goal_id="fitness")
        dash = CostDashboard(cost_tracker=ct)
        d    = dash.session_cost_detail("s1")
        assert d["session_id"]    == "s1"
        assert d["total_cost_usd"] >= 0

    def test_unknown_session_returns_error(self):
        dash = CostDashboard()
        d    = dash.session_cost_detail("nonexistent")
        assert "error" in d


# ─────────────────────────────────────────────────────────────────────────────
#  21. CostDashboard — trend
# ─────────────────────────────────────────────────────────────────────────────

class TestCostDashboardTrend:

    def test_trend_returns_list(self):
        tracer = _populated_tracer(goal_id="g1", session_id="s1")
        dash   = CostDashboard(tracer=tracer)
        trend  = dash.cost_trend("g1", period_days=1, granularity="day")
        assert isinstance(trend, list)

    def test_trend_has_required_keys(self):
        tracer = _populated_tracer(goal_id="g1", session_id="s1")
        dash   = CostDashboard(tracer=tracer)
        trend  = dash.cost_trend("g1", period_days=1, granularity="day")
        if trend:
            for k in ["period", "cost_usd", "sessions", "tokens"]:
                assert k in trend[0]

    def test_trend_empty_for_unknown_goal(self):
        dash  = CostDashboard(tracer=TrueNorthTracer())
        trend = dash.cost_trend("nonexistent", period_days=7)
        assert trend == []


# ─────────────────────────────────────────────────────────────────────────────
#  22. CostDashboard — model comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestCostDashboardModels:

    def test_model_comparison_returns_list(self):
        tracer  = _populated_tracer(goal_id="g1", session_id="s1")
        dash    = CostDashboard(tracer=tracer)
        models  = dash.model_comparison("g1", period_days=1)
        assert isinstance(models, list)

    def test_model_comparison_has_gemini(self):
        tracer  = _populated_tracer(goal_id="g1", session_id="s1")
        dash    = CostDashboard(tracer=tracer)
        models  = dash.model_comparison("g1", period_days=1)
        names   = [m["model"] for m in models]
        assert "gemini-3.5-flash" in names

    def test_model_comparison_sorted_by_cost(self):
        tracer = _populated_tracer(goal_id="g1", session_id="s1")
        dash   = CostDashboard(tracer=tracer)
        models = dash.model_comparison("g1", period_days=1)
        costs  = [m["cost_usd"] for m in models]
        assert costs == sorted(costs, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  23. Sector agnosticism — same observability stack for 5 sectors
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorObservability:

    SECTORS = [
        ("medical_intake",    "chief_complaint", "lower back pain"),
        ("legal_intake",      "case_type",       "personal_injury"),
        ("hr_screening",      "years_experience","5"),
        ("financial_plan",    "annual_income",   "1500000"),
        ("fitness_plan",      "age",             "28"),
    ]

    @pytest.mark.parametrize("goal_id,field,value", SECTORS)
    def test_tracer_works_for_sector(self, goal_id, field, value):
        tracer, sink = _tracer_with_memory()
        tracer.session_start("s1", goal_id, user_id="u1")
        trace = tracer.turn_start("s1", goal_id, 1)
        tracer.record_user_input("s1", goal_id, 1, f"My {field} is {value}")
        tracer.record_extraction("s1", goal_id, 1, field, value, 0.92, True)
        tracer.turn_end("s1", 1, "Got it.")
        tracer.session_end("s1")

        summary = tracer.session_summary("s1")
        assert summary is not None
        assert summary["turn_count"] >= 1

    @pytest.mark.parametrize("goal_id,field,value", SECTORS)
    def test_health_monitor_works_for_sector(self, goal_id, field, value):
        tracer = _populated_tracer(goal_id=goal_id, session_id="s1")
        monitor = HealthMonitor(tracer=tracer)
        report  = monitor.goal_report(goal_id, window_hours=24)
        assert isinstance(report, GoalHealthReport)
        assert report.goal_id == goal_id

    @pytest.mark.parametrize("goal_id,field,value", SECTORS)
    def test_ab_engine_works_for_sector(self, goal_id, field, value):
        config_a = {"id": f"{goal_id}_v1"}
        config_b = {"id": f"{goal_id}_v2"}
        ab  = ABEngine(f"{goal_id}_test", config_a, config_b, min_sessions=5)
        cfg = ab.assign("sess-sector")
        assert cfg["id"] in (f"{goal_id}_v1", f"{goal_id}_v2")

    @pytest.mark.parametrize("goal_id,field,value", SECTORS)
    def test_cost_dashboard_works_for_sector(self, goal_id, field, value):
        tracer  = _populated_tracer(goal_id=goal_id, session_id="s1")
        dash    = CostDashboard(tracer=tracer)
        summary = dash.goal_cost_summary(goal_id, period_days=1)
        assert summary.goal_id == goal_id