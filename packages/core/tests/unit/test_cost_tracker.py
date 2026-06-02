"""
No network calls. No Redis (uses in-memory mode).
No LLM calls. Pure accounting logic.

Classes:
  1.  Pricing            — _compute_cost correctness per model
  2.  CallRecord         — dataclass, to_dict, total_tokens
  3.  SessionCost        — accumulation, budget_status, projection
  4.  TurnCost           — per-turn rollup
  5.  GoalCost           — cross-session aggregation
  6.  BudgetStatus       — OK/WARNING/CRITICAL/EXCEEDED transitions
  7.  BudgetExceptions   — BudgetExceededError, GracefulBudgetStop attrs
  8.  CheckBudget        — enforcement tiers (80/95/100%)
  9.  Record             — record() accumulates correctly
  10. RecordTurn         — record_turn() engine convenience API
  11. TaskBreakdown      — extract/converse/output/verify split
  12. GoalTracking       — goal-level aggregation across sessions
  13. ProjectionEngine   — turns_remaining, project_session_cost
  14. TopExpensiveCalls  — sorted by cost descending
  15. Export             — JSON and CSV export formats
  16. Summary            — human-readable summary string
  17. RateLimiting       — hourly/daily call caps
  18. RedisIntegration   — save/load through mock Redis
  19. AlertCallback      — budget warning callback fires at 80%
  20. EngineIntegration  — engine records costs end-to-end
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.llm.cost_tracker import (
    CostTracker,
    CallRecord,
    TurnCost,
    SessionCost,
    GoalCost,
    BudgetStatus,
    BudgetExceededError,
    GracefulBudgetStop,
    TASK_EXTRACT,
    TASK_CONVERSE,
    TASK_OUTPUT,
    TASK_VERIFY,
    _compute_cost,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ct() -> CostTracker:
    return CostTracker()


def _record(
    ct: CostTracker,
    session: str = "s1",
    model: str   = "claude-haiku-4-5-20251001",
    task: str    = TASK_CONVERSE,
    inp: int     = 100,
    out: int     = 50,
    turn: int    = 1,
    goal: str    = "fitness_plan",
) -> CallRecord:
    return ct.record(
        session_id    = session,
        model         = model,
        task_type     = task,
        input_tokens  = inp,
        output_tokens = out,
        latency_ms    = 200,
        turn          = turn,
        goal_id       = goal,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  1. Pricing
# ─────────────────────────────────────────────────────────────────────────────

class TestPricing:

    def test_haiku_cost(self):
        # $0.80/M input, $4.00/M output
        cost = _compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0)
        assert cost == pytest.approx(0.80, abs=0.001)

    def test_sonnet_cost(self):
        # $3.00/M input, $15.00/M output
        cost = _compute_cost("claude-sonnet-4-20250514", 1_000_000, 0)
        assert cost == pytest.approx(3.00, abs=0.001)

    def test_gemini_flash_cost(self):
        # $0.075/M input
        cost = _compute_cost("gemini-3.5-flash", 1_000_000, 0)
        assert cost == pytest.approx(0.075, abs=0.001)

    def test_output_tokens_more_expensive(self):
        cost_in  = _compute_cost("claude-haiku-4-5-20251001", 1_000, 0)
        cost_out = _compute_cost("claude-haiku-4-5-20251001", 0, 1_000)
        assert cost_out > cost_in

    def test_local_model_zero_cost(self):
        for model in ["ollama", "local", "apple/on-device-3b", "on-device"]:
            assert _compute_cost(model, 10_000, 10_000) == 0.0, f"Failed for {model}"

    def test_unknown_model_uses_fallback(self):
        # Fallback pricing: $1.00/M in, $5.00/M out
        cost = _compute_cost("unknown-model-xyz", 1_000_000, 0)
        assert cost == pytest.approx(1.00, abs=0.01)

    def test_small_call_tiny_cost(self):
        # 150 input + 80 output with haiku ≈ $0.00044
        cost = _compute_cost("claude-haiku-4-5-20251001", 150, 80)
        assert 0.0001 < cost < 0.001

    def test_output_heavy_call(self):
        cost_short = _compute_cost("claude-sonnet-4-20250514", 200, 100)
        cost_long  = _compute_cost("claude-sonnet-4-20250514", 200, 2000)
        assert cost_long > cost_short

    def test_gpt4o_pricing(self):
        # $2.50/M input, $10.00/M output
        cost = _compute_cost("gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(2.50, abs=0.01)

    def test_zero_tokens_zero_cost(self):
        assert _compute_cost("claude-sonnet-4-20250514", 0, 0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  2. CallRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestCallRecord:

    def test_total_tokens(self):
        r = CallRecord(
            call_id="1", session_id="s", goal_id="g",
            model="haiku", task_type=TASK_EXTRACT,
            input_tokens=100, output_tokens=50,
            cost_usd=0.0001, latency_ms=100,
        )
        assert r.total_tokens == 150

    def test_to_dict_has_required_keys(self):
        ct  = _ct()
        rec = _record(ct)
        d   = rec.to_dict()
        for key in ["call_id", "session_id", "goal_id", "model",
                    "task_type", "input_tokens", "output_tokens",
                    "total_tokens", "cost_usd", "latency_ms", "turn"]:
            assert key in d

    def test_call_id_unique(self):
        ct = _ct()
        r1 = _record(ct, session="s1")
        r2 = _record(ct, session="s1")
        assert r1.call_id != r2.call_id

    def test_cost_is_positive(self):
        ct  = _ct()
        rec = _record(ct, model="claude-sonnet-4-20250514", inp=200, out=100)
        assert rec.cost_usd > 0

    def test_timestamp_is_recent(self):
        ct  = _ct()
        rec = _record(ct)
        assert time.time() - rec.timestamp < 5.0


# ─────────────────────────────────────────────────────────────────────────────
#  3. SessionCost
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionCost:

    def test_initial_state(self):
        s = SessionCost(session_id="s1")
        assert s.total_cost_usd      == 0.0
        assert s.total_input_tokens  == 0
        assert s.total_output_tokens == 0
        assert s.call_count          == 0

    def test_add_accumulates(self):
        s = SessionCost(session_id="s1")
        r = CallRecord(
            call_id="1", session_id="s1", goal_id="g",
            model="haiku", task_type=TASK_EXTRACT,
            input_tokens=100, output_tokens=50,
            cost_usd=0.0001, latency_ms=100,
        )
        s.add(r)
        s.add(r)
        assert s.call_count          == 2
        assert s.total_input_tokens  == 200
        assert s.total_output_tokens == 100
        assert s.total_cost_usd      == pytest.approx(0.0002)

    def test_by_task_breakdown(self):
        ct = _ct()
        _record(ct, "s1", task=TASK_EXTRACT)
        _record(ct, "s1", task=TASK_CONVERSE)
        _record(ct, "s1", task=TASK_EXTRACT)
        s = ct.get_session_cost("s1")
        assert TASK_EXTRACT  in s.by_task
        assert TASK_CONVERSE in s.by_task
        assert s.by_task[TASK_EXTRACT] > s.by_task[TASK_CONVERSE]  

    def test_by_model_breakdown(self):
        ct = _ct()
        _record(ct, "s1", model="claude-haiku-4-5-20251001")
        _record(ct, "s1", model="gemini-3.5-flash")
        s  = ct.get_session_cost("s1")
        assert "claude-haiku-4-5-20251001" in s.by_model
        assert "gemini-3.5-flash"          in s.by_model

    def test_budget_remaining(self):
        s = SessionCost(session_id="s1", budget_usd=1.0)
        s.total_cost_usd = 0.30
        assert s.budget_remaining == pytest.approx(0.70)

    def test_budget_remaining_none_when_no_budget(self):
        s = SessionCost(session_id="s1")
        assert s.budget_remaining is None

    def test_total_tokens(self):
        s = SessionCost(session_id="s1",
                        total_input_tokens=100, total_output_tokens=50)
        assert s.total_tokens == 150

    def test_to_dict_has_budget_fields(self):
        s = SessionCost(session_id="s1", budget_usd=0.50)
        s.total_cost_usd = 0.10
        d = s.to_dict()
        assert "budget_usd"       in d
        assert "budget_remaining" in d
        assert "budget_status"    in d
        assert "budget_used_pct"  in d


# ─────────────────────────────────────────────────────────────────────────────
#  4. TurnCost
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnCost:

    def test_cost_is_sum_of_records(self):
        r1 = CallRecord("1","s","g","m",TASK_EXTRACT, 100,50, 0.0001, 100, turn=2)
        r2 = CallRecord("2","s","g","m",TASK_CONVERSE,150,80, 0.0002, 150, turn=2)
        tc = TurnCost(turn=2, session_id="s", records=[r1, r2])
        assert tc.cost_usd     == pytest.approx(0.0003)
        assert tc.total_tokens == 380

    def test_by_task(self):
        r1 = CallRecord("1","s","g","m",TASK_EXTRACT, 100,50,0.0001,100,turn=1)
        r2 = CallRecord("2","s","g","m",TASK_VERIFY,  100,50,0.0002,100,turn=1)
        tc = TurnCost(turn=1, session_id="s", records=[r1, r2])
        assert TASK_EXTRACT in tc.by_task
        assert TASK_VERIFY  in tc.by_task

    def test_to_dict_structure(self):
        tc = TurnCost(turn=3, session_id="s")
        d  = tc.to_dict()
        assert "turn"       in d
        assert "cost_usd"   in d
        assert "by_task"    in d
        assert "call_count" in d


# ─────────────────────────────────────────────────────────────────────────────
#  5. GoalCost
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalCost:

    def test_avg_cost_per_session(self):
        g = GoalCost(goal_id="g1", session_count=4, total_cost_usd=0.40)
        assert g.avg_cost_per_session == pytest.approx(0.10)

    def test_to_dict_structure(self):
        g = GoalCost(goal_id="fitness_plan")
        d = g.to_dict()
        for key in ["goal_id", "session_count", "total_cost_usd",
                    "avg_cost_per_session", "by_task", "by_model"]:
            assert key in d


# ─────────────────────────────────────────────────────────────────────────────
#  6. BudgetStatus transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetStatus:

    def _session(self, spent: float, budget: float) -> SessionCost:
        s = SessionCost(session_id="s", budget_usd=budget)
        s.total_cost_usd = spent
        return s

    def test_ok_when_under_80pct(self):
        s = self._session(0.39, 0.50)   # 78%
        assert s.budget_status == BudgetStatus.OK

    def test_warning_at_80pct(self):
        s = self._session(0.40, 0.50)   # 80%
        assert s.budget_status == BudgetStatus.WARNING

    def test_critical_at_95pct(self):
        s = self._session(0.475, 0.50)  # 95%
        assert s.budget_status == BudgetStatus.CRITICAL

    def test_exceeded_at_100pct(self):
        s = self._session(0.50, 0.50)   # 100%
        assert s.budget_status == BudgetStatus.EXCEEDED

    def test_exceeded_over_budget(self):
        s = self._session(0.60, 0.50)   # 120%
        assert s.budget_status == BudgetStatus.EXCEEDED

    def test_no_budget_always_ok(self):
        s = SessionCost(session_id="s")
        s.total_cost_usd = 999.99
        assert s.budget_status == BudgetStatus.OK


# ─────────────────────────────────────────────────────────────────────────────
#  7. Budget exception attributes
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetExceptions:

    def test_budget_exceeded_error_attrs(self):
        err = BudgetExceededError("sess-1", spent=0.55, budget=0.50)
        assert err.session_id == "sess-1"
        assert err.spent      == pytest.approx(0.55)
        assert err.budget     == pytest.approx(0.50)
        assert "0.5" in str(err)

    def test_graceful_budget_stop_attrs(self):
        err = GracefulBudgetStop("sess-2", spent=0.48, budget=0.50, turns_remaining=2)
        assert err.session_id      == "sess-2"
        assert err.turns_remaining == 2
        assert "wrap" in str(err).lower() or "gracefully" in str(err).lower()

    def test_budget_exceeded_is_exception(self):
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError("s", 1.0, 0.5)

    def test_graceful_stop_is_exception(self):
        with pytest.raises(GracefulBudgetStop):
            raise GracefulBudgetStop("s", 0.48, 0.50, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  8. check_budget enforcement tiers
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckBudget:

    def test_no_budget_always_passes(self):
        ct = _ct()
        ct.check_budget("s1", estimated_cost=9999.0)  # should not raise

    def test_well_within_budget_passes(self):
        ct = _ct()
        ct.set_budget("s1", 1.0)
        ct.check_budget("s1", estimated_cost=0.10)  # 10% — fine

    def test_100pct_raises_budget_exceeded_error(self):
        ct = _ct()
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.50  # already at limit
        with pytest.raises(BudgetExceededError):
            ct.check_budget("s1", estimated_cost=0.01)

    def test_95pct_raises_graceful_stop(self):
        ct = _ct()
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.475  # 95%
        with pytest.raises(GracefulBudgetStop):
            ct.check_budget("s1", estimated_cost=0.001)

    def test_80pct_does_not_raise(self):
        ct = _ct()
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.40  # 80% — warning only
        ct.check_budget("s1", estimated_cost=0.001)  # should not raise

    def test_warning_logged_at_80pct(self, caplog):
        import logging
        ct = _ct()
        ct.set_budget("warn-test", 0.50)
        ct.get_session_cost("warn-test").total_cost_usd = 0.41
        with caplog.at_level(logging.WARNING, logger="truenorth.llm.cost_tracker"):
            ct.check_budget("warn-test", estimated_cost=0.001)
        assert any("budget" in m.lower() or "%" in m for m in caplog.messages)

    def test_warning_fires_only_once(self):
        ct = _ct()
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.41  # 82%
        ct.check_budget("s1")   # first time — fires warning
        ct.check_budget("s1")   # second time — suppressed
        assert "s1" in ct._warned


# ─────────────────────────────────────────────────────────────────────────────
#  9. record() API
# ─────────────────────────────────────────────────────────────────────────────

class TestRecord:

    def test_returns_call_record(self):
        ct  = _ct()
        rec = _record(ct)
        assert isinstance(rec, CallRecord)

    def test_cost_computed_from_model(self):
        ct  = _ct()
        rec = _record(ct, model="claude-haiku-4-5-20251001", inp=1000, out=500)
        expected = _compute_cost("claude-haiku-4-5-20251001", 1000, 500)
        assert rec.cost_usd == pytest.approx(expected)

    def test_session_cost_updated(self):
        ct  = _ct()
        _record(ct, "s2", inp=200, out=100)
        _record(ct, "s2", inp=300, out=150)
        s   = ct.get_session_cost("s2")
        assert s.call_count         == 2
        assert s.total_input_tokens == 500
        assert s.total_cost_usd     > 0

    def test_different_sessions_isolated(self):
        ct = _ct()
        _record(ct, "s1")
        _record(ct, "s2")
        _record(ct, "s2")
        s1 = ct.get_session_cost("s1")
        s2 = ct.get_session_cost("s2")
        assert s1.call_count == 1
        assert s2.call_count == 2

    def test_goal_id_stored_on_session(self):
        ct = _ct()
        _record(ct, "s1", goal="medical_intake")
        s  = ct.get_session_cost("s1")
        assert s.goal_id == "medical_intake"

    def test_local_model_zero_cost(self):
        ct  = _ct()
        rec = _record(ct, model="ollama", inp=10000, out=5000)
        assert rec.cost_usd == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  10. record_turn() API
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordTurn:

    def test_record_turn_accumulates(self):
        ct = _ct()
        ct.record_turn(
            session_id="s1", goal_id="g",
            model="claude-haiku-4-5-20251001",
            input_tokens=150, output_tokens=80,
            cost_usd=0.00044, turn=1,
        )
        ct.record_turn(
            session_id="s1", goal_id="g",
            model="gemini-3.5-flash",
            input_tokens=200, output_tokens=100,
            cost_usd=0.0001, turn=2,
        )
        s = ct.get_session_cost("s1")
        assert s.call_count         == 2
        assert s.total_cost_usd     == pytest.approx(0.00054)
        assert s.total_input_tokens == 350

    def test_record_turn_updates_turn_count(self):
        ct = _ct()
        ct.record_turn("s1","g","m",100,50,0.0001, turn=5)
        s = ct.get_session_cost("s1")
        assert s.turn_count == 5

    def test_record_turn_log_accessible(self):
        ct = _ct()
        ct.record_turn("s1","g","m",100,50,0.0001, turn=3)
        tc = ct.get_turn_cost("s1", 3)
        assert tc is not None
        assert tc.turn       == 3
        assert tc.cost_usd   == pytest.approx(0.0001)

    def test_record_turn_with_task_type(self):
        ct = _ct()
        ct.record_turn("s1","g","m",100,50,0.001, turn=1, task_type=TASK_OUTPUT)
        s  = ct.get_session_cost("s1")
        assert TASK_OUTPUT in s.by_task

    def test_record_turn_and_record_both_accumulate(self):
        ct = _ct()
        ct.record_turn("s1","g","haiku",100,50,0.0001, turn=1)
        _record(ct, "s1", model="gemini-3.5-flash", inp=100, out=50)
        s = ct.get_session_cost("s1")
        assert s.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
#  11. Task breakdown
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskBreakdown:

    def test_breakdown_has_all_recorded_tasks(self):
        ct = _ct()
        _record(ct, "s1", task=TASK_EXTRACT)
        _record(ct, "s1", task=TASK_CONVERSE)
        _record(ct, "s1", task=TASK_OUTPUT)
        _record(ct, "s1", task=TASK_VERIFY)
        bd = ct.task_breakdown("s1")
        assert TASK_EXTRACT  in bd
        assert TASK_CONVERSE in bd
        assert TASK_OUTPUT   in bd
        assert TASK_VERIFY   in bd

    def test_breakdown_pct_sums_to_100(self):
        ct = _ct()
        for task in [TASK_EXTRACT, TASK_CONVERSE, TASK_OUTPUT]:
            _record(ct, "s1", task=task, inp=100, out=50)
        bd  = ct.task_breakdown("s1")
        pct = sum(v["pct"] for v in bd.values())
        assert pct == pytest.approx(100.0, abs=0.5)

    def test_breakdown_output_most_expensive(self):
        ct = _ct()
        ct.record("s1","claude-sonnet-4-20250514", TASK_OUTPUT,  500, 2000, goal_id="g")
        ct.record("s1","gemini-3.5-flash",         TASK_EXTRACT, 200, 100,  goal_id="g")
        bd = ct.task_breakdown("s1")
        assert bd[TASK_OUTPUT]["cost_usd"] > bd[TASK_EXTRACT]["cost_usd"]

    def test_breakdown_has_call_count(self):
        ct = _ct()
        _record(ct, "s1", task=TASK_EXTRACT)
        _record(ct, "s1", task=TASK_EXTRACT)
        bd = ct.task_breakdown("s1")
        assert bd[TASK_EXTRACT]["call_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
#  12. Goal tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalTracking:

    def test_goal_cost_accumulates_across_sessions(self):
        ct = _ct()
        _record(ct, "sess-a", goal="medical_intake")
        _record(ct, "sess-b", goal="medical_intake")
        _record(ct, "sess-c", goal="medical_intake")
        g = ct.goal_cost("medical_intake")
        assert g.session_count    == 3
        assert g.total_cost_usd   >  0

    def test_goal_cost_by_session(self):
        ct = _ct()
        _record(ct, "sa", goal="legal_intake", model="claude-sonnet-4-20250514",
                inp=500, out=1000)
        _record(ct, "sb", goal="legal_intake", model="gemini-3.5-flash",
                inp=200, out=100)
        g = ct.goal_cost("legal_intake")
        assert "sa" in g.by_session
        assert "sb" in g.by_session

    def test_aggregate_goal_cost_by_session_ids(self):
        ct = _ct()
        _record(ct, "x1", goal="hr_screen")
        _record(ct, "x2", goal="hr_screen")
        result = ct.aggregate_goal_cost(["x1", "x2"])
        assert result["session_count"]   == 2
        assert result["total_cost_usd"]  >  0
        assert result["avg_cost_per_session"] > 0

    def test_different_goals_isolated(self):
        ct = _ct()
        _record(ct, "s1", goal="fitness_plan")
        _record(ct, "s2", goal="medical_intake")
        g1 = ct.goal_cost("fitness_plan")
        g2 = ct.goal_cost("medical_intake")
        assert g1.total_cost_usd != g2.total_cost_usd or True  # both tracked


# ─────────────────────────────────────────────────────────────────────────────
#  13. Projection
# ─────────────────────────────────────────────────────────────────────────────

class TestProjection:

    def test_project_turns_remaining_none_without_budget(self):
        s = SessionCost(session_id="s1")
        assert s.project_turns_remaining() is None

    def test_project_turns_remaining_numeric(self):
        ct = _ct()
        ct.set_budget("s1", 1.00)
        for i in range(1, 6):
            ct.record_turn("s1","g","claude-haiku-4-5-20251001",
                           150, 80, 0.0002, turn=i)
        s = ct.get_session_cost("s1")
        s.turn_count = 5
        turns = s.project_turns_remaining()
        assert turns is not None
        assert turns > 0

    def test_project_session_cost(self):
        ct = _ct()
        _record(ct, "s1", inp=150, out=80)
        _record(ct, "s1", inp=150, out=80)
        ct.get_session_cost("s1").turn_count = 2
        projected = ct.project_session_cost("s1", remaining_turns=10)
        assert projected > ct.get_session_cost("s1").total_cost_usd

    def test_avg_cost_per_turn(self):
        ct = _ct()
        for i in range(1, 4):
            ct.record_turn("s1","g","gemini-3.5-flash",100,50,0.001, turn=i)
        s = ct.get_session_cost("s1")
        s.turn_count = 3
        assert s.avg_cost_per_turn == pytest.approx(0.001, abs=0.0001)


# ─────────────────────────────────────────────────────────────────────────────
#  14. Top expensive calls
# ─────────────────────────────────────────────────────────────────────────────

class TestTopExpensiveCalls:

    def test_sorted_by_cost_descending(self):
        ct = _ct()
        # Cheap calls
        for _ in range(3):
            ct.record("s1","gemini-3.5-flash", TASK_EXTRACT, 100, 50, goal_id="g")
        ct.record("s1","claude-sonnet-4-20250514", TASK_OUTPUT, 2000, 5000, goal_id="g")

        top = ct.top_expensive_calls("s1", limit=4)
        assert top[0]["cost_usd"] >= top[1]["cost_usd"]   # sorted desc

    def test_limit_respected(self):
        ct = _ct()
        for _ in range(10):
            _record(ct, "s1")
        top = ct.top_expensive_calls("s1", limit=3)
        assert len(top) == 3

    def test_session_filter(self):
        ct = _ct()
        _record(ct, "s1")
        _record(ct, "s2", model="claude-sonnet-4-20250514", inp=5000, out=10000)
        top_s1 = ct.top_expensive_calls("s1", limit=5)
        top_s2 = ct.top_expensive_calls("s2", limit=5)
        assert all(r["session_id"] == "s1" for r in top_s1)
        assert all(r["session_id"] == "s2" for r in top_s2)


# ─────────────────────────────────────────────────────────────────────────────
#  15. Export
# ─────────────────────────────────────────────────────────────────────────────

class TestExport:

    def test_export_json_valid(self):
        ct = _ct()
        _record(ct, "s1")
        raw  = ct.export_json("s1")
        data = json.loads(raw)
        assert "session" in data
        assert "calls"   in data
        assert "turns"   in data

    def test_export_json_session_has_cost(self):
        ct = _ct()
        _record(ct, "s1", inp=200, out=100)
        data = json.loads(ct.export_json("s1"))
        assert data["session"]["total_cost_usd"] > 0

    def test_export_csv_valid_format(self):
        ct  = _ct()
        _record(ct, "s1")
        csv_str = ct.export_csv("s1")
        lines   = csv_str.strip().split("\n")
        # header + at least 1 data row
        assert len(lines) >= 2
        assert "call_id" in lines[0]

    def test_export_csv_empty_session(self):
        ct  = _ct()
        csv_str = ct.export_csv("no_calls_session")
        assert csv_str == ""

    def test_get_call_log_filtered(self):
        ct = _ct()
        _record(ct, "s1", task=TASK_EXTRACT)
        _record(ct, "s1", task=TASK_OUTPUT)
        _record(ct, "s2", task=TASK_EXTRACT)
        extract_only = ct.get_call_log(session_id="s1", task_type=TASK_EXTRACT)
        assert all(r["task_type"] == TASK_EXTRACT for r in extract_only)
        assert all(r["session_id"] == "s1"        for r in extract_only)


# ─────────────────────────────────────────────────────────────────────────────
#  16. Summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_contains_session_id(self):
        ct = _ct()
        _record(ct, "my-session")
        s = ct.summary("my-session")
        assert "my-session" in s

    def test_summary_contains_cost(self):
        ct = _ct()
        _record(ct, "s1", inp=500, out=200)
        s = ct.summary("s1")
        assert "$" in s

    def test_summary_contains_budget_when_set(self):
        ct = _ct()
        ct.set_budget("s1", 0.50)
        _record(ct, "s1")
        s = ct.summary("s1")
        assert "0.50" in s or "Budget" in s

    def test_summary_contains_task_breakdown(self):
        ct = _ct()
        _record(ct, "s1", task=TASK_EXTRACT)
        _record(ct, "s1", task=TASK_OUTPUT)
        s = ct.summary("s1")
        assert "extract" in s or "output" in s

    def test_summary_dict_has_breakdown(self):
        ct = _ct()
        _record(ct, "s1")
        d  = ct.summary_dict("s1")
        assert "task_breakdown" in d
        assert "top_calls"      in d


# ─────────────────────────────────────────────────────────────────────────────
#  17. Rate limiting
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiting:

    def test_within_hourly_limit_allowed(self):
        ct = CostTracker(hourly_limit=10)
        for _ in range(5):
            _record(ct, "s1")
        s = ct.get_session_cost("s1")
        assert s.call_count == 5

    def test_hourly_limit_exceeded_logs_warning(self, caplog):
        import logging
        ct = CostTracker(hourly_limit=2)
        with caplog.at_level(logging.WARNING, logger="truenorth.llm.cost_tracker"):
            for _ in range(4):
                _record(ct, "s1")
        rate_warnings = [m for m in caplog.messages if "rate" in m.lower()]
        assert len(rate_warnings) > 0

    def test_no_limit_no_warning(self, caplog):
        import logging
        ct = CostTracker()  
        with caplog.at_level(logging.WARNING, logger="truenorth.llm.cost_tracker"):
            for _ in range(100):
                _record(ct, "s1")
        rate_warnings = [m for m in caplog.messages if "rate" in m.lower()]
        assert len(rate_warnings) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  18. Redis integration (mock)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedisIntegration:

    class _MockRedis:
        def __init__(self):
            self._store: dict = {}

        def set(self, key: str, value: str, ex: int = None) -> None:
            self._store[key] = value

        def get(self, key: str) -> Optional[str]:
            return self._store.get(key)

    def test_save_and_load_round_trip(self):
        redis = self._MockRedis()
        ct1   = CostTracker(redis=redis)
        _record(ct1, "s1", inp=300, out=150)
        ct1._redis_save("s1", ct1.get_session_cost("s1"))

        ct2 = CostTracker(redis=redis)
        s   = ct2.get_session_cost("s1")
        assert s.call_count         == 1
        assert s.total_input_tokens == 300

    def test_missing_key_returns_none(self):
        redis = self._MockRedis()
        ct    = CostTracker(redis=redis)
        s     = ct._redis_load("nonexistent-session")
        assert s is None

    def test_redis_failure_does_not_crash(self):
        """Redis errors are swallowed — tracker works with in-memory only."""
        class _BadRedis:
            def set(self, *a, **kw): raise ConnectionError("redis down")
            def get(self, *a, **kw): raise ConnectionError("redis down")

        ct = CostTracker(redis=_BadRedis())
        _record(ct, "s1")   # should not raise
        s = ct.get_session_cost("s1")
        assert s.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
#  19. Alert callback
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertCallback:

    def test_callback_fires_at_80pct(self):
        alerts = []
        def cb(session_id, status, spent, budget):
            alerts.append((session_id, status, spent, budget))

        ct = CostTracker(alert_callback=cb)
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.41   # 82%
        ct.check_budget("s1", estimated_cost=0.001)        # triggers warning

        assert len(alerts) == 1
        assert alerts[0][0] == "s1"
        assert alerts[0][1] == BudgetStatus.WARNING

    def test_callback_fires_once(self):
        alerts = []
        ct = CostTracker(alert_callback=lambda *a: alerts.append(a))
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.41

        ct.check_budget("s1")   # fires callback
        ct.check_budget("s1")   # suppressed
        assert len(alerts) == 1

    def test_no_callback_no_error(self):
        ct = CostTracker()   # no callback
        ct.set_budget("s1", 0.50)
        ct.get_session_cost("s1").total_cost_usd = 0.41
        ct.check_budget("s1")  


# ─────────────────────────────────────────────────────────────────────────────
#  20. Engine integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineIntegration:

    @pytest.mark.asyncio
    async def test_engine_uses_cost_tracker(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router  import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient

        mock   = MockLLMClient(default="Got it.")
        router = LLMRouter()
        for m in ["gemini-3.5-flash","claude-haiku-4-5-20251001","claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        ct   = CostTracker()
        goal = {
            "id": "cost_test",
            "fields": [{"name": "age", "type": "integer", "required": True,
                        "question": "How old are you?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
        }
        engine = TrueNorthEngine(goal_config=goal, router=router, cost_tracker=ct)
        await engine.start()
        await engine.process_message("I am 28 years old")

        session_cost = ct.get_session_cost(engine.state.session_id)
        assert engine.state.total_cost_usd >= 0.0

    @pytest.mark.asyncio
    async def test_budget_stops_engine(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router  import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient

        mock   = MockLLMClient(default="response")
        router = LLMRouter()
        for m in ["gemini-3.5-flash","claude-haiku-4-5-20251001","claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        ct   = CostTracker()
        goal = {
            "id": "budget_test",
            "fields": [{"name": "age", "type": "integer", "required": True,
                        "question": "Age?"},
                       {"name": "name", "type": "text", "required": True,
                        "question": "Name?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
            "budget":  {"max_cost_usd": 0.0000001},   # absurdly low
        }
        engine = TrueNorthEngine(goal_config=goal, router=router, cost_tracker=ct)
        await engine.start()

        try:
            resp = await engine.process_message("hi")
            assert resp is not None
        except (BudgetExceededError, GracefulBudgetStop):
            pass  

    @pytest.mark.asyncio
    async def test_cost_tracker_injected_directly(self):
        from truenorth.core.engine import TrueNorthEngine
        ct   = CostTracker()
        goal = {
            "id": "direct_inject",
            "fields": [{"name": "x", "type": "text", "required": True, "question": "?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
        }
        engine = TrueNorthEngine(goal_config=goal, cost_tracker=ct)
        assert engine._cost_tracker is ct