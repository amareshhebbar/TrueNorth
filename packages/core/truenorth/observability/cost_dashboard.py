"""
CostDashboard — cost analytics API for the TrueNorth Studio.

Exposes programmatic cost analytics that the Studio dashboard
and CLI both consume. Designed so the same functions power both
a REST endpoint and the truenorth cost CLI command.

Core queries:
  goal_cost_summary(goal_id, period_days)
      → per-model, per-task cost breakdown for a goal

  session_cost_detail(session_id)
      → turn-by-turn cost for one session

  top_expensive_sessions(goal_id, limit, period_days)
      → which sessions cost the most

  cost_trend(goal_id, period_days, granularity)
      → daily/hourly cost time series

  model_comparison(goal_id, period_days)
      → cost per model, avg latency, error rate

  budget_status(session_id or goal_id)
      → how much budget remains

URL patterns (when mounted):
    GET /analytics/cost?goal=fitness_plan&period=7
    GET /analytics/cost/session/{session_id}
    GET /analytics/cost/top?goal=fitness_plan&limit=10
    GET /analytics/cost/trend?goal=fitness_plan&period=30&granularity=day
    GET /analytics/cost/models?goal=fitness_plan&period=7
    GET /analytics/budget/{session_id}
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.observability.tracer import TrueNorthTracer
    from truenorth.llm.cost_tracker     import CostTracker

@dataclass
class CostSummary:
    """Cost summary for a goal or session over a time window."""
    goal_id:        str
    period_days:    int
    session_count:  int
    total_cost_usd: float
    total_tokens:   int
    avg_cost_per_session: float
    by_model:       Dict[str, dict]
    by_task:        Dict[str, dict]
    generated_at:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "goal_id":              self.goal_id,
            "period_days":          self.period_days,
            "session_count":        self.session_count,
            "total_cost_usd":       round(self.total_cost_usd, 6),
            "total_tokens":         self.total_tokens,
            "avg_cost_per_session": round(self.avg_cost_per_session, 6),
            "by_model":             self.by_model,
            "by_task":              self.by_task,
            "generated_at":         self.generated_at,
        }

class CostDashboard:
    """
    Cost analytics API for TrueNorth Studio and CLI.

    Aggregates cost data from the CostTracker (per-session records)
    and TrueNorthTracer (per-turn LLM call events) to produce
    rich analytics for operators.
    """

    def __init__(
        self,
        cost_tracker: Optional["CostTracker"] = None,
        tracer:       Optional["TrueNorthTracer"] = None,
    ):
        self._ct     = cost_tracker
        self._tracer = tracer

    def goal_cost_summary(
        self,
        goal_id:     str,
        period_days: int = 7,
    ) -> CostSummary:
        """
        Aggregate cost for all sessions of a goal over the last N days.
        """
        cutoff    = time.time() - period_days * 86400
        llm_calls = self._calls_for_goal(goal_id, cutoff)

        if not llm_calls:
            return CostSummary(
                goal_id=goal_id, period_days=period_days,
                session_count=0, total_cost_usd=0, total_tokens=0,
                avg_cost_per_session=0, by_model={}, by_task={},
            )

        session_ids    = set(c["session_id"] for c in llm_calls)
        total_cost     = sum(c["cost_usd"] for c in llm_calls)
        total_tokens   = sum(c.get("total_tokens", 0) for c in llm_calls)
        avg_cost       = total_cost / len(session_ids)

        by_model = self._aggregate_by_model(llm_calls)
        by_task  = self._aggregate_by_task(llm_calls, total_cost)

        return CostSummary(
            goal_id             = goal_id,
            period_days         = period_days,
            session_count       = len(session_ids),
            total_cost_usd      = total_cost,
            total_tokens        = total_tokens,
            avg_cost_per_session = avg_cost,
            by_model            = by_model,
            by_task             = by_task,
        )

    def session_cost_detail(self, session_id: str) -> dict:
        """
        Full cost breakdown for one session: per-turn, per-model, per-task.
        """
        if self._ct:
            session = self._ct.get_session_cost(session_id)
            turns   = self._ct.get_all_turns(session_id)
            top     = self._ct.top_expensive_calls(session_id, limit=10)
            bd      = self._ct.task_breakdown(session_id)
            return {
                "session_id":     session_id,
                "total_cost_usd": round(session.total_cost_usd, 6),
                "total_tokens":   session.total_tokens,
                "turn_count":     session.turn_count,
                "budget_status":  session.budget_status.value if session.budget_usd else None,
                "budget_used_pct":session.budget_used_pct,
                "task_breakdown": bd,
                "turns":          [t.to_dict() for t in turns],
                "top_calls":      top,
            }

        if self._tracer:
            sess = self._tracer.get_session(session_id)
            if sess:
                return {
                    "session_id":     session_id,
                    "total_cost_usd": round(sess.total_cost_usd, 6),
                    "total_tokens":   sess.total_tokens,
                    "turn_count":     sess.turn_count,
                    "turns":          [t.to_dict() for t in sess.turns],
                }

        return {"session_id": session_id, "error": "no data found"}

    def top_expensive_sessions(
        self,
        goal_id:     str,
        limit:       int = 10,
        period_days: int = 7,
    ) -> List[dict]:
        """Return the N most expensive sessions for a goal."""
        cutoff = time.time() - period_days * 86400
        calls  = self._calls_for_goal(goal_id, cutoff)

        by_session: Dict[str, float] = defaultdict(float)
        for c in calls:
            by_session[c["session_id"]] += c["cost_usd"]

        sorted_sessions = sorted(
            by_session.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {"session_id": sid, "cost_usd": round(cost, 6)}
            for sid, cost in sorted_sessions[:limit]
        ]

    def cost_trend(
        self,
        goal_id:     str,
        period_days: int = 30,
        granularity: str = "day",
    ) -> List[dict]:
        """
        Time series of cost. Returns [{period, cost_usd, sessions, tokens}].
        """
        cutoff = time.time() - period_days * 86400
        calls  = self._calls_for_goal(goal_id, cutoff)

        if granularity == "hour":
            bucket_s = 3600
            fmt      = "%Y-%m-%dT%H:00"
        else:
            bucket_s = 86400
            fmt      = "%Y-%m-%d"

        import datetime
        buckets: Dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "sessions": set(), "tokens": 0})
        for c in calls:
            ts     = c.get("timestamp", time.time())
            dt     = datetime.datetime.utcfromtimestamp(ts)
            period = dt.strftime(fmt)
            b      = buckets[period]
            b["cost_usd"] += c["cost_usd"]
            b["tokens"]   += c.get("total_tokens", 0)
            b["sessions"].add(c["session_id"])

        return [
            {
                "period":    period,
                "cost_usd":  round(d["cost_usd"], 6),
                "sessions":  len(d["sessions"]),
                "tokens":    d["tokens"],
            }
            for period, d in sorted(buckets.items())
        ]

    def model_comparison(
        self,
        goal_id:     str,
        period_days: int = 7,
    ) -> List[dict]:
        """
        Per-model cost stats for a goal: cost, calls, avg_latency, error_rate.
        """
        cutoff = time.time() - period_days * 86400
        calls  = self._calls_for_goal(goal_id, cutoff)

        models: Dict[str, dict] = defaultdict(lambda: {
            "calls": 0, "cost_usd": 0.0, "tokens": 0, "latency_total": 0,
        })
        for c in calls:
            m = models[c.get("model", "unknown")]
            m["calls"]         += 1
            m["cost_usd"]      += c["cost_usd"]
            m["tokens"]        += c.get("total_tokens", 0)
            m["latency_total"] += c.get("latency_ms", 0)

        result = []
        for model, m in sorted(models.items(), key=lambda x: x[1]["cost_usd"], reverse=True):
            result.append({
                "model":         model,
                "calls":         m["calls"],
                "cost_usd":      round(m["cost_usd"], 6),
                "total_tokens":  m["tokens"],
                "avg_latency_ms":int(m["latency_total"] / max(m["calls"], 1)),
            })
        return result

    def budget_status(self, session_id: str) -> dict:
        """Return current budget status for a session."""
        if not self._ct:
            return {"session_id": session_id, "error": "no cost_tracker configured"}
        session = self._ct.get_session_cost(session_id)
        return {
            "session_id":      session_id,
            "total_cost_usd":  round(session.total_cost_usd, 6),
            "budget_usd":      session.budget_usd,
            "budget_remaining":session.budget_remaining,
            "budget_used_pct": session.budget_used_pct,
            "budget_status":   session.budget_status.value,
            "turns_remaining": session.project_turns_remaining(),
        }

    def make_fastapi_router(self, prefix: str = "/analytics"):
        """
        Create a FastAPI router exposing all dashboard endpoints.

        Mount with: app.include_router(dashboard.make_fastapi_router())
        """
        try:
            from fastapi import APIRouter, Query
            from fastapi.responses import JSONResponse
        except ImportError:
            raise RuntimeError("fastapi not installed — pip install fastapi")

        router = APIRouter(prefix=prefix, tags=["cost-analytics"])

        @router.get("/cost")
        async def goal_cost(
            goal:   str = Query(..., description="Goal ID"),
            period: int = Query(7,   description="Days"),
        ):
            return JSONResponse(self.goal_cost_summary(goal, period).to_dict())

        @router.get("/cost/session/{session_id}")
        async def session_cost(session_id: str):
            return JSONResponse(self.session_cost_detail(session_id))

        @router.get("/cost/top")
        async def top_sessions(
            goal:   str = Query(...),
            limit:  int = Query(10),
            period: int = Query(7),
        ):
            return JSONResponse(self.top_expensive_sessions(goal, limit, period))

        @router.get("/cost/trend")
        async def trend(
            goal:        str = Query(...),
            period:      int = Query(30),
            granularity: str = Query("day"),
        ):
            return JSONResponse(self.cost_trend(goal, period, granularity))

        @router.get("/cost/models")
        async def models(
            goal:   str = Query(...),
            period: int = Query(7),
        ):
            return JSONResponse(self.model_comparison(goal, period))

        @router.get("/budget/{session_id}")
        async def budget(session_id: str):
            return JSONResponse(self.budget_status(session_id))

        return router

    def _calls_for_goal(self, goal_id: str, since: float) -> List[dict]:
        """Collect all LLM call records for a goal from the tracer."""
        if not self._tracer:
            return []

        result = []
        for session_id in self._tracer.all_session_ids():
            sess = self._tracer.get_session(session_id)
            if not sess or sess.goal_id != goal_id:
                continue
            if sess.started_at < since:
                continue
            for turn in sess.turns:
                for call in turn.llm_calls:
                    result.append({
                        "session_id":   session_id,
                        "turn":         turn.turn,
                        "timestamp":    turn.started_at,
                        "model":        call.get("model", ""),
                        "task_type":    call.get("task", ""),
                        "input_tokens": call.get("input_tokens", 0),
                        "output_tokens":call.get("output_tokens", 0),
                        "total_tokens": call.get("total_tokens", 0),
                        "cost_usd":     call.get("cost_usd", 0.0),
                        "latency_ms":   call.get("latency_ms", 0),
                    })
        return result

    @staticmethod
    def _aggregate_by_model(calls: List[dict]) -> Dict[str, dict]:
        models: Dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "tokens": 0, "calls": 0})
        for c in calls:
            m = models[c.get("model", "unknown")]
            m["cost_usd"] += c["cost_usd"]
            m["tokens"]   += c.get("total_tokens", 0)
            m["calls"]    += 1
        return {
            model: {
                "cost_usd": round(m["cost_usd"], 6),
                "tokens":   m["tokens"],
                "calls":    m["calls"],
            }
            for model, m in sorted(models.items(), key=lambda x: x[1]["cost_usd"], reverse=True)
        }

    @staticmethod
    def _aggregate_by_task(calls: List[dict], total: float) -> Dict[str, dict]:
        tasks: Dict[str, dict] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0})
        for c in calls:
            t = tasks[c.get("task_type", "other")]
            t["cost_usd"] += c["cost_usd"]
            t["calls"]    += 1
        return {
            task: {
                "cost_usd": round(t["cost_usd"], 6),
                "calls":    t["calls"],
                "pct":      round(t["cost_usd"] / max(total, 1e-10) * 100, 2),
            }
            for task, t in sorted(tasks.items(), key=lambda x: x[1]["cost_usd"], reverse=True)
        }
