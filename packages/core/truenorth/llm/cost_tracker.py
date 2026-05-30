"""
  ✓ Three-tier budget enforcement: WARNING (80%) → CRITICAL (95%) → EXCEEDED (100%)
  ✓ GracefulBudgetStop at 95% — lets session wrap up cleanly
  ✓ BudgetExceededError at 100% — hard stop
  ✓ Per-turn cost rollup (TurnCost) for the Studio turn-by-turn dashboard
  ✓ Per-goal aggregation across sessions (GoalCost)
  ✓ Task breakdown with % share: extract vs converse vs output vs verify
  ✓ Projection engine: avg_cost_per_turn → turns_remaining → project_session_cost
  ✓ Top-N most expensive calls (sorted by cost_usd desc)
  ✓ JSON + CSV export for analytics pipeline
  ✓ Alert callback: fires once at 80% (never fires twice for the same session)
  ✓ Hourly call rate limiter (warn when exceeded — non-blocking)
  ✓ Redis persistence (save/load round-trip, failure is silently swallowed)
  ✓ record_turn() convenience API for engine Stage 12
  ✓ record() raw API for router-level recording
  ✓ Sector-agnostic: tracks costs for fitness, medical, legal, HR, finance goals
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Task type constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_EXTRACT  = "extract"
TASK_CONVERSE = "converse"
TASK_OUTPUT   = "output"
TASK_CLASSIFY = "classify"
TASK_VERIFY   = "verify"
TASK_OTHER    = "other"

_ALL_TASKS = (TASK_EXTRACT, TASK_CONVERSE, TASK_OUTPUT,
              TASK_CLASSIFY, TASK_VERIFY, TASK_OTHER)


# ─────────────────────────────────────────────────────────────────────────────
#  Pricing table (USD per 1M tokens — updated mid-2025)
# ─────────────────────────────────────────────────────────────────────────────

_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-20250514":     (15.00, 75.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "claude-opus-4-8":            (15.00, 75.00),
    "claude-sonnet-4-20250514":   ( 3.00, 15.00),
    "claude-haiku-4-5-20251001":  ( 0.80,  4.00),
    # Google
    "gemini-1.5-flash":           ( 0.075, 0.30),
    "gemini-1.5-pro":             ( 3.50, 10.50),
    "gemini-2.0-flash":           ( 0.10,  0.40),
    "gemini-2.0-flash-lite":      ( 0.075, 0.30),
    # OpenAI
    "gpt-4o":                     ( 2.50, 10.00),
    "gpt-4o-mini":                ( 0.15,  0.60),
    "gpt-4-turbo":                (10.00, 30.00),
    "o1":                         (15.00, 60.00),
    "o3-mini":                    ( 1.10,  4.40),
    # Cohere
    "command-r":                  ( 0.50,  1.50),
    "command-r-plus":             ( 3.00, 15.00),
    # Groq
    "llama-3.1-70b-versatile":    ( 0.59,  0.79),
    # Local / free
    "ollama":                     ( 0.00,  0.00),
    "local":                      ( 0.00,  0.00),
    "apple/on-device-3b":         ( 0.00,  0.00),
    "gemini-nano":                ( 0.00,  0.00),
    "on-device":                  ( 0.00,  0.00),
}

_FALLBACK_PRICING: Tuple[float, float] = (1.00, 5.00)

# Model prefixes whose full cost is 0 (on-device)
_FREE_PREFIXES = ("ollama", "local", "apple/", "gemini-nano", "on-device",
                  "mobile", "llama-cpp", "lmstudio")


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute USD cost for one LLM call.
    Returns 0.0 for on-device / local models.
    Falls back to conservative estimate for unknown models.
    """
    if not model:
        return 0.0
    base = model.split(":")[0].strip()  # strip provider prefix ("ollama:llama3.1")
    if any(base.startswith(pfx) for pfx in _FREE_PREFIXES):
        return 0.0
    pin, pout = _PRICING.get(base, _FALLBACK_PRICING)
    return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 8)


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class BudgetStatus(str, Enum):
    OK       = "ok"        # < 80% of budget used
    WARNING  = "warning"   # 80–94% — alert, but keep going
    CRITICAL = "critical"  # 95–99% — trigger GracefulBudgetStop
    EXCEEDED = "exceeded"  # ≥ 100% — raise BudgetExceededError


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class BudgetExceededError(Exception):
    """100%+ of budget used — hard stop."""
    def __init__(self, session_id: str, spent: float, budget: float):
        self.session_id = session_id
        self.spent      = spent
        self.budget     = budget
        super().__init__(
            f"Budget ${budget:.4f} exceeded for session={session_id} "
            f"(spent=${spent:.4f})"
        )


class GracefulBudgetStop(Exception):
    """95% of budget reached — graceful stop (finish current turn, then stop)."""
    def __init__(
        self,
        session_id: str,
        spent:      float,
        budget:     float,
        turns_remaining: Optional[int] = None,
    ):
        self.session_id      = session_id
        self.spent           = spent
        self.budget          = budget
        self.turns_remaining = turns_remaining
        super().__init__(
            f"Approaching budget limit — wrapping up session gracefully. "
            f"session={session_id} spent=${spent:.4f}/{budget:.4f} "
            + (f"est. {turns_remaining} turns remaining" if turns_remaining else "")
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    """One completed LLM API call — the atomic unit of cost tracking."""
    call_id:       str
    session_id:    str
    goal_id:       str
    model:         str
    task_type:     str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    int
    turn:          int   = 0
    timestamp:     float = field(default_factory=time.time)
    metadata:      dict  = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "call_id":       self.call_id,
            "session_id":    self.session_id,
            "goal_id":       self.goal_id,
            "model":         self.model,
            "task_type":     self.task_type,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.total_tokens,
            "cost_usd":      round(self.cost_usd, 8),
            "latency_ms":    self.latency_ms,
            "turn":          self.turn,
            "timestamp":     self.timestamp,
        }


@dataclass
class TurnCost:
    """Aggregated cost for one conversation turn (may have multiple LLM calls)."""
    turn:          int
    session_id:    str
    records:       List[CallRecord] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def call_count(self) -> int:
        return len(self.records)

    @property
    def by_task(self) -> Dict[str, float]:
        breakdown: Dict[str, float] = {}
        for r in self.records:
            breakdown[r.task_type] = breakdown.get(r.task_type, 0.0) + r.cost_usd
        return breakdown

    def to_dict(self) -> dict:
        return {
            "turn":         self.turn,
            "session_id":   self.session_id,
            "cost_usd":     round(self.cost_usd, 8),
            "total_tokens": self.total_tokens,
            "call_count":   self.call_count,
            "by_task":      {k: round(v, 8) for k, v in self.by_task.items()},
        }


@dataclass
class SessionCost:
    """Accumulated cost for one conversation session."""
    session_id:          str
    goal_id:             str   = ""
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_cost_usd:      float = 0.0
    call_count:          int   = 0
    turn_count:          int   = 0
    by_task:             Dict[str, float] = field(default_factory=dict)
    by_model:            Dict[str, float] = field(default_factory=dict)
    by_task_calls:       Dict[str, int]   = field(default_factory=dict)
    budget_usd:          Optional[float]  = None
    created_at:          float = field(default_factory=time.time)

    def add(self, record: CallRecord) -> None:
        self.total_input_tokens  += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_cost_usd      += record.cost_usd
        self.call_count          += 1
        if record.goal_id and not self.goal_id:
            self.goal_id = record.goal_id
        self.by_task[record.task_type] = (
            self.by_task.get(record.task_type, 0.0) + record.cost_usd
        )
        self.by_model[record.model] = (
            self.by_model.get(record.model, 0.0) + record.cost_usd
        )
        self.by_task_calls[record.task_type] = (
            self.by_task_calls.get(record.task_type, 0) + 1
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def budget_remaining(self) -> Optional[float]:
        if self.budget_usd is None:
            return None
        return round(max(0.0, self.budget_usd - self.total_cost_usd), 8)

    @property
    def budget_used_pct(self) -> Optional[float]:
        if self.budget_usd is None or self.budget_usd == 0:
            return None
        return round(self.total_cost_usd / self.budget_usd * 100, 2)

    @property
    def budget_exceeded(self) -> bool:
        if self.budget_usd is None:
            return False
        return self.total_cost_usd >= self.budget_usd

    @property
    def budget_status(self) -> BudgetStatus:
        if self.budget_usd is None or self.budget_usd == 0:
            return BudgetStatus.OK
        pct = self.total_cost_usd / self.budget_usd
        if pct >= 1.00:
            return BudgetStatus.EXCEEDED
        if pct >= 0.95:
            return BudgetStatus.CRITICAL
        if pct >= 0.80:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    @property
    def avg_cost_per_turn(self) -> float:
        if self.turn_count == 0:
            return 0.0
        return self.total_cost_usd / self.turn_count

    def project_turns_remaining(self) -> Optional[int]:
        """Estimate turns remaining before budget is exhausted."""
        if self.budget_usd is None or self.budget_remaining is None:
            return None
        if self.avg_cost_per_turn == 0:
            return None
        remaining = self.budget_remaining
        return max(0, int(remaining / self.avg_cost_per_turn))

    def to_dict(self) -> dict:
        pct = self.budget_used_pct
        return {
            "session_id":          self.session_id,
            "goal_id":             self.goal_id,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens":        self.total_tokens,
            "total_cost_usd":      round(self.total_cost_usd, 6),
            "call_count":          self.call_count,
            "turn_count":          self.turn_count,
            "avg_cost_per_turn":   round(self.avg_cost_per_turn, 8),
            "by_task":             {k: round(v, 6) for k, v in self.by_task.items()},
            "by_model":            {k: round(v, 6) for k, v in self.by_model.items()},
            "budget_usd":          self.budget_usd,
            "budget_remaining":    self.budget_remaining,
            "budget_used_pct":     pct,
            "budget_status":       self.budget_status.value,
            "budget_exceeded":     self.budget_exceeded,
        }


@dataclass
class GoalCost:
    """Aggregated cost across all sessions for one goal YAML."""
    goal_id:       str
    session_count: int   = 0
    total_cost_usd: float = 0.0
    total_tokens:  int   = 0
    by_task:       Dict[str, float] = field(default_factory=dict)
    by_model:      Dict[str, float] = field(default_factory=dict)
    by_session:    Dict[str, float] = field(default_factory=dict)  # session_id → cost

    @property
    def avg_cost_per_session(self) -> float:
        return round(self.total_cost_usd / max(self.session_count, 1), 8)

    def to_dict(self) -> dict:
        return {
            "goal_id":              self.goal_id,
            "session_count":        self.session_count,
            "total_cost_usd":       round(self.total_cost_usd, 6),
            "total_tokens":         self.total_tokens,
            "avg_cost_per_session": self.avg_cost_per_session,
            "by_task":              {k: round(v, 6) for k, v in self.by_task.items()},
            "by_model":             {k: round(v, 6) for k, v in self.by_model.items()},
            "by_session":           {k: round(v, 6) for k, v in self.by_session.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CostTracker
# ─────────────────────────────────────────────────────────────────────────────

class CostTracker:
    """
    Production-grade LLM cost tracker for TrueNorth.

    Tracks cost at three levels:
      call     → CallRecord (one LLM request)
      turn     → TurnCost   (all calls in one conversation turn)
      session  → SessionCost (all turns in one conversation)
      goal     → GoalCost   (all sessions for one goal YAML)

    Budget enforcement tiers:
      80%  → WARNING logged + alert_callback fired (once per session)
      95%  → GracefulBudgetStop raised (let current turn finish, then stop)
      100% → BudgetExceededError raised (hard stop)

    Sector-agnostic: same tracker for healthcare, legal, HR, finance, fitness.
    The goal_id field distinguishes cost by domain.
    """

    REDIS_KEY_PREFIX = "tn:cost:"
    REDIS_TTL        = 86_400 * 3   # 3 days

    BUDGET_WARNING_PCT  = 0.80
    BUDGET_CRITICAL_PCT = 0.95

    def __init__(
        self,
        redis:          Any                                              = None,
        alert_callback: Optional[Callable[[str, BudgetStatus, float, float], None]] = None,
        hourly_limit:   Optional[int]                                    = None,
    ):
        self._redis          = redis
        self._alert_callback = alert_callback
        self._hourly_limit   = hourly_limit

        # In-memory stores
        self._sessions:  Dict[str, SessionCost]        = {}
        self._turns:     Dict[str, Dict[int, TurnCost]] = {}   # session_id → {turn: TurnCost}
        self._call_log:  List[CallRecord]               = []
        self._goal_map:  Dict[str, List[str]]           = {}   # goal_id → [session_id, ...]
        self._warned:    set                            = set() # sessions with 80% warning fired

        # Rate limiting: {hour_key: call_count}
        self._hourly_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Budget configuration
    # ------------------------------------------------------------------

    def set_budget(self, session_id: str, budget_usd: float) -> None:
        """Set a USD cost cap for a session."""
        s = self._get_or_create(session_id)
        s.budget_usd = budget_usd
        logger.info("cost_tracker: session=%s budget=$%.4f", session_id, budget_usd)

    def check_budget(
        self,
        session_id:     str,
        estimated_cost: float = 0.0,
    ) -> None:
        """
        Check if this session is within budget.

        Raises:
          GracefulBudgetStop  — if ≥ 95% used (warn caller to finish up)
          BudgetExceededError — if ≥ 100% used (hard stop)

        Logs WARNING + fires alert_callback at 80% (once per session).
        """
        s = self._get_or_create(session_id)
        if s.budget_usd is None:
            return

        projected = s.total_cost_usd + estimated_cost
        # budget_usd=0 means zero spending allowed — any projected cost exceeds it
        if s.budget_usd == 0:
            if projected > 0 or s.total_cost_usd > 0:
                raise BudgetExceededError(
                    session_id = session_id,
                    spent      = s.total_cost_usd,
                    budget     = s.budget_usd,
                )
            return
        pct       = projected / s.budget_usd

        # 100% → hard stop
        if pct >= 1.0:
            raise BudgetExceededError(
                session_id = session_id,
                spent      = s.total_cost_usd,
                budget     = s.budget_usd,
            )

        # 95% → graceful stop
        if pct >= self.BUDGET_CRITICAL_PCT:
            turns_remaining = s.project_turns_remaining()
            raise GracefulBudgetStop(
                session_id      = session_id,
                spent           = s.total_cost_usd,
                budget          = s.budget_usd,
                turns_remaining = turns_remaining,
            )

        # 80% → warning (fire once per session)
        if pct >= self.BUDGET_WARNING_PCT and session_id not in self._warned:
            self._warned.add(session_id)
            logger.warning(
                "cost_tracker: budget WARNING session=%s %.0f%% used ($%.4f / $%.4f)",
                session_id, pct * 100, s.total_cost_usd, s.budget_usd,
            )
            if self._alert_callback:
                try:
                    self._alert_callback(
                        session_id, BudgetStatus.WARNING,
                        s.total_cost_usd, s.budget_usd,
                    )
                except Exception as e:
                    logger.debug("cost_tracker: alert_callback error: %s", e)

    # ------------------------------------------------------------------
    # Core recording API
    # ------------------------------------------------------------------

    def record(
        self,
        session_id:    str,
        model:         str,
        task_type:     str,
        input_tokens:  int,
        output_tokens: int,
        latency_ms:    int   = 0,
        turn:          int   = 0,
        goal_id:       str   = "",
        metadata:      Optional[dict] = None,
    ) -> CallRecord:
        """
        Record a completed LLM call. Cost is computed from the model pricing table.
        This is the canonical recording method — all cost flows through here.
        """
        cost_usd = _compute_cost(model, input_tokens, output_tokens)
        call_id  = str(uuid.uuid4())[:16]

        record = CallRecord(
            call_id       = call_id,
            session_id    = session_id,
            goal_id       = goal_id,
            model         = model,
            task_type     = task_type,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            cost_usd      = cost_usd,
            latency_ms    = latency_ms,
            turn          = turn,
            metadata      = metadata or {},
        )

        s = self._get_or_create(session_id)
        s.add(record)

        if turn > 0:
            if session_id not in self._turns:
                self._turns[session_id] = {}
            tc = self._turns[session_id].get(turn)
            if tc is None:
                tc = TurnCost(turn=turn, session_id=session_id)
                self._turns[session_id][turn] = tc
            tc.records.append(record)

        self._call_log.append(record)

        if goal_id:
            if goal_id not in self._goal_map:
                self._goal_map[goal_id] = []
            if session_id not in self._goal_map[goal_id]:
                self._goal_map[goal_id].append(session_id)

        self._check_rate_limit(session_id)

        self._redis_save(session_id, s)

        logger.info(
            "cost_tracker: session=%s model=%s task=%s "
            "in=%d out=%d cost=$%.6f latency=%dms turn=%d",
            session_id, model, task_type,
            input_tokens, output_tokens, cost_usd, latency_ms, turn,
        )
        return record

    def record_turn(
        self,
        session_id:    str,
        goal_id:       str,
        model:         str,
        input_tokens:  int,
        output_tokens: int,
        cost_usd:      float,
        turn:          int,
        task_type:     str  = TASK_CONVERSE,
        latency_ms:    int  = 0,
    ) -> CallRecord:
        """
        Engine Stage 12 convenience method.
        Uses the pre-computed cost_usd from the router (avoids double computation).
        Still delegates to record() for unified accumulation.

        The engine calls this with cost_usd already computed by the router.
        We store the record with that cost, overriding the pricing table lookup.
        """
        call_id = str(uuid.uuid4())[:16]
        record  = CallRecord(
            call_id       = call_id,
            session_id    = session_id,
            goal_id       = goal_id,
            model         = model,
            task_type     = task_type,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            cost_usd      = cost_usd,
            latency_ms    = latency_ms,
            turn          = turn,
        )

        # Accumulate into session
        s = self._get_or_create(session_id)
        s.add(record)
        s.turn_count = max(s.turn_count, turn)

        if turn > 0:
            if session_id not in self._turns:
                self._turns[session_id] = {}
            tc = self._turns[session_id].get(turn)
            if tc is None:
                tc = TurnCost(turn=turn, session_id=session_id)
                self._turns[session_id][turn] = tc
            tc.records.append(record)

        self._call_log.append(record)

        if goal_id:
            if goal_id not in self._goal_map:
                self._goal_map[goal_id] = []
            if session_id not in self._goal_map[goal_id]:
                self._goal_map[goal_id].append(session_id)

        self._check_rate_limit(session_id)
        self._redis_save(session_id, s)

        logger.info(
            "cost_tracker: turn=%d session=%s model=%s cost=$%.6f",
            turn, session_id, model, cost_usd,
        )
        return record

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_session_cost(self, session_id: str) -> SessionCost:
        """Return accumulated cost for a session."""
        if session_id not in self._sessions:
            loaded = self._redis_load(session_id)
            self._sessions[session_id] = (
                loaded or SessionCost(session_id=session_id)
            )
        return self._sessions[session_id]

    def get_turn_cost(self, session_id: str, turn: int) -> Optional[TurnCost]:
        """Return cost for a specific turn in a session."""
        return self._turns.get(session_id, {}).get(turn)

    def get_all_turns(self, session_id: str) -> List[TurnCost]:
        """Return all TurnCost objects for a session, sorted by turn number."""
        turns = self._turns.get(session_id, {})
        return sorted(turns.values(), key=lambda t: t.turn)

    def get_call_log(
        self,
        session_id: Optional[str] = None,
        task_type:  Optional[str] = None,
        limit:      int           = 0,
    ) -> List[dict]:
        """
        Return call log records as dicts.
        Optionally filter by session_id and/or task_type.
        limit=0 means no limit.
        """
        records = self._call_log
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        if task_type:
            records = [r for r in records if r.task_type == task_type]
        if limit > 0:
            records = records[-limit:]
        return [r.to_dict() for r in records]

    def goal_cost(self, goal_id: str) -> GoalCost:
        """Aggregate cost for all sessions of a goal."""
        session_ids = self._goal_map.get(goal_id, [])
        gc = GoalCost(goal_id=goal_id)
        for sid in session_ids:
            s = self.get_session_cost(sid)
            gc.session_count  += 1
            gc.total_cost_usd += s.total_cost_usd
            gc.total_tokens   += s.total_tokens
            gc.by_session[sid] = s.total_cost_usd
            for task, cost in s.by_task.items():
                gc.by_task[task] = gc.by_task.get(task, 0.0) + cost
            for model, cost in s.by_model.items():
                gc.by_model[model] = gc.by_model.get(model, 0.0) + cost
        return gc

    def aggregate_goal_cost(self, session_ids: List[str]) -> dict:
        """Aggregate cost across a given list of session IDs (legacy API)."""
        total_cost   = 0.0
        total_tokens = 0
        by_model:    Dict[str, float] = {}
        by_task:     Dict[str, float] = {}

        for sid in session_ids:
            c = self.get_session_cost(sid)
            total_cost   += c.total_cost_usd
            total_tokens += c.total_tokens
            for model, cost in c.by_model.items():
                by_model[model] = by_model.get(model, 0.0) + cost
            for task, cost in c.by_task.items():
                by_task[task] = by_task.get(task, 0.0) + cost

        return {
            "session_count":        len(session_ids),
            "total_cost_usd":       round(total_cost, 6),
            "total_tokens":         total_tokens,
            "by_model":             {k: round(v, 6) for k, v in by_model.items()},
            "by_task":              {k: round(v, 6) for k, v in by_task.items()},
            "avg_cost_per_session": round(total_cost / max(len(session_ids), 1), 6),
        }

    def task_breakdown(self, session_id: str) -> Dict[str, dict]:
        """
        Per-task cost breakdown with % share.
        Returns {task_type: {cost_usd, call_count, pct}}.
        """
        s = self.get_session_cost(session_id)
        if not s.by_task:
            return {}

        total = s.total_cost_usd or 1e-10
        result: Dict[str, dict] = {}
        for task, cost in s.by_task.items():
            result[task] = {
                "cost_usd":   round(cost, 8),
                "call_count": s.by_task_calls.get(task, 0),
                "pct":        round(cost / total * 100, 2),
            }
        return result

    def top_expensive_calls(
        self,
        session_id: Optional[str] = None,
        limit:      int           = 10,
    ) -> List[dict]:
        """Return the N most expensive calls, sorted by cost_usd descending."""
        records = self._call_log
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        sorted_records = sorted(records, key=lambda r: r.cost_usd, reverse=True)
        return [r.to_dict() for r in sorted_records[:limit]]

    def project_session_cost(self, session_id: str, remaining_turns: int) -> float:
        """Estimate total cost if session continues for N more turns."""
        s = self.get_session_cost(session_id)
        avg = s.avg_cost_per_turn
        return round(s.total_cost_usd + avg * remaining_turns, 8)

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a hypothetical call without recording it."""
        return _compute_cost(model, input_tokens, output_tokens)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, session_id: str) -> str:
        """Export session cost data as JSON string."""
        s     = self.get_session_cost(session_id)
        turns = [tc.to_dict() for tc in self.get_all_turns(session_id)]
        calls = self.get_call_log(session_id=session_id)
        data  = {
            "session":  s.to_dict(),
            "turns":    turns,
            "calls":    calls,
            "exported_at": time.time(),
        }
        return json.dumps(data, default=str, indent=2)

    def export_csv(self, session_id: str) -> str:
        """Export call log as CSV string. Returns empty string if no calls."""
        calls = self.get_call_log(session_id=session_id)
        if not calls:
            return ""
        buf    = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=calls[0].keys())
        writer.writeheader()
        writer.writerows(calls)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Human-readable summaries
    # ------------------------------------------------------------------

    def summary(self, session_id: str) -> str:
        """Human-readable cost summary string for CLI / dashboard display."""
        s   = self.get_session_cost(session_id)
        bd  = self.task_breakdown(session_id)

        lines = [
            f"Cost Summary — session: {session_id}",
            f"  Total:     ${s.total_cost_usd:.4f}  ({s.total_tokens:,} tokens, "
            f"{s.call_count} calls, {s.turn_count} turns)",
        ]
        if s.budget_usd:
            pct = s.budget_used_pct or 0
            lines.append(
                f"  Budget:    ${s.total_cost_usd:.4f} / ${s.budget_usd:.4f} "
                f"({pct:.1f}%) — {s.budget_status.value}"
            )
            remaining = s.project_turns_remaining()
            if remaining is not None:
                lines.append(f"  Est. turns remaining: {remaining}")

        if bd:
            lines.append("  By task:")
            for task in sorted(bd, key=lambda t: bd[t]["cost_usd"], reverse=True):
                v = bd[task]
                lines.append(
                    f"    {task:<12} ${v['cost_usd']:.4f}  ({v['pct']:.1f}%)"
                    f"  [{v['call_count']} calls]"
                )
        return "\n".join(lines)

    def summary_dict(self, session_id: str) -> dict:
        """Structured summary dict for Studio dashboard."""
        s   = self.get_session_cost(session_id)
        top = self.top_expensive_calls(session_id, limit=5)
        bd  = self.task_breakdown(session_id)
        return {
            "session":        s.to_dict(),
            "task_breakdown": bd,
            "top_calls":      top,
            "turns":          [tc.to_dict() for tc in self.get_all_turns(session_id)],
        }

    # ------------------------------------------------------------------
    # Rate limiting (soft — warns, does not block)
    # ------------------------------------------------------------------

    def _check_rate_limit(self, session_id: str) -> None:
        if not self._hourly_limit:
            return
        hour_key = f"{session_id}:{int(time.time() // 3600)}"
        count    = self._hourly_counts.get(hour_key, 0) + 1
        self._hourly_counts[hour_key] = count
        if count > self._hourly_limit:
            logger.warning(
                "cost_tracker: rate limit exceeded session=%s calls=%d limit=%d/hr",
                session_id, count, self._hourly_limit,
            )

    # ------------------------------------------------------------------
    # In-memory helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, session_id: str) -> SessionCost:
        if session_id not in self._sessions:
            loaded = self._redis_load(session_id)
            self._sessions[session_id] = (
                loaded or SessionCost(session_id=session_id)
            )
        return self._sessions[session_id]

    # ------------------------------------------------------------------
    # Redis persistence (best-effort — failures are swallowed)
    # ------------------------------------------------------------------

    def _redis_save(self, session_id: str, cost: SessionCost) -> None:
        if not self._redis:
            return
        try:
            key = f"{self.REDIS_KEY_PREFIX}{session_id}"
            if hasattr(self._redis, "set"):
                self._redis.set(key, json.dumps(cost.to_dict()), ex=self.REDIS_TTL)
        except Exception as e:
            logger.debug("cost_tracker: redis_save failed: %s", e)

    def _redis_load(self, session_id: str) -> Optional[SessionCost]:
        if not self._redis:
            return None
        try:
            key = f"{self.REDIS_KEY_PREFIX}{session_id}"
            raw = self._redis.get(key) if hasattr(self._redis, "get") else None
            if not raw:
                return None
            data = json.loads(raw)
            s = SessionCost(
                session_id          = data["session_id"],
                goal_id             = data.get("goal_id", ""),
                total_input_tokens  = data["total_input_tokens"],
                total_output_tokens = data["total_output_tokens"],
                total_cost_usd      = data["total_cost_usd"],
                call_count          = data["call_count"],
                turn_count          = data.get("turn_count", 0),
                by_task             = data.get("by_task", {}),
                by_model            = data.get("by_model", {}),
                by_task_calls       = data.get("by_task_calls", {}),
                budget_usd          = data.get("budget_usd"),
            )
            return s
        except Exception as e:
            logger.debug("cost_tracker: redis_load failed: %s", e)
            return None