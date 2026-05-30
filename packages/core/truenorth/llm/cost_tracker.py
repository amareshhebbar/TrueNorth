"""
truenorth/llm/cost_tracker.py

Tracks LLM token usage and USD cost at three levels of granularity:
  - Per LLM call (individual request)
  - Per session  (all calls in one conversation)
  - Per goal     (aggregate across all sessions of a goal YAML)

Pricing lives in pricing.py — this module is purely accounting.

Key features:
  - Budget cap enforcement (raises BudgetExceededError)
  - Cost breakdown by task type (extract / converse / output / other)
  - Redis-backed per-session accumulator (survives process restart)
  - Structured log line per call (feeds Observability layer)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing table (as of mid-2025 — update in pricing.py when rates change)
# ---------------------------------------------------------------------------

_PRICING: Dict[str, tuple[float, float]] = {

    "claude-opus-4-20250514":     (15.00, 75.00),
    "claude-sonnet-4-20250514":   ( 3.00, 15.00),
    "claude-haiku-4-5-20251001":  ( 0.80,  4.00),

    "gemini-1.5-flash":           ( 0.075, 0.30),
    "gemini-1.5-pro":             ( 3.50,  10.50),
    "gemini-2.0-flash":           ( 0.10,  0.40),

    "gpt-4o":                     ( 2.50,  10.00),
    "gpt-4o-mini":                ( 0.15,   0.60),
    "gpt-4-turbo":                (10.00,  30.00),
  
    "ollama":                     ( 0.00,   0.00),
    "local":                      ( 0.00,   0.00),
}

_FALLBACK_PRICING = (1.00, 5.00) 

TASK_EXTRACT   = "extract"
TASK_CONVERSE  = "converse"
TASK_OUTPUT    = "output"
TASK_CLASSIFY  = "classify"
TASK_OTHER     = "other"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """One LLM API call."""
    call_id:       str
    session_id:    str
    model:         str
    task_type:     str           # extract / converse / output / classify / other
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    int
    timestamp:     float = field(default_factory=time.time)
    metadata:      dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionCost:
    session_id:         str
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_cost_usd:      float = 0.0
    by_task:             dict  = field(default_factory=dict)  # task_type → cost_usd
    by_model:            dict  = field(default_factory=dict)  # model     → cost_usd
    call_count:          int   = 0
    budget_usd:          Optional[float] = None  # None = no cap

    def add(self, record: CallRecord) -> None:
        self.total_input_tokens  += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_cost_usd      += record.cost_usd
        self.call_count          += 1
        self.by_task[record.task_type] = (
            self.by_task.get(record.task_type, 0.0) + record.cost_usd
        )
        self.by_model[record.model] = (
            self.by_model.get(record.model, 0.0) + record.cost_usd
        )

    @property
    def budget_remaining(self) -> Optional[float]:
        if self.budget_usd is None:
            return None
        return max(0.0, self.budget_usd - self.total_cost_usd)

    @property
    def budget_exceeded(self) -> bool:
        if self.budget_usd is None:
            return False
        return self.total_cost_usd >= self.budget_usd

    def to_dict(self) -> dict:
        return {
            "session_id":          self.session_id,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd":      round(self.total_cost_usd, 6),
            "by_task":             {k: round(v, 6) for k, v in self.by_task.items()},
            "by_model":            {k: round(v, 6) for k, v in self.by_model.items()},
            "call_count":          self.call_count,
            "budget_usd":          self.budget_usd,
            "budget_remaining":    round(self.budget_remaining, 6) if self.budget_remaining is not None else None,
            "budget_exceeded":     self.budget_exceeded,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised before an LLM call when the session budget would be exceeded."""
    def __init__(self, session_id: str, spent: float, budget: float):
        self.session_id = session_id
        self.spent      = spent
        self.budget     = budget
        super().__init__(
            f"session={session_id}: cost budget ${budget:.4f} exceeded "
            f"(spent=${spent:.4f})"
        )


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Thread-safe cost tracker. Accumulates token usage per session in memory
    and optionally in Redis for persistence across restarts.

    Usage:
        tracker = CostTracker(redis=redis_client)

        # Before an LLM call — may raise BudgetExceededError
        tracker.check_budget(session_id, estimated_cost=0.002)

        # After the LLM call returns
        record = tracker.record(
            session_id="sess_abc",
            model="claude-haiku-4-5-20251001",
            task_type=TASK_CONVERSE,
            input_tokens=150,
            output_tokens=80,
            latency_ms=420,
        )
        print(record.cost_usd)   # → 0.000044
    """

    REDIS_KEY_PREFIX = "tn:cost:"
    REDIS_TTL        = 86400 * 3  # 3 days

    def __init__(self, redis=None):
        self._redis = redis
        # In-process accumulator — primary for the process lifetime
        self._sessions: Dict[str, SessionCost] = {}
        self._call_log: list[CallRecord] = []   # rolling in-memory log

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def set_budget(self, session_id: str, budget_usd: float) -> None:
        """Set a USD cost cap for a session. Enforced before each LLM call."""
        cost = self._get_or_create(session_id)
        cost.budget_usd = budget_usd
        logger.info("session=%s budget set to $%.4f", session_id, budget_usd)

    def check_budget(self, session_id: str, estimated_cost: float = 0.0) -> None:
        """
        Raise BudgetExceededError if the session is over budget (or would be
        if the estimated_cost is added). Call this BEFORE each LLM request.
        """
        cost = self._get_or_create(session_id)
        if cost.budget_usd is None:
            return
        projected = cost.total_cost_usd + estimated_cost
        if projected >= cost.budget_usd:
            raise BudgetExceededError(
                session_id=session_id,
                spent=cost.total_cost_usd,
                budget=cost.budget_usd,
            )

    def record(
        self,
        session_id:    str,
        model:         str,
        task_type:     str,
        input_tokens:  int,
        output_tokens: int,
        latency_ms:    int = 0,
        metadata:      Optional[dict] = None,
    ) -> CallRecord:
        """
        Record a completed LLM call. Returns the CallRecord with computed cost.

        This is the ONLY place where cost is calculated — don't compute it elsewhere.
        """
        import uuid
        cost_usd = self._compute_cost(model, input_tokens, output_tokens)
        call_id  = str(uuid.uuid4())

        record = CallRecord(
            call_id       = call_id,
            session_id    = session_id,
            model         = model,
            task_type     = task_type,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            cost_usd      = cost_usd,
            latency_ms    = latency_ms,
            metadata      = metadata or {},
        )

        session_cost = self._get_or_create(session_id)
        session_cost.add(record)
        self._call_log.append(record)
        self._redis_save(session_id, session_cost)
        logger.info(
            "llm_call session=%s model=%s task=%s in=%d out=%d cost=$%.6f latency=%dms",
            session_id, model, task_type, input_tokens, output_tokens, cost_usd, latency_ms,
        )

        return record

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_session_cost(self, session_id: str) -> SessionCost:
        """Return accumulated cost for a session. Loads from Redis if not in memory."""
        if session_id not in self._sessions:
            loaded = self._redis_load(session_id)
            if loaded:
                self._sessions[session_id] = loaded
            else:
                self._sessions[session_id] = SessionCost(session_id=session_id)
        return self._sessions[session_id]

    def get_call_log(self, session_id: Optional[str] = None) -> list[dict]:
        """Return call log, optionally filtered by session_id."""
        records = (
            [r for r in self._call_log if r.session_id == session_id]
            if session_id else self._call_log
        )
        return [r.to_dict() for r in records]

    def summary(self, session_id: str) -> dict:
        """
        Human-readable cost summary for a session.
        Suitable for display in the Studio dashboard or CLI.
        """
        cost = self.get_session_cost(session_id)
        d = cost.to_dict()
        d["total_tokens"] = cost.total_input_tokens + cost.total_output_tokens
        d["cost_formatted"] = f"${cost.total_cost_usd:.4f}"
        if cost.budget_usd:
            pct = cost.total_cost_usd / cost.budget_usd * 100
            d["budget_used_pct"] = round(pct, 1)
        return d

    def estimate(
        self,
        model:         str,
        input_tokens:  int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a hypothetical call without recording it."""
        return self._compute_cost(model, input_tokens, output_tokens)

    # ------------------------------------------------------------------
    # Pricing calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Compute USD cost for a model call.
        Prices are per 1M tokens; we normalise to per-token here.
        """
        # Normalise model name — strip snapshot date suffix for lookup
        base_model = model.split(":")[0]  # handle "ollama:llama3.1" style
        price_in, price_out = _PRICING.get(base_model, _FALLBACK_PRICING)
        cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        return round(cost, 8)

    # ------------------------------------------------------------------
    # In-memory helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, session_id: str) -> SessionCost:
        if session_id not in self._sessions:
            # Try loading from Redis first
            loaded = self._redis_load(session_id)
            self._sessions[session_id] = loaded or SessionCost(session_id=session_id)
        return self._sessions[session_id]

    # ------------------------------------------------------------------
    # Redis persistence helpers (best-effort, non-blocking)
    # ------------------------------------------------------------------

    def _redis_save(self, session_id: str, cost: SessionCost) -> None:
        if not self._redis:
            return
        try:
            key = f"{self.REDIS_KEY_PREFIX}{session_id}"
            # Fire-and-forget — use sync if redis client is sync, skip if async
            if hasattr(self._redis, "set"):
                self._redis.set(key, json.dumps(cost.to_dict()), ex=self.REDIS_TTL)
        except Exception as e:
            logger.debug("cost_tracker redis_save failed: %s", e)

    def _redis_load(self, session_id: str) -> Optional[SessionCost]:
        if not self._redis:
            return None
        try:
            key = f"{self.REDIS_KEY_PREFIX}{session_id}"
            raw = self._redis.get(key) if hasattr(self._redis, "get") else None
            if not raw:
                return None
            data = json.loads(raw)
            cost = SessionCost(
                session_id          = data["session_id"],
                total_input_tokens  = data["total_input_tokens"],
                total_output_tokens = data["total_output_tokens"],
                total_cost_usd      = data["total_cost_usd"],
                by_task             = data["by_task"],
                by_model            = data["by_model"],
                call_count          = data["call_count"],
                budget_usd          = data.get("budget_usd"),
            )
            return cost
        except Exception as e:
            logger.debug("cost_tracker redis_load failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Goal-level aggregation (called by analytics API)
    # ------------------------------------------------------------------

    def aggregate_goal_cost(self, session_ids: list[str]) -> dict:
        """
        Aggregate cost across multiple sessions (e.g. all sessions for a goal).
        """
        total_cost   = 0.0
        total_tokens = 0
        by_model:    dict[str, float] = {}
        by_task:     dict[str, float] = {}

        for sid in session_ids:
            c = self.get_session_cost(sid)
            total_cost   += c.total_cost_usd
            total_tokens += c.total_input_tokens + c.total_output_tokens
            for model, cost in c.by_model.items():
                by_model[model] = by_model.get(model, 0.0) + cost
            for task, cost in c.by_task.items():
                by_task[task] = by_task.get(task, 0.0) + cost

        return {
            "session_count": len(session_ids),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens":   total_tokens,
            "by_model":       {k: round(v, 6) for k, v in by_model.items()},
            "by_task":        {k: round(v, 6) for k, v in by_task.items()},
            "avg_cost_per_session": round(total_cost / max(len(session_ids), 1), 6),
        }