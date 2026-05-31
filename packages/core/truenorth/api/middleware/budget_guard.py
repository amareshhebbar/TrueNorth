"""
BudgetGuard — API-layer budget enforcement.

Sits in front of every request that starts or continues a session.
Checks three budget scopes before allowing the request through:

  1. Session budget  — per-conversation USD cap (set in goal YAML or API params)
  2. Goal budget     — monthly USD cap per goal across all sessions
  3. Tenant budget   — monthly USD cap for the whole account

Response on breach:
    HTTP 402 Payment Required
    {
      "error":        "budget_exceeded",
      "scope":        "session",
      "spent_usd":    0.48,
      "limit_usd":    0.50,
      "session_id":   "sess-abc",
      "message":      "Session budget exceeded..."
    }

The guard does NOT block by default when Redis is down or the
cost_tracker has no data — it fails open to avoid false positives.
Set strict=True for financial/compliance workloads.

Usage (FastAPI):
    guard = BudgetGuard(cost_tracker=ct, redis=redis_client)

    @app.middleware("http")
    async def budget_middleware(request: Request, call_next):
        session_id = request.path_params.get("session_id")
        result = await guard.check(
            session_id = session_id,
            tenant_id  = request.state.auth.tenant_id,
            goal_id    = request.path_params.get("goal_id"),
        )
        if result.blocked:
            return JSONResponse(result.to_response(), status_code=402)
        return await call_next(request)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.llm.cost_tracker import CostTracker


class BudgetScope(str, Enum):
    SESSION = "session"
    GOAL    = "goal"
    TENANT  = "tenant"


@dataclass
class BudgetCheckResult:
    blocked:    bool
    scope:      Optional[BudgetScope] = None
    spent_usd:  float = 0.0
    limit_usd:  float = 0.0
    message:    str   = ""
    session_id: str   = ""
    goal_id:    str   = ""
    tenant_id:  str   = ""

    @property
    def pct_used(self) -> float:
        return (self.spent_usd / self.limit_usd * 100) if self.limit_usd > 0 else 0.0

    def to_response(self) -> dict:
        return {
            "error":      "budget_exceeded",
            "scope":      self.scope.value if self.scope else "",
            "spent_usd":  round(self.spent_usd, 4),
            "limit_usd":  round(self.limit_usd, 4),
            "pct_used":   round(self.pct_used, 1),
            "session_id": self.session_id,
            "goal_id":    self.goal_id,
            "tenant_id":  self.tenant_id,
            "message":    self.message,
        }


@dataclass
class TenantBudgetConfig:
    """Monthly budget configuration for a tenant."""
    tenant_id:       str
    monthly_limit:   float            = 100.0     # USD
    alert_pct:       float            = 0.80      # warn at 80%
    auto_pause:      bool             = True       # block at 100%
    goal_limits:     Dict[str, float] = field(default_factory=dict)  


class BudgetGuard:
    """
    API-layer budget enforcement across session, goal, and tenant scopes.

    Reads from CostTracker for session/goal budgets.
    Stores tenant monthly totals in Redis (in-memory fallback for dev).

    Usage:
        guard = BudgetGuard(cost_tracker=ct, redis=redis_client)

        result = await guard.check(
            session_id="sess-abc",
            tenant_id="ten-123",
            goal_id="fitness_plan",
        )
        if result.blocked:
            # return 402
    """

    REDIS_PREFIX = "tn:budget:"

    def __init__(
        self,
        cost_tracker: Optional["CostTracker"] = None,
        redis:        Optional[Any]            = None,
        strict:       bool                     = False,   
        tenant_configs: Optional[Dict[str, TenantBudgetConfig]] = None,
    ):
        self._ct            = cost_tracker
        self._redis         = redis
        self._strict        = strict
        self._tenant_cfgs   = tenant_configs or {}
        self._memory_totals: Dict[str, float] = {} 

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    async def check(
        self,
        session_id: str         = "",
        tenant_id:  str         = "",
        goal_id:    str         = "",
        estimated_cost: float   = 0.0,
    ) -> BudgetCheckResult:
        """
        Check all budget scopes. Returns on first violation.
        """
        if session_id and self._ct:
            result = self._check_session(session_id, goal_id, estimated_cost)
            if result.blocked:
                return result

        if goal_id and tenant_id and self._ct:
            result = self._check_goal(goal_id, tenant_id)
            if result.blocked:
                return result

        if tenant_id:
            result = await self._check_tenant(tenant_id, goal_id, session_id)
            if result.blocked:
                return result

        return BudgetCheckResult(blocked=False)

    # ------------------------------------------------------------------
    # Budget setters
    # ------------------------------------------------------------------

    def set_session_budget(self, session_id: str, budget_usd: float) -> None:
        """Set a per-session USD cap on the CostTracker."""
        if self._ct:
            self._ct.set_budget(session_id, budget_usd)

    def configure_tenant(self, config: TenantBudgetConfig) -> None:
        """Register a tenant budget configuration."""
        self._tenant_cfgs[config.tenant_id] = config

    def record_spend(self, tenant_id: str, amount_usd: float) -> None:
        """Record tenant spend (called after each LLM response)."""
        current = self._get_tenant_total(tenant_id)
        new_val = current + amount_usd
        self._set_tenant_total(tenant_id, new_val)

    # ------------------------------------------------------------------
    # Scope-specific checks
    # ------------------------------------------------------------------

    def _check_session(
        self,
        session_id:     str,
        goal_id:        str,
        estimated_cost: float,
    ) -> BudgetCheckResult:
        """Check per-session budget from CostTracker."""
        try:
            session = self._ct.get_session_cost(session_id)
            if session.budget_usd is None:
                return BudgetCheckResult(blocked=False)

            projected = session.total_cost_usd + estimated_cost
            if projected >= session.budget_usd:
                return BudgetCheckResult(
                    blocked    = True,
                    scope      = BudgetScope.SESSION,
                    spent_usd  = session.total_cost_usd,
                    limit_usd  = session.budget_usd,
                    session_id = session_id,
                    goal_id    = goal_id,
                    message    = (
                        f"Session budget ${session.budget_usd:.2f} reached "
                        f"(spent ${session.total_cost_usd:.4f}). "
                        f"Start a new session to continue."
                    ),
                )
        except Exception:
            if self._strict:
                return BudgetCheckResult(
                    blocked=True, scope=BudgetScope.SESSION,
                    message="Budget check failed (strict mode)",
                )
        return BudgetCheckResult(blocked=False)

    def _check_goal(self, goal_id: str, tenant_id: str) -> BudgetCheckResult:
        """Check per-goal monthly budget from TenantBudgetConfig."""
        cfg = self._tenant_cfgs.get(tenant_id)
        if not cfg or goal_id not in cfg.goal_limits:
            return BudgetCheckResult(blocked=False)

        limit = cfg.goal_limits[goal_id]
        spent = self._get_goal_monthly_total(goal_id, tenant_id)
        if spent >= limit:
            return BudgetCheckResult(
                blocked   = True,
                scope     = BudgetScope.GOAL,
                spent_usd = spent,
                limit_usd = limit,
                goal_id   = goal_id,
                tenant_id = tenant_id,
                message   = (
                    f"Monthly goal budget for '{goal_id}' reached "
                    f"(${spent:.2f} / ${limit:.2f}). Resets next month."
                ),
            )
        return BudgetCheckResult(blocked=False)

    async def _check_tenant(
        self,
        tenant_id:  str,
        goal_id:    str,
        session_id: str,
    ) -> BudgetCheckResult:
        """Check tenant monthly budget."""
        cfg = self._tenant_cfgs.get(tenant_id)
        if not cfg or not cfg.auto_pause:
            return BudgetCheckResult(blocked=False)

        spent = self._get_tenant_total(tenant_id)
        limit = cfg.monthly_limit
        if spent >= limit:
            return BudgetCheckResult(
                blocked   = True,
                scope     = BudgetScope.TENANT,
                spent_usd = spent,
                limit_usd = limit,
                tenant_id = tenant_id,
                goal_id   = goal_id,
                session_id = session_id,
                message   = (
                    f"Account monthly budget ${limit:.2f} reached "
                    f"(${spent:.2f} spent). Upgrade your plan or wait until next month."
                ),
            )
        return BudgetCheckResult(blocked=False)

    # ------------------------------------------------------------------
    # Tenant spend tracking (Redis / in-memory)
    # ------------------------------------------------------------------

    def _get_tenant_total(self, tenant_id: str) -> float:
        key = self._monthly_key(tenant_id)
        if self._redis:
            try:
                val = self._redis.get(key)
                return float(val) if val else 0.0
            except Exception:
                pass
        return self._memory_totals.get(key, 0.0)

    def _set_tenant_total(self, tenant_id: str, amount: float) -> None:
        key = self._monthly_key(tenant_id)
        if self._redis:
            try:
                self._redis.set(key, str(amount), ex=self._seconds_until_month_end())
            except Exception:
                pass
        self._memory_totals[key] = amount

    def _get_goal_monthly_total(self, goal_id: str, tenant_id: str) -> float:
        """Approximate goal spend from CostTracker."""
        if not self._ct:
            return 0.0
        try:
            gc = self._ct.goal_cost(goal_id)
            return gc.total_cost_usd
        except Exception:
            return 0.0

    @staticmethod
    def _monthly_key(tenant_id: str) -> str:
        import datetime
        now = datetime.datetime.utcnow()
        return f"tn:budget:tenant:{tenant_id}:{now.year}-{now.month:02d}"

    @staticmethod
    def _seconds_until_month_end() -> int:
        import datetime
        now   = datetime.datetime.utcnow()
        if now.month == 12:
            next_m = datetime.datetime(now.year + 1, 1, 1)
        else:
            next_m = datetime.datetime(now.year, now.month + 1, 1)
        return max(int((next_m - now).total_seconds()), 3600)

    # ------------------------------------------------------------------
    # Status query
    # ------------------------------------------------------------------

    def tenant_status(self, tenant_id: str) -> dict:
        """Return current budget status for a tenant."""
        cfg   = self._tenant_cfgs.get(tenant_id)
        spent = self._get_tenant_total(tenant_id)
        limit = cfg.monthly_limit if cfg else None
        return {
            "tenant_id":     tenant_id,
            "spent_usd":     round(spent, 4),
            "limit_usd":     limit,
            "pct_used":      round(spent / limit * 100, 1) if limit else None,
            "auto_pause":    cfg.auto_pause if cfg else False,
        }