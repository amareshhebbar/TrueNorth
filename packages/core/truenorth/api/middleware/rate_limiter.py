"""
Redis sliding-window rate limiter for TrueNorth API.

Three independent limit dimensions, all evaluated per request:
  1. Per API key      — controls a tenant's total API usage
  2. Per goal         — prevents one goal from monopolising capacity
  3. Per user         — prevents one end-user from flooding the API

Each dimension runs a sliding-window counter in Redis so limits
reset smoothly (no cliff at the top of the hour).

Algorithm: Redis sorted-set sliding window
  Key:   tn:rl:{dimension}:{identifier}:{window_seconds}
  Value: sorted set of (timestamp, request_uuid)
  On each request:
    1. Remove members older than now - window_seconds
    2. Count remaining members
    3. If count >= limit → reject with 429
    4. Add (now, uuid) to the set
    5. Set TTL on the key

Limits are configurable per plan (FREE / STARTER / PRO / ENTERPRISE).
Defaults work without Redis — falls back to in-memory (dev mode).

Sector-agnostic: same limiter for medical, legal, HR, fitness APIs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

class Plan(str, Enum):
    FREE       = "free"
    STARTER    = "starter"
    PRO        = "pro"
    ENTERPRISE = "enterprise"

_PLAN_LIMITS: Dict[Plan, Dict[str, Tuple[int, int]]] = {
    Plan.FREE: {
        "api_key":  (100,   3600),
        "goal":     (50,    3600),
        "user":     (20,    3600),
    },
    Plan.STARTER: {
        "api_key":  (1_000,  3600),
        "goal":     (500,    3600),
        "user":     (100,    3600),
    },
    Plan.PRO: {
        "api_key":  (10_000, 3600),
        "goal":     (5_000,  3600),
        "user":     (1_000,  3600),
    },
    Plan.ENTERPRISE: {
        "api_key":  (100_000, 3600),
        "goal":     (50_000,  3600),
        "user":     (10_000,  3600),
    },
}

REDIS_KEY_PREFIX = "tn:rl:"

@dataclass
class RateLimitResult:
    allowed:       bool
    dimension:     str     = ""
    limit:         int     = 0
    remaining:     int     = 0
    reset_at:      float   = 0.0
    reason:        str     = ""
    retry_after_s: int     = 60

    @property
    def headers(self) -> Dict[str, str]:
        """HTTP headers to attach to every response."""
        return {
            "X-RateLimit-Limit":     str(self.limit),
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
            "X-RateLimit-Reset":     str(int(self.reset_at)),
            **({"Retry-After": str(self.retry_after_s)} if not self.allowed else {}),
        }

class _MemoryWindow:
    """Sliding-window counter using a list of timestamps (no Redis dep)."""

    def __init__(self):
        self._windows: Dict[str, List[float]] = {}

    def check_and_record(
        self,
        key:            str,
        limit:          int,
        window_seconds: int,
    ) -> Tuple[bool, int]:
        """
        Returns (allowed, current_count).
        Removes expired entries, checks count, records if allowed.
        """
        now = time.time()
        cutoff = now - window_seconds
        history = self._windows.get(key, [])
        history = [t for t in history if t > cutoff]

        count = len(history)
        if count >= limit:
            return False, count

        history.append(now)
        self._windows[key] = history
        return True, count + 1

    def reset(self, key: str) -> None:
        self._windows.pop(key, None)

    def count(self, key: str, window_seconds: int) -> int:
        now    = time.time()
        cutoff = now - window_seconds
        return sum(1 for t in self._windows.get(key, []) if t > cutoff)

class _RedisWindow:
    """
    Sliding-window using Redis sorted sets.
    Lua script ensures atomicity of the check-and-record operation.
    """

    _LUA_SCRIPT = """
local key       = KEYS[1]
local now       = tonumber(ARGV[1])
local cutoff    = tonumber(ARGV[2])
local limit     = tonumber(ARGV[3])
local member    = ARGV[4]
local ttl       = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)
local count = redis.call('ZCARD', key)
if count >= limit then
    return {0, count}
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, ttl)
local new_count = redis.call('ZCARD', key)
return {1, new_count}
"""

    def __init__(self, redis: Any):
        self._redis = redis
        self._script = None

    def _get_script(self):
        if self._script is None:
            self._script = self._redis.register_script(self._LUA_SCRIPT)
        return self._script

    def check_and_record(
        self,
        key:            str,
        limit:          int,
        window_seconds: int,
    ) -> Tuple[bool, int]:
        now    = time.time()
        cutoff = now - window_seconds
        member = str(uuid.uuid4())
        try:
            result = self._get_script()(
                keys=[key],
                args=[now, cutoff, limit, member, window_seconds + 10],
            )
            allowed = bool(result[0])
            count   = int(result[1])
            return allowed, count
        except Exception:
            return True, 0

    def reset(self, key: str) -> None:
        try:
            self._redis.delete(key)
        except Exception:
            pass

    def count(self, key: str, window_seconds: int) -> int:
        now    = time.time()
        cutoff = now - window_seconds
        try:
            self._redis.zremrangebyscore(key, 0, cutoff)
            return self._redis.zcard(key)
        except Exception:
            return 0

class RateLimiter:
    """
    Three-dimensional sliding-window rate limiter.

    Checks api_key, goal, and user dimensions independently.
    First dimension that hits its limit causes rejection.

    Falls back to in-memory counting when no Redis is configured
    (suitable for single-instance deployments and tests).

    Usage:
        limiter = RateLimiter(redis=redis_client, plan=Plan.PRO)
        result  = await limiter.check("key-abc", goal_id="fitness", user_id="u1")
    """

    def __init__(
        self,
        redis:        Optional[Any]   = None,
        plan:         Plan            = Plan.STARTER,
        custom_limits: Optional[Dict[str, Tuple[int, int]]] = None,
        skip_dimensions: Optional[List[str]] = None,
    ):
        self._plan   = plan
        self._limits = dict(_PLAN_LIMITS[plan])
        if custom_limits:
            self._limits.update(custom_limits)
        self._skip = set(skip_dimensions or [])

        if redis is not None:
            self._window = _RedisWindow(redis)
        else:
            self._window = _MemoryWindow()

    @classmethod
    def from_env(cls) -> "RateLimiter":
        """Build from environment variables."""
        import os
        plan_str = os.environ.get("TRUENORTH_PLAN", "starter").lower()
        plan     = Plan(plan_str) if plan_str in Plan._value2member_map_ else Plan.STARTER

        redis = None
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            try:
                import redis as redis_lib
                redis = redis_lib.from_url(redis_url)
            except ImportError:
                pass

        return cls(redis=redis, plan=plan)

    async def check(
        self,
        api_key: str,
        goal_id: str = "",
        user_id: str = "",
    ) -> RateLimitResult:
        """
        Check all three rate-limit dimensions.
        Returns on first rejection or allows if all pass.
        """
        checks = [
            ("api_key", api_key, "api_key"),
            ("goal",    goal_id, "goal"),
            ("user",    user_id, "user"),
        ]

        last_limit = 0
        last_remaining = 0
        last_reset = time.time() + 3600

        for dimension, identifier, limit_key in checks:
            if dimension in self._skip or not identifier:
                continue

            limit_req, window_s = self._limits.get(limit_key, (1000, 3600))
            redis_key = f"{REDIS_KEY_PREFIX}{dimension}:{identifier}:{window_s}"

            allowed, count = self._window.check_and_record(redis_key, limit_req, window_s)
            remaining      = max(limit_req - count, 0)
            reset_at       = time.time() + window_s
            last_limit     = limit_req
            last_remaining = remaining
            last_reset     = reset_at

            if not allowed:
                return RateLimitResult(
                    allowed       = False,
                    dimension     = dimension,
                    limit         = limit_req,
                    remaining     = 0,
                    reset_at      = reset_at,
                    reason        = (
                        f"Rate limit exceeded: {dimension}={identifier!r} "
                        f"({limit_req} requests per {window_s}s)"
                    ),
                    retry_after_s = min(window_s, 3600),
                )

        return RateLimitResult(
            allowed   = True,
            limit     = last_limit,
            remaining = last_remaining,
            reset_at  = last_reset,
        )

    def check_sync(
        self,
        api_key: str,
        goal_id: str = "",
        user_id: str = "",
    ) -> RateLimitResult:
        """Synchronous version of check(). Same logic, no await."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            return asyncio.ensure_future(self.check(api_key, goal_id, user_id))
        except RuntimeError:
            return asyncio.run(self.check(api_key, goal_id, user_id))

    def reset(self, api_key: str = "", goal_id: str = "", user_id: str = "") -> None:
        """Reset rate limit counters (admin use — e.g. after a billing upgrade)."""
        for dim, identifier in [("api_key", api_key), ("goal", goal_id), ("user", user_id)]:
            if identifier:
                _, window_s = self._limits.get(dim, (0, 3600))
                key = f"{REDIS_KEY_PREFIX}{dim}:{identifier}:{window_s}"
                self._window.reset(key)

    def get_count(self, dimension: str, identifier: str) -> int:
        """Return current request count for a dimension/identifier pair."""
        _, window_s = self._limits.get(dimension, (0, 3600))
        key = f"{REDIS_KEY_PREFIX}{dimension}:{identifier}:{window_s}"
        return self._window.count(key, window_s)

    def update_plan(self, plan: Plan) -> None:
        """Upgrade/downgrade limits at runtime (e.g. after plan change)."""
        self._plan   = plan
        self._limits = dict(_PLAN_LIMITS[plan])

    def limits(self) -> Dict[str, Dict[str, int]]:
        """Return current configured limits."""
        return {
            dim: {"limit": lim, "window_s": win}
            for dim, (lim, win) in self._limits.items()
        }
