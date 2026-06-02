"""
No real Redis, no real HTTP, no network calls.

Classes:
  1.  RateLimiter_InMemory       — sliding window without Redis
  2.  RateLimiter_Plans          — FREE / STARTER / PRO limits
  3.  RateLimiter_Dimensions     — per-key, per-goal, per-user
  4.  RateLimiter_Reset          — admin reset
  5.  RateLimitResult            — headers, remaining, retry_after
  6.  APIKeyManager              — create, validate, revoke
  7.  JWTHandler                 — issue, verify, expiry
  8.  AuthMiddleware_APIKey       — header extraction + validation
  9.  AuthMiddleware_JWT          — bearer token auth
  10. AuthMiddleware_Public       — public paths bypass auth
  11. BudgetGuard_Session        — session budget enforcement
  12. BudgetGuard_Tenant         — tenant monthly budget
  13. BudgetGuard_Allowed        — no budget configured → allow
  14. BudgetCheckResult          — to_response structure
  15. SelfHostConfig_Minimal     — minimal profile compose output
  16. SelfHostConfig_Standard    — standard profile
  17. SelfHostConfig_Files       — generates all required files
  18. GoalRegistry_Publish       — publish + version validation
  19. GoalRegistry_Install       — install by name + version
  20. GoalRegistry_Search        — keyword + sector + tag search
  21. GoalRegistry_Curated       — official goals seeded
  22. SectorProduction           — full auth+rate+budget for 5 sectors
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.api.middleware.rate_limiter import (
    RateLimiter, RateLimitResult, Plan, _MemoryWindow,
)
from truenorth.api.middleware.auth import (
    AuthMiddleware, APIKeyManager, JWTHandler, AuthResult, AuthScheme,
)
from truenorth.api.middleware.budget_guard import (
    BudgetGuard, BudgetCheckResult, BudgetScope,
    TenantBudgetConfig,
)
from truenorth.cloud.self_host_config import (
    SelfHostConfig, DeployProfile, cli_init,
)
from truenorth.marketplace.goal_registry import (
    GoalRegistry,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cost_tracker_stub(session_budget: Optional[float] = None, spent: float = 0.0):
    """Minimal CostTracker-compatible stub."""
    from truenorth.llm.cost_tracker import CostTracker
    ct = CostTracker()
    if session_budget is not None:
        ct.set_budget("sess-1", session_budget)
    if spent > 0:
        ct.record("sess-1", "claude-haiku-4-5-20251001", "extract",
                  int(spent * 1_000_000 / 0.8), 0, goal_id="fitness")
    return ct


SAMPLE_GOAL_YAML = """\
name: test-goal
version: 1.0.0
author: "@test"
description: "Test goal for unit tests"
sector: fitness
tags: [test, fitness]
license: MIT
fields:
  - name: age
    type: integer
    required: true
    question: "How old are you?"
output:
  format: json
"""


# ─────────────────────────────────────────────────────────────────────────────
#  1. RateLimiter — in-memory sliding window
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiterInMemory:

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        rl     = RateLimiter(plan=Plan.STARTER)
        result = await rl.check("key-1", goal_id="fitness", user_id="u1")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_blocks_after_limit_exceeded(self):
        rl = RateLimiter(
            plan=Plan.FREE,
            custom_limits={"api_key": (3, 3600)},   # 3 requests/hr
            skip_dimensions=["goal", "user"],
        )
        for _ in range(3):
            await rl.check("key-block-test")
        result = await rl.check("key-block-test")
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_different_keys_independent(self):
        rl = RateLimiter(
            custom_limits={"api_key": (2, 3600)},
            skip_dimensions=["goal", "user"],
        )
        await rl.check("key-A")
        await rl.check("key-A")
        blocked = await rl.check("key-A")
        allowed = await rl.check("key-B")
        assert blocked.allowed is False
        assert allowed.allowed is True

    def test_memory_window_check_and_record(self):
        w  = _MemoryWindow()
        ok1, c1 = w.check_and_record("k", 3, 3600)
        ok2, c2 = w.check_and_record("k", 3, 3600)
        ok3, c3 = w.check_and_record("k", 3, 3600)
        ok4, c4 = w.check_and_record("k", 3, 3600)
        assert ok1 is True and ok2 is True and ok3 is True
        assert ok4 is False

    def test_memory_window_reset(self):
        w = _MemoryWindow()
        w.check_and_record("k", 1, 3600)
        w.check_and_record("k", 1, 3600)   # blocked
        w.reset("k")
        ok, _ = w.check_and_record("k", 1, 3600)
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
#  2. RateLimiter — plan limits
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiterPlans:

    def test_free_plan_limits(self):
        rl = RateLimiter(plan=Plan.FREE)
        limits = rl.limits()
        assert limits["api_key"]["limit"] == 100
        assert limits["goal"]["limit"]    == 50
        assert limits["user"]["limit"]    == 20

    def test_pro_plan_limits(self):
        rl = RateLimiter(plan=Plan.PRO)
        limits = rl.limits()
        assert limits["api_key"]["limit"] == 10_000

    def test_enterprise_plan_limits(self):
        rl = RateLimiter(plan=Plan.ENTERPRISE)
        limits = rl.limits()
        assert limits["api_key"]["limit"] == 100_000

    def test_update_plan_at_runtime(self):
        rl = RateLimiter(plan=Plan.FREE)
        assert rl.limits()["api_key"]["limit"] == 100
        rl.update_plan(Plan.PRO)
        assert rl.limits()["api_key"]["limit"] == 10_000


# ─────────────────────────────────────────────────────────────────────────────
#  3. RateLimiter — dimensions
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiterDimensions:

    @pytest.mark.asyncio
    async def test_skip_empty_dimensions(self):
        """Empty user_id or goal_id should not be checked."""
        rl = RateLimiter(custom_limits={"user": (0, 3600)})
        # user_id="" — user check skipped → should be allowed
        result = await rl.check("key-1", goal_id="g", user_id="")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_skip_dimensions_config(self):
        rl = RateLimiter(skip_dimensions=["goal", "user"])
        result = await rl.check("key-1", goal_id="g", user_id="u1")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_blocked_result_has_dimension(self):
        rl = RateLimiter(
            custom_limits={"api_key": (1, 3600)},
            skip_dimensions=["goal", "user"],
        )
        await rl.check("key-dim")
        result = await rl.check("key-dim")
        assert result.dimension == "api_key"

    @pytest.mark.asyncio
    async def test_get_count(self):
        rl = RateLimiter(skip_dimensions=["goal", "user"])
        await rl.check("key-count")
        await rl.check("key-count")
        count = rl.get_count("api_key", "key-count")
        assert count == 2


# ─────────────────────────────────────────────────────────────────────────────
#  4. RateLimiter — reset
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiterReset:

    @pytest.mark.asyncio
    async def test_reset_allows_again(self):
        rl = RateLimiter(
            custom_limits={"api_key": (2, 3600)},
            skip_dimensions=["goal", "user"],
        )
        await rl.check("key-reset")
        await rl.check("key-reset")
        blocked = await rl.check("key-reset")
        assert blocked.allowed is False

        rl.reset(api_key="key-reset")
        allowed = await rl.check("key-reset")
        assert allowed.allowed is True


# ─────────────────────────────────────────────────────────────────────────────
#  5. RateLimitResult
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimitResult:

    def test_allowed_result_has_no_retry_after(self):
        r = RateLimitResult(allowed=True, limit=100, remaining=99, reset_at=time.time()+3600)
        assert "Retry-After" not in r.headers

    def test_blocked_result_has_retry_after(self):
        r = RateLimitResult(allowed=False, limit=100, remaining=0,
                            reset_at=time.time()+3600, retry_after_s=60)
        assert "Retry-After" in r.headers
        assert r.headers["Retry-After"] == "60"

    def test_headers_have_rate_limit_keys(self):
        r = RateLimitResult(allowed=True, limit=1000, remaining=999, reset_at=time.time()+3600)
        h = r.headers
        assert "X-RateLimit-Limit"     in h
        assert "X-RateLimit-Remaining" in h
        assert "X-RateLimit-Reset"     in h


# ─────────────────────────────────────────────────────────────────────────────
#  6. APIKeyManager
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIKeyManager:

    def test_create_and_validate_key(self):
        km = APIKeyManager()
        raw, key_id = km.create_key("ten-1", plan="pro")
        assert raw.startswith("tn_live_")
        meta = km.validate(raw)
        assert meta is not None
        assert meta["tenant_id"] == "ten-1"
        assert meta["plan"]      == "pro"

    def test_test_key_prefix(self):
        km  = APIKeyManager()
        raw, _ = km.create_key("ten-1", is_test=True)
        assert raw.startswith("tn_test_")

    def test_revoke_key(self):
        km  = APIKeyManager()
        raw, _ = km.create_key("ten-1")
        assert km.validate(raw) is not None
        km.revoke_key(raw)
        assert km.validate(raw) is None

    def test_invalid_prefix_rejected(self):
        km = APIKeyManager()
        assert km.validate("sk-abc123") is None

    def test_empty_key_rejected(self):
        km = APIKeyManager()
        assert km.validate("") is None

    def test_register_known_key(self):
        km  = APIKeyManager()
        raw = "tn_live_" + "a" * 40
        km.register_key(raw, "ten-2", plan="enterprise")
        meta = km.validate(raw)
        assert meta["plan"] == "enterprise"

    def test_scopes_stored(self):
        km = APIKeyManager()
        raw, _ = km.create_key("ten-1", scopes=["read", "admin"])
        meta   = km.validate(raw)
        assert "admin" in meta["scopes"]


# ─────────────────────────────────────────────────────────────────────────────
#  7. JWTHandler
# ─────────────────────────────────────────────────────────────────────────────

class TestJWTHandler:

    def _jwt(self) -> JWTHandler:
        return JWTHandler(secret="super-secret-key-for-tests-at-least-32chars")

    def test_issue_and_verify(self):
        jwt     = self._jwt()
        token   = jwt.issue("ten-1", "u-1", plan="pro")
        payload = jwt.verify(token)
        assert payload is not None
        assert payload["tenant_id"] == "ten-1"
        assert payload["sub"]       == "u-1"
        assert payload["plan"]      == "pro"

    def test_expired_token_rejected(self):
        jwt   = self._jwt()
        token = jwt.issue("ten-1", "u-1", ttl=-1)   # already expired
        assert jwt.verify(token) is None

    def test_tampered_signature_rejected(self):
        jwt   = self._jwt()
        token = jwt.issue("ten-1", "u-1")
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + "x" + "." + parts[2]
        assert jwt.verify(tampered) is None

    def test_wrong_secret_rejected(self):
        jwt1  = JWTHandler("secret-a-must-be-long-enough-here")
        jwt2  = JWTHandler("secret-b-must-be-long-enough-here")
        token = jwt1.issue("ten-1", "u-1")
        assert jwt2.verify(token) is None

    def test_scopes_in_payload(self):
        jwt     = self._jwt()
        token   = jwt.issue("ten-1", "u-1", scopes=["read", "admin"])
        payload = jwt.verify(token)
        assert "admin" in payload["scopes"]

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError):
            JWTHandler(secret="")


# ─────────────────────────────────────────────────────────────────────────────
#  8. AuthMiddleware — API key
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthMiddlewareAPIKey:

    def _auth(self) -> AuthMiddleware:
        km = APIKeyManager()
        return AuthMiddleware(key_manager=km)

    @pytest.mark.asyncio
    async def test_valid_key_authenticates(self):
        auth = self._auth()
        raw, _ = auth._keys.create_key("ten-1", plan="pro")
        result = await auth.verify({
            "headers": {"X-TrueNorth-Key": raw},
            "path": "/session",
        })
        assert result.authenticated is True
        assert result.scheme        == AuthScheme.API_KEY
        assert result.tenant_id     == "ten-1"
        assert result.plan          == "pro"

    @pytest.mark.asyncio
    async def test_invalid_key_rejected(self):
        auth   = self._auth()
        result = await auth.verify({
            "headers": {"X-TrueNorth-Key": "tn_live_invalid"},
            "path":    "/session",
        })
        assert result.authenticated is False
        assert "Invalid" in result.reason

    @pytest.mark.asyncio
    async def test_no_credentials_rejected(self):
        auth   = self._auth()
        result = await auth.verify({"headers": {}, "path": "/session"})
        assert result.authenticated is False

    def test_test_key_flagged(self):
        km  = APIKeyManager()
        raw, _ = km.create_key("ten-1", is_test=True)
        auth   = AuthMiddleware(key_manager=km)
        result = auth._verify_api_key(raw)
        assert result.is_test_key is True

    def test_create_api_key_via_middleware(self):
        auth   = self._auth()
        raw, _ = auth.create_api_key("ten-2", plan="starter")
        assert raw.startswith("tn_live_")


# ─────────────────────────────────────────────────────────────────────────────
#  9. AuthMiddleware — JWT
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthMiddlewareJWT:

    def _auth_with_jwt(self) -> tuple[AuthMiddleware, JWTHandler]:
        jwt  = JWTHandler("test-secret-must-be-at-least-32-chars!!")
        auth = AuthMiddleware(
            key_manager = APIKeyManager(),
            jwt_handler = jwt,
        )
        return auth, jwt

    @pytest.mark.asyncio
    async def test_valid_jwt_authenticates(self):
        auth, jwt = self._auth_with_jwt()
        token     = jwt.issue("ten-1", "u-1", plan="pro")
        result    = await auth.verify({
            "headers": {"Authorization": f"Bearer {token}"},
            "path":    "/studio/dashboard",
        })
        assert result.authenticated is True
        assert result.scheme        == AuthScheme.JWT
        assert result.tenant_id     == "ten-1"
        assert result.user_id       == "u-1"

    @pytest.mark.asyncio
    async def test_expired_jwt_rejected(self):
        auth, jwt = self._auth_with_jwt()
        token     = jwt.issue("ten-1", "u-1", ttl=-1)
        result    = await auth.verify({
            "headers": {"Authorization": f"Bearer {token}"},
            "path":    "/session",
        })
        assert result.authenticated is False

    def test_issue_jwt_via_middleware(self):
        auth, jwt = self._auth_with_jwt()
        token     = auth.issue_jwt("ten-1", "u-5", plan="enterprise")
        assert token is not None
        assert "." in token


# ─────────────────────────────────────────────────────────────────────────────
#  10. AuthMiddleware — public paths
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthMiddlewarePublic:

    @pytest.mark.asyncio
    async def test_health_path_public(self):
        auth   = AuthMiddleware(require_auth=True)
        result = await auth.verify({"headers": {}, "path": "/health"})
        assert result.authenticated is True

    @pytest.mark.asyncio
    async def test_well_known_path_public(self):
        auth   = AuthMiddleware(require_auth=True)
        result = await auth.verify({
            "headers": {}, "path": "/.well-known/agent.json"
        })
        assert result.authenticated is True

    def test_is_admin_scope(self):
        r = AuthResult(authenticated=True, scopes=["admin"])
        assert r.is_admin  is True
        assert r.can_write is True

    def test_read_only_cannot_write(self):
        r = AuthResult(authenticated=True, scopes=["read"])
        assert r.is_admin  is False
        assert r.can_write is False


# ─────────────────────────────────────────────────────────────────────────────
#  11. BudgetGuard — session
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetGuardSession:

    @pytest.mark.asyncio
    async def test_blocks_when_session_exceeded(self):
        ct    = _cost_tracker_stub(session_budget=0.001, spent=0.002)
        guard = BudgetGuard(cost_tracker=ct)
        result = await guard.check(session_id="sess-1", tenant_id="ten-1")
        assert result.blocked is True
        assert result.scope   == BudgetScope.SESSION

    @pytest.mark.asyncio
    async def test_allows_within_session_budget(self):
        ct    = _cost_tracker_stub(session_budget=1.00, spent=0.01)
        guard = BudgetGuard(cost_tracker=ct)
        result = await guard.check(session_id="sess-1", tenant_id="ten-1")
        assert result.blocked is False

    @pytest.mark.asyncio
    async def test_allows_no_budget_set(self):
        ct    = _cost_tracker_stub()   # no budget
        guard = BudgetGuard(cost_tracker=ct)
        result = await guard.check(session_id="sess-1")
        assert result.blocked is False

    def test_set_session_budget(self):
        from truenorth.llm.cost_tracker import CostTracker
        ct    = CostTracker()
        guard = BudgetGuard(cost_tracker=ct)
        guard.set_session_budget("sess-2", 0.50)
        sess = ct.get_session_cost("sess-2")
        assert sess.budget_usd == pytest.approx(0.50)


# ─────────────────────────────────────────────────────────────────────────────
#  12. BudgetGuard — tenant
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetGuardTenant:

    @pytest.mark.asyncio
    async def test_blocks_at_tenant_limit(self):
        guard = BudgetGuard()
        guard.configure_tenant(TenantBudgetConfig(
            tenant_id     = "ten-over",
            monthly_limit = 10.0,
            auto_pause    = True,
        ))
        guard.record_spend("ten-over", 10.01)   # over limit
        result = await guard.check(tenant_id="ten-over")
        assert result.blocked is True
        assert result.scope   == BudgetScope.TENANT

    @pytest.mark.asyncio
    async def test_allows_within_tenant_limit(self):
        guard = BudgetGuard()
        guard.configure_tenant(TenantBudgetConfig(
            tenant_id     = "ten-ok",
            monthly_limit = 100.0,
            auto_pause    = True,
        ))
        guard.record_spend("ten-ok", 5.0)
        result = await guard.check(tenant_id="ten-ok")
        assert result.blocked is False

    @pytest.mark.asyncio
    async def test_no_config_no_block(self):
        guard  = BudgetGuard()
        result = await guard.check(tenant_id="unconfigured-tenant")
        assert result.blocked is False

    def test_tenant_status(self):
        guard = BudgetGuard()
        guard.configure_tenant(TenantBudgetConfig("ten-1", monthly_limit=50.0))
        guard.record_spend("ten-1", 25.0)
        status = guard.tenant_status("ten-1")
        assert status["spent_usd"]  == pytest.approx(25.0)
        assert status["pct_used"]   == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
#  13. BudgetGuard — allowed
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetGuardAllowed:

    @pytest.mark.asyncio
    async def test_no_config_always_allowed(self):
        guard  = BudgetGuard()
        result = await guard.check(session_id="s", tenant_id="t", goal_id="g")
        assert result.blocked is False

    @pytest.mark.asyncio
    async def test_all_scopes_pass(self):
        ct    = _cost_tracker_stub(session_budget=100.0, spent=1.0)
        guard = BudgetGuard(cost_tracker=ct)
        guard.configure_tenant(TenantBudgetConfig("t", monthly_limit=1000.0))
        guard.record_spend("t", 5.0)
        result = await guard.check(session_id="sess-1", tenant_id="t", goal_id="g")
        assert result.blocked is False


# ─────────────────────────────────────────────────────────────────────────────
#  14. BudgetCheckResult
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetCheckResult:

    def test_to_response_structure(self):
        r = BudgetCheckResult(
            blocked=True, scope=BudgetScope.SESSION,
            spent_usd=0.48, limit_usd=0.50,
            session_id="sess-abc", message="budget exceeded",
        )
        d = r.to_response()
        assert d["error"]     == "budget_exceeded"
        assert d["scope"]     == "session"
        assert d["spent_usd"] == pytest.approx(0.48)
        assert d["limit_usd"] == pytest.approx(0.50)
        assert d["pct_used"]  == pytest.approx(96.0)

    def test_pct_used_zero_when_no_limit(self):
        r = BudgetCheckResult(blocked=False, spent_usd=1.0, limit_usd=0.0)
        assert r.pct_used == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  15. SelfHostConfig — minimal
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfHostConfigMinimal:

    def test_minimal_compose_has_core_services(self):
        cfg = SelfHostConfig(profile=DeployProfile.MINIMAL)
        compose = cfg._docker_compose()
        assert "truenorth-api" in compose
        assert "postgres"      in compose
        assert "redis"         in compose

    def test_minimal_no_nginx(self):
        cfg = SelfHostConfig(profile=DeployProfile.MINIMAL)
        compose = cfg._docker_compose()
        assert "nginx" not in compose

    def test_env_template_has_api_keys(self):
        cfg = SelfHostConfig()
        env = cfg._env_template()
        assert "ANTHROPIC_API_KEY" in env
        assert "GEMINI_API_KEY"    in env
        assert "TRUENORTH_API_KEY" in env

    def test_readme_has_quickstart(self):
        cfg    = SelfHostConfig()
        readme = cfg._readme()
        assert "docker compose" in readme.lower() or "docker-compose" in readme
        assert "curl"           in readme


# ─────────────────────────────────────────────────────────────────────────────
#  16. SelfHostConfig — standard
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfHostConfigStandard:

    def test_standard_has_nginx(self):
        cfg = SelfHostConfig(profile=DeployProfile.STANDARD, domain="api.example.com")
        compose = cfg._docker_compose()
        assert "nginx" in compose

    def test_standard_has_worker(self):
        cfg = SelfHostConfig(profile=DeployProfile.STANDARD, with_worker=True)
        compose = cfg._docker_compose()
        assert "truenorth-worker" in compose

    def test_enterprise_has_monitoring(self):
        cfg = SelfHostConfig(profile=DeployProfile.ENTERPRISE)
        compose = cfg._docker_compose()
        assert "prometheus" in compose
        assert "grafana"    in compose

    def test_nginx_conf_has_tls(self):
        cfg    = SelfHostConfig(domain="api.example.com")
        nginx  = cfg._nginx_conf()
        assert "ssl" in nginx
        assert "api.example.com" in nginx


# ─────────────────────────────────────────────────────────────────────────────
#  17. SelfHostConfig — file generation
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfHostConfigFiles:

    def test_generate_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg   = SelfHostConfig(profile=DeployProfile.MINIMAL)
            files = cfg.generate(tmpdir)
            paths = [Path(f) for f in files]
            for p in paths:
                assert p.exists(), f"Expected file {p} not created"

    def test_docker_compose_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            SelfHostConfig().generate(tmpdir)
            compose = (Path(tmpdir) / "docker-compose.yml").read_text()
            assert "truenorth-api" in compose

    def test_env_template_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            SelfHostConfig().generate(tmpdir)
            env = (Path(tmpdir) / ".env.template").read_text()
            assert "TRUENORTH_JWT_SECRET" in env

    def test_enterprise_generates_prometheus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            SelfHostConfig(profile=DeployProfile.ENTERPRISE).generate(tmpdir)
            prom = (Path(tmpdir) / "prometheus.yml").read_text()
            assert "truenorth-api" in prom

    def test_cli_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = cli_init(output_dir=tmpdir, profile="minimal")
            assert len(files) >= 3


# ─────────────────────────────────────────────────────────────────────────────
#  18. GoalRegistry — publish
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalRegistryPublish:

    def test_publish_valid_goal(self):
        reg = GoalRegistry()
        pkg = reg.publish(SAMPLE_GOAL_YAML, author="@test")
        assert pkg.name    == "test-goal"
        assert pkg.version == "1.0.0"

    def test_published_goal_retrievable(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        info = reg.info("test-goal")
        assert info is not None
        assert info["name"] == "test-goal"

    def test_duplicate_version_raises(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        with pytest.raises(ValueError, match="already exists"):
            reg.publish(SAMPLE_GOAL_YAML)

    def test_overwrite_flag(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        pkg = reg.publish(SAMPLE_GOAL_YAML, overwrite=True)
        assert pkg.name == "test-goal"

    def test_invalid_name_raises(self):
        yaml_bad = SAMPLE_GOAL_YAML.replace("name: test-goal", "name: INVALID CAPS!")
        reg = GoalRegistry()
        with pytest.raises(ValueError, match="name"):
            reg.publish(yaml_bad)

    def test_invalid_version_raises(self):
        yaml_bad = SAMPLE_GOAL_YAML.replace("version: 1.0.0", "version: v1")
        reg = GoalRegistry()
        with pytest.raises(ValueError, match="semver"):
            reg.publish(yaml_bad)

    def test_checksum_computed(self):
        reg = GoalRegistry()
        pkg = reg.publish(SAMPLE_GOAL_YAML)
        assert len(pkg.checksum) == 64   # sha256 hex


# ─────────────────────────────────────────────────────────────────────────────
#  19. GoalRegistry — install
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalRegistryInstall:

    def test_install_returns_config_dict(self):
        reg    = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        config = reg.install("test-goal", save_local=False)
        assert isinstance(config, dict)
        assert config.get("name") == "test-goal"

    def test_install_latest_by_default(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        v2  = SAMPLE_GOAL_YAML.replace("version: 1.0.0", "version: 2.0.0")
        reg.publish(v2)
        config = reg.install("test-goal", save_local=False)
        assert config["version"] == "2.0.0"

    def test_install_specific_version(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        v2  = SAMPLE_GOAL_YAML.replace("version: 1.0.0", "version: 2.0.0")
        reg.publish(v2)
        config = reg.install("test-goal@1.0.0", save_local=False)
        assert config["version"] == "1.0.0"

    def test_install_unknown_raises(self):
        reg = GoalRegistry()
        with pytest.raises(LookupError):
            reg.install("nonexistent-goal-xyz", save_local=False)

    def test_install_tracked_in_installed(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        reg.install("test-goal", save_local=False)
        installed = [p["name"] for p in reg.list_installed()]
        assert "test-goal" in installed

    def test_from_file(self):
        reg = GoalRegistry()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(SAMPLE_GOAL_YAML)
            f.flush()
            config = reg.install_from_file(f.name)
        assert config["name"] == "test-goal"


# ─────────────────────────────────────────────────────────────────────────────
#  20. GoalRegistry — search
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalRegistrySearch:

    def test_search_by_keyword(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        results = reg.search("test")
        assert any(r["name"] == "test-goal" for r in results)

    def test_search_by_sector(self):
        reg = GoalRegistry()
        reg.publish(SAMPLE_GOAL_YAML)
        fitness = reg.search(sector="fitness")
        assert all(r["sector"] == "fitness" for r in fitness)

    def test_search_returns_all_when_no_query(self):
        reg     = GoalRegistry()
        results = reg.search()
        assert len(results) >= 1

    def test_search_limit(self):
        reg     = GoalRegistry()
        results = reg.search(limit=3)
        assert len(results) <= 3

    def test_search_no_match_empty(self):
        reg     = GoalRegistry()
        results = reg.search("xyzzy_completely_nonexistent_term")
        assert len(results) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  21. GoalRegistry — curated goals
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalRegistryCurated:

    def test_official_goals_seeded(self):
        reg   = GoalRegistry()
        names = [p["name"] for p in reg.search()]
        assert "fitness-coach"  in names
        assert "medical-intake" in names
        assert "legal-intake"   in names
        assert "hr-screening"   in names
        assert "financial-plan" in names

    def test_fitness_coach_info(self):
        reg  = GoalRegistry()
        info = reg.info("fitness-coach")
        assert info is not None
        assert info["sector"]   == "fitness"
        assert info["downloads"] >= 0

    def test_install_curated_goal(self):
        reg    = GoalRegistry()
        config = reg.install("fitness-coach", save_local=False)
        assert isinstance(config, dict)

    def test_search_medical_sector(self):
        reg     = GoalRegistry()
        results = reg.search(sector="medical")
        assert any(r["name"] == "medical-intake" for r in results)


# ─────────────────────────────────────────────────────────────────────────────
#  22. Sector production — auth + rate + budget for 5 sectors
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorProduction:

    SECTORS = [
        ("healthcare",     "medical-intake",  "ten-med"),
        ("legal",          "legal-intake",    "ten-leg"),
        ("hr_recruitment", "hr-screening",    "ten-hr"),
        ("financial",      "financial-plan",  "ten-fin"),
        ("fitness",        "fitness-coach",   "ten-fit"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector,goal_id,tenant_id", SECTORS)
    async def test_auth_rate_budget_pipeline(self, sector, goal_id, tenant_id):
        """Full production pipeline: auth → rate check → budget check."""
        # Auth
        km   = APIKeyManager()
        raw, _ = km.create_key(tenant_id, plan="pro")
        auth   = AuthMiddleware(key_manager=km)
        result = await auth.verify({
            "headers": {"X-TrueNorth-Key": raw},
            "path":    f"/session/{goal_id}",
        })
        assert result.authenticated, f"{sector} auth failed"
        assert result.tenant_id == tenant_id
        rl     = RateLimiter(plan=Plan.PRO)
        rl_res = await rl.check(raw, goal_id=goal_id, user_id="u1")
        assert rl_res.allowed, f"{sector} rate limit failed"

        ct    = _cost_tracker_stub(session_budget=1.00, spent=0.01)
        guard = BudgetGuard(cost_tracker=ct)
        bg_res = await guard.check(session_id="sess-1", tenant_id=tenant_id, goal_id=goal_id)
        assert not bg_res.blocked, f"{sector} budget blocked unexpectedly"

    @pytest.mark.parametrize("sector,goal_id,tenant_id", SECTORS)
    def test_goal_in_registry(self, sector, goal_id, tenant_id):
        """Every sector goal is in the curated registry."""
        reg  = GoalRegistry()
        info = reg.info(goal_id)
        assert info is not None, f"{sector} goal '{goal_id}' not in registry"
        assert info["sector"] is not None