"""
Classes:
  1.  RoutingTable        — default routing, env overrides, YAML config
  2.  FallbackChain       — primary fails → secondary → emergency
  3.  RetryBackoff        — transient errors retried, fatal errors not
  4.  CircuitBreaker      — N consecutive failures → provider skipped
  5.  BudgetGuard         — request rejected when cost > budget
  6.  ProviderDetection   — model prefix → provider mapping
  7.  ClientRegistry      — register_client, lazy init
  8.  HealthCheck         — providers probed, failing ones circuit-opened
  9.  Stats               — per-model call counts, latency, error rate
  10. Streaming           — stream with fallback on connection failure
  11. Sector              — router works identically for healthcare/legal/HR
  12. V1Compatibility     — old generate() / register_client() API preserved
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.llm.router import (
    LLMRouter,
    BudgetExceededError,
    AllProvidersFailedError,
    ModelStats,
    TASK_EXTRACT,
    TASK_CONVERSE,
    TASK_OUTPUT,
    TASK_CLASSIFY,
    TASK_VERIFY,
    TASK_OTHER,
    _DEFAULT_ROUTING,
)
from truenorth.llm.base import LLMBase, LLMResponse, Message
from truenorth.testing.mock_llm import MockLLMClient

def _msg(text: str = "hello") -> List[Message]:
    return [Message(role="user", content=text)]

def _mock_router(default: str = "ok") -> LLMRouter:
    mock = MockLLMClient(default=default)
    router = LLMRouter()
    for model in list(_DEFAULT_ROUTING.values()):
        router.register_client(model, mock)
    return router

class FailingClient(LLMBase):
    """Client that always raises an exception."""
    model_name = "failing"

    def __init__(self, error: str = "provider error", fatal: bool = False):
        super().__init__()
        self._error = error
        self._fatal = fatal
        self.call_count = 0

    async def generate(self, messages, system=None, max_tokens=1024, temperature=0.7, **kw):
        self.call_count += 1
        msg = "invalid api key: " + self._error if self._fatal else self._error
        raise RuntimeError(msg)

    async def generate_stream(self, messages, system=None, max_tokens=1024, temperature=0.7, **kw):
        raise RuntimeError(self._error)
        yield

class TestRoutingTable:

    def test_default_routing_populated(self):
        router = LLMRouter()
        rt = router.routing_table()
        assert TASK_EXTRACT  in rt
        assert TASK_CONVERSE in rt
        assert TASK_OUTPUT   in rt
        assert TASK_VERIFY   in rt

    def test_extract_defaults_to_gemini_flash(self):
        router = LLMRouter()
        assert router.routing_table()[TASK_EXTRACT] == "gemini-3.5-flash"

    def test_output_defaults_to_claude_sonnet(self):
        router = LLMRouter()
        assert "sonnet" in router.routing_table()[TASK_OUTPUT].lower()

    def test_custom_routing_overrides_default(self):
        router = LLMRouter(routing={TASK_EXTRACT: "gpt-4o-mini"})
        assert router.routing_table()[TASK_EXTRACT] == "gpt-4o-mini"

    def test_set_routing_at_runtime(self):
        router = LLMRouter()
        router.set_routing(TASK_CONVERSE, "gpt-4o")
        assert router.routing_table()[TASK_CONVERSE] == "gpt-4o"

    def test_from_env_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("TRUENORTH_MODEL_EXTRACT", "gpt-4o-mini")
        monkeypatch.setenv("TRUENORTH_MODEL_OUTPUT",  "gpt-4o")
        router = LLMRouter.from_env()
        rt = router.routing_table()
        assert rt[TASK_EXTRACT] == "gpt-4o-mini"
        assert rt[TASK_OUTPUT]  == "gpt-4o"

    def test_from_env_budget_env_var(self, monkeypatch):
        monkeypatch.setenv("TRUENORTH_BUDGET_USD", "0.25")
        router = LLMRouter.from_env()
        assert router._budget_usd == pytest.approx(0.25)

    def test_from_config_reads_yaml_routing(self):
        config = {
            "routing": {
                TASK_EXTRACT:  "gemini-2.0-flash",
                TASK_CONVERSE: "gpt-4o-mini",
            },
            "budget_usd": 0.10,
        }
        router = LLMRouter.from_config(config)
        rt = router.routing_table()
        assert rt[TASK_EXTRACT]  == "gemini-2.0-flash"
        assert rt[TASK_CONVERSE] == "gpt-4o-mini"
        assert router._budget_usd == pytest.approx(0.10)

    def test_from_config_fallbacks(self):
        config = {
            "fallbacks": {TASK_OUTPUT: ["gpt-4o", "gpt-4o-mini"]},
        }
        router = LLMRouter.from_config(config)
        assert router.fallback_table()[TASK_OUTPUT] == ["gpt-4o", "gpt-4o-mini"]

    def test_verify_task_routed_to_best_model(self):
        router = LLMRouter()
        rt = router.routing_table()

        assert rt[TASK_VERIFY] == "claude-sonnet-4-20250514"

class TestFallbackChain:

    @pytest.mark.asyncio
    async def test_fallback_used_when_primary_fails(self):
        """Primary fails → router tries secondary → succeeds."""
        failing = FailingClient()
        success = MockLLMClient(default="fallback worked")

        router = LLMRouter(
            routing={TASK_CONVERSE: "primary-model"},
            fallbacks={TASK_CONVERSE: ["fallback-model"]},
            max_retries=0,
        )
        router.register_client("primary-model",  failing)
        router.register_client("fallback-model", success)

        resp = await router.generate(TASK_CONVERSE, _msg())
        assert "fallback worked" in resp.content

    @pytest.mark.asyncio
    async def test_all_fallbacks_tried_before_raising(self):
        """All providers fail → AllProvidersFailedError."""
        router = LLMRouter(
            routing={TASK_CONVERSE: "m1"},
            fallbacks={TASK_CONVERSE: ["m2", "m3"]},
            max_retries=0,
        )
        for m in ["m1", "m2", "m3"]:
            router.register_client(m, FailingClient())

        with pytest.raises(AllProvidersFailedError):
            await router.generate(TASK_CONVERSE, _msg())

    @pytest.mark.asyncio
    async def test_chain_deduplicates_models(self):
        """Fallback chain shouldn't repeat the primary model."""
        router = LLMRouter(
            routing={TASK_CONVERSE: "model-a"},
            fallbacks={TASK_CONVERSE: ["model-a", "model-b"]},
        )
        chain = router._build_chain(TASK_CONVERSE)

        assert chain.count("model-a") == 1

    @pytest.mark.asyncio
    async def test_explicit_model_overrides_primary(self):
        """Passing model= arg bypasses routing table."""
        success = MockLLMClient(default="explicit worked")
        router  = LLMRouter(max_retries=0)
        router.register_client("explicit-model", success)

        resp = await router.generate(TASK_CONVERSE, _msg(), model="explicit-model")
        assert "explicit worked" in resp.content

class TestRetryBackoff:

    @pytest.mark.asyncio
    async def test_transient_error_retried(self):
        """Client fails twice then succeeds — router retries."""

        class FlakeyClient(LLMBase):
            model_name = "flakey"
            def __init__(self):
                super().__init__()
                self.attempts = 0

            async def generate(self, messages, system=None, max_tokens=1024, temperature=0.7, **kw):
                self.attempts += 1
                if self.attempts < 3:
                    raise RuntimeError("connection reset")
                return LLMResponse(content="success", model="flakey", input_tokens=5, output_tokens=10)

            async def generate_stream(self, *args, **kwargs):
                return
                yield

        flakey = FlakeyClient()
        router = LLMRouter(
            routing={TASK_CONVERSE: "flakey"},
            fallbacks={TASK_CONVERSE: []},
            max_retries=3,
        )
        router.register_client("flakey", flakey)

        original_sleep = asyncio.sleep
        async def fast_sleep(s): pass
        asyncio.sleep = fast_sleep
        try:
            resp = await router.generate(TASK_CONVERSE, _msg())
            assert resp.content == "success"
            assert flakey.attempts == 3
        finally:
            asyncio.sleep = original_sleep

    @pytest.mark.asyncio
    async def test_fatal_error_not_retried(self):
        """Auth error → not retried, immediately tries fallback."""
        fatal   = FailingClient("invalid api key: wrong", fatal=True)
        success = MockLLMClient(default="fallback")

        router = LLMRouter(
            routing={TASK_CONVERSE: "fatal-model"},
            fallbacks={TASK_CONVERSE: ["good-model"]},
            max_retries=3,
        )
        router.register_client("fatal-model", fatal)
        router.register_client("good-model",  success)

        resp = await router.generate(TASK_CONVERSE, _msg())
        assert "fallback" in resp.content
        assert fatal.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_zero_no_retry(self):
        """max_retries=0 means one attempt per provider."""
        failing = FailingClient()
        router  = LLMRouter(
            routing={TASK_CONVERSE: "fail"},
            fallbacks={TASK_CONVERSE: []},
            max_retries=0,
        )
        router.register_client("fail", failing)
        with pytest.raises(AllProvidersFailedError):
            await router.generate(TASK_CONVERSE, _msg())
        assert failing.call_count == 1

class TestCircuitBreaker:

    def test_circuit_opens_after_threshold(self):
        stats = ModelStats(model="test")
        from truenorth.llm.router import _CIRCUIT_BREAKER_THRESHOLD
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            stats.record_error("error")
        assert stats.circuit_open is True

    def test_circuit_does_not_open_below_threshold(self):
        stats = ModelStats(model="test")
        from truenorth.llm.router import _CIRCUIT_BREAKER_THRESHOLD
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD - 1):
            stats.record_error("error")
        assert stats.circuit_open is False

    def test_success_resets_consecutive_errors(self):
        stats = ModelStats(model="test")
        stats.record_error("error")
        stats.record_error("error")
        stats.record_success(100, 50, 0.001)
        assert stats.consecutive_errors == 0
        assert stats.circuit_open is False

    @pytest.mark.asyncio
    async def test_circuit_open_provider_skipped(self):
        """Circuit-open primary → router goes straight to fallback."""
        success = MockLLMClient(default="fallback works")
        router  = LLMRouter(
            routing={TASK_CONVERSE: "broken"},
            fallbacks={TASK_CONVERSE: ["good"]},
            max_retries=0,
        )
        router.register_client("broken", FailingClient())
        router.register_client("good",   success)

        router._get_stats("broken").circuit_open = True

        resp = await router.generate(TASK_CONVERSE, _msg())
        assert "fallback works" in resp.content

    def test_reset_circuit(self):
        router = LLMRouter()
        stats  = router._get_stats("some-model")
        stats.circuit_open = True
        router.reset_circuit("some-model")
        assert stats.circuit_open is False

class TestBudgetGuard:

    @pytest.mark.asyncio
    async def test_request_rejected_when_over_budget(self):
        router = LLMRouter(budget_usd=0.000001)
        with pytest.raises(BudgetExceededError) as exc:
            await router.generate(TASK_OUTPUT, _msg("a" * 5000), max_tokens=2000)
        assert exc.value.estimated > exc.value.budget

    @pytest.mark.asyncio
    async def test_request_allowed_within_budget(self):
        mock   = MockLLMClient(default="ok")
        router = LLMRouter(budget_usd=10.0, max_retries=0)
        for m in _DEFAULT_ROUTING.values():
            router.register_client(m, mock)

        resp = await router.generate(TASK_EXTRACT, _msg("hi"), max_tokens=10)
        assert resp.content == "ok"

    @pytest.mark.asyncio
    async def test_per_request_budget_overrides_instance(self):
        mock   = MockLLMClient(default="ok")
        router = LLMRouter(budget_usd=10.0, max_retries=0)
        for m in _DEFAULT_ROUTING.values():
            router.register_client(m, mock)
        with pytest.raises(BudgetExceededError):
            await router.generate(
                TASK_OUTPUT,
                _msg("x" * 10000),
                max_tokens = 5000,
                budget_usd = 0.000001,
            )

    def test_budget_exceeded_error_has_amounts(self):
        err = BudgetExceededError(0.05, 0.01)
        assert err.estimated == pytest.approx(0.05)
        assert err.budget    == pytest.approx(0.01)

class TestProviderDetection:

    def _detect(self, model: str) -> str:
        return LLMRouter._detect_provider(model)

    def test_claude_is_anthropic(self):
        assert self._detect("claude-sonnet-4-20250514") == "anthropic"
        assert self._detect("claude-haiku-4-5-20251001") == "anthropic"

    def test_gpt_is_openai(self):
        assert self._detect("gpt-4o")      == "openai"
        assert self._detect("gpt-4o-mini") == "openai"

    def test_gemini_is_gemini(self):
        assert self._detect("gemini-3.5-flash") == "gemini"
        assert self._detect("gemini-2.0-flash") == "gemini"

    def test_ollama_is_local(self):
        assert self._detect("ollama")         == "local"
        assert self._detect("llama3.1")       == "local"
        assert self._detect("mistral")        == "local"
        assert self._detect("phi-3")          == "local"
        assert self._detect("deepseek-coder") == "local"

    def test_command_is_cohere(self):
        assert self._detect("command-r") == "cohere"

    def test_unknown_defaults_to_anthropic(self):
        assert self._detect("unknown-model-xyz") == "anthropic"

class TestClientRegistry:

    def test_register_and_retrieve_client(self):
        mock   = MockLLMClient(default="registered")
        router = LLMRouter()
        router.register_client("my-model", mock)
        client = router._get_client("my-model")
        assert client is mock

    def test_same_client_returned_for_same_model(self):
        router = LLMRouter()
        router.register_client("model-a", MockLLMClient())
        c1 = router._get_client("model-a")
        c2 = router._get_client("model-a")
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_registered_client_used_in_generate(self):
        mock   = MockLLMClient(default="from registry")
        router = LLMRouter(routing={TASK_EXTRACT: "registered-model"}, max_retries=0)
        router.register_client("registered-model", mock)
        resp = await router.generate(TASK_EXTRACT, _msg())
        assert "from registry" in resp.content

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_healthy_provider_returns_true(self):
        mock   = MockLLMClient(default="pong")
        router = LLMRouter()
        for m in _DEFAULT_ROUTING.values():
            router.register_client(m, mock)
        results = await router.health_check()
        assert all(results.values())

    @pytest.mark.asyncio
    async def test_failing_provider_returns_false(self):
        failing = FailingClient()
        router  = LLMRouter(routing={TASK_EXTRACT: "failing-model"})
        router.register_client("failing-model", failing)
        results = await router.health_check(tasks=[TASK_EXTRACT])
        assert results.get("failing-model") is False

    @pytest.mark.asyncio
    async def test_health_check_opens_circuit_for_failing(self):
        failing = FailingClient()
        router  = LLMRouter(routing={TASK_EXTRACT: "bad"})
        router.register_client("bad", failing)
        await router.health_check(tasks=[TASK_EXTRACT])
        assert router._get_stats("bad").circuit_open is True

class TestStats:

    @pytest.mark.asyncio
    async def test_stats_populated_after_call(self):
        mock   = MockLLMClient(default="ok")
        router = LLMRouter(routing={TASK_EXTRACT: "test-model"}, max_retries=0)
        router.register_client("test-model", mock)
        await router.generate(TASK_EXTRACT, _msg())
        stats = router.get_stats()
        assert "test-model" in stats
        assert stats["test-model"]["call_count"] == 1

    def test_model_stats_latency_tracking(self):
        stats = ModelStats(model="test")
        stats.record_success(100, 50, 0.001)
        stats.record_success(200, 50, 0.001)
        stats.record_success(300, 50, 0.001)
        assert stats.p50_ms in (100, 200, 300)

    def test_model_stats_error_rate(self):
        stats = ModelStats(model="test")
        stats.record_success(100, 50, 0.0)
        stats.record_error("err")
        assert stats.error_rate == pytest.approx(0.5)

    def test_stats_to_dict_structure(self):
        stats = ModelStats(model="test")
        d = stats.to_dict()
        for key in ["model", "call_count", "error_count", "error_rate",
                    "total_tokens", "total_cost_usd", "p50_ms", "circuit_open"]:
            assert key in d

    @pytest.mark.asyncio
    async def test_error_recorded_on_all_fail(self):
        router = LLMRouter(
            routing={TASK_CONVERSE: "bad"},
            fallbacks={TASK_CONVERSE: []},
            max_retries=0,
        )
        router.register_client("bad", FailingClient())
        try:
            await router.generate(TASK_CONVERSE, _msg())
        except AllProvidersFailedError:
            pass
        stats = router.get_stats()
        assert stats["bad"]["error_count"] >= 1

class TestStreaming:

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        mock   = MockLLMClient(default="hello world")
        router = LLMRouter(routing={TASK_CONVERSE: "stream-model"}, max_retries=0)
        router.register_client("stream-model", mock)

        chunks = []
        async for chunk in router.generate_stream(TASK_CONVERSE, _msg()):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_fallback_on_connection_failure(self):
        success = MockLLMClient(default="stream fallback")
        router  = LLMRouter(
            routing={TASK_CONVERSE: "bad-stream"},
            fallbacks={TASK_CONVERSE: ["good-stream"]},
        )
        router.register_client("bad-stream",  FailingClient())
        router.register_client("good-stream", success)

        chunks = []
        async for chunk in router.generate_stream(TASK_CONVERSE, _msg()):
            chunks.append(chunk)
        assert len(chunks) >= 1

class TestSectorAgnosticism:
    """
    Router works identically for any domain.
    The YAML changes; the router doesn't care.
    """

    SECTORS = [

        ("healthcare",    ["chief_complaint", "pain_scale", "medications"],
         "I have lower back pain, scale 7 out of 10, taking ibuprofen"),
        ("legal_intake",  ["case_type", "incident_date", "jurisdiction"],
         "Personal injury case, happened on 2024-03-15, in Karnataka"),
        ("hr_screening",  ["years_experience", "desired_salary", "notice_period"],
         "5 years experience, expecting 25 LPA, 30 days notice"),
        ("financial_plan",["annual_income", "risk_tolerance", "investment_horizon"],
         "Income 15 LPA, moderate risk, 10 year horizon"),
        ("fitness",       ["age", "weight_kg", "primary_goal"],
         "I am 28 years old, 65 kg, want to lose weight"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector,fields,message", SECTORS)
    async def test_sector_generates_successfully(self, sector, fields, message):
        """Router should work for any domain without code changes."""
        mock   = MockLLMClient(default=f"response for {sector}")
        router = LLMRouter(max_retries=0)
        for m in _DEFAULT_ROUTING.values():
            router.register_client(m, mock)

        resp = await router.generate(
            TASK_EXTRACT,
            [Message(role="user", content=message)],
        )
        assert sector in resp.content or "response" in resp.content

    @pytest.mark.asyncio
    async def test_same_router_handles_multiple_sectors_sequentially(self):
        """One router instance, many different goal types — no interference."""
        mock   = MockLLMClient(default="ok")
        router = LLMRouter(max_retries=0)
        for m in _DEFAULT_ROUTING.values():
            router.register_client(m, mock)

        for task in [TASK_EXTRACT, TASK_CONVERSE, TASK_OUTPUT, TASK_CLASSIFY]:
            resp = await router.generate(task, _msg("sector test"))
            assert resp.content == "ok"

        stats = router.get_stats()
        total_calls = sum(s["call_count"] for s in stats.values())
        assert total_calls == 4

class TestV1Compatibility:

    @pytest.mark.asyncio
    async def test_generate_same_signature(self):
        """v1 callers pass task, messages, system, max_tokens, temperature."""
        mock   = MockLLMClient(default="v1 compat")
        router = LLMRouter(max_retries=0)
        for m in _DEFAULT_ROUTING.values():
            router.register_client(m, mock)

        resp = await router.generate(
            task        = TASK_CONVERSE,
            messages    = _msg("hello"),
            system      = "You are a helpful assistant.",
            max_tokens  = 200,
            temperature = 0.5,
        )
        assert resp.content == "v1 compat"

    def test_register_client_still_works(self):
        mock   = MockLLMClient()
        router = LLMRouter()
        router.register_client("some-model", mock)
        assert router._get_client("some-model") is mock

    def test_routing_table_returns_dict(self):
        router = LLMRouter()
        rt = router.routing_table()
        assert isinstance(rt, dict)
        assert len(rt) >= 4

    def test_set_routing_still_works(self):
        router = LLMRouter()
        router.set_routing(TASK_EXTRACT, "gpt-4o-mini")
        assert router.routing_table()[TASK_EXTRACT] == "gpt-4o-mini"

    def test_task_constants_unchanged(self):
        assert TASK_EXTRACT  == "extract"
        assert TASK_CONVERSE == "converse"
        assert TASK_OUTPUT   == "output"
        assert TASK_CLASSIFY == "classify"
        assert TASK_OTHER    == "other"

    @pytest.mark.asyncio
    async def test_from_config_returns_router(self):
        router = LLMRouter.from_config({})
        assert isinstance(router, LLMRouter)

    @pytest.mark.asyncio
    async def test_from_env_returns_router(self):
        router = LLMRouter.from_env()
        assert isinstance(router, LLMRouter)
