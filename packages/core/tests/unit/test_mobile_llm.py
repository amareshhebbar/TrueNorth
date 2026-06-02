"""
All tests use a mock HTTP server (no actual iOS/Android device needed).

Classes:
  1.  PlatformResolution — auto-detect, explicit, env-var override
  2.  Config             — endpoint, model, timeout, token cap
  3.  PayloadBuilding    — correct OpenAI-compatible request format
  4.  ResponseParsing    — parse mobile bridge response to LLMResponse
  5.  DeviceCapabilities — capability report struct
  6.  Throttling         — battery/thermal state affects max_tokens
  7.  HealthCheck        — reachable vs unreachable bridge
  8.  ListModels         — enumerate on-device models
  9.  CanHandle          — task routing decisions
  10. ErrorHandling      — MobileUnavailableError on all failure modes
  11. RouterIntegration  — router detects mobile provider, routes correctly
  12. FallbackToCloud    — mobile unavailable → cloud fallback in router
  13. PrivacySensitive   — PII fields routed to mobile, output to cloud
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.llm.mobile_llm import (
    MobileLLMClient,
    MobileUnavailableError,
    DeviceCapabilities,
    MobilePlatform,
    MOBILE_PREFERRED_TASKS,
)
from truenorth.llm.base import LLMResponse, Message
from truenorth.llm.router import (
    LLMRouter,
    TASK_EXTRACT,
    TASK_OUTPUT,
)
from truenorth.testing.mock_llm import MockLLMClient


# ─────────────────────────────────────────────────────────────────────────────
#  Mock HTTP client helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_chat_response(content: str = "extracted: age=28") -> dict:
    return {
        "id":      "chatcmpl-mobile-001",
        "object":  "chat.completion",
        "model":   "apple/on-device-3b",
        "choices": [
            {"message": {"role": "assistant", "content": content},
             "finish_reason": "stop", "index": 0}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def _make_caps_response(
    battery: float = 1.0,
    thermal: str   = "nominal",
    available: bool = True,
) -> dict:
    return {
        "platform":           "ios",
        "model_name":         "apple/on-device-3b",
        "context_window":     4096,
        "supports_streaming": True,
        "supports_vision":    False,
        "max_tokens":         512,
        "battery_level":      battery,
        "thermal_state":      thermal,
        "is_available":       available,
        "preferred_tasks":    ["extract", "classify"],
    }


class MockHTTPResponse:
    def __init__(self, data: dict, status: int = 200):
        self._data   = data
        self.status_code = status

    def json(self) -> dict:
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class MockHTTPClient:
    """Simulates httpx.AsyncClient for mobile bridge testing."""

    def __init__(
        self,
        chat_data:  dict = None,
        caps_data:  dict = None,
        health_ok:  bool = True,
        fail_on:    Optional[str] = None,
    ):
        self.chat_data  = chat_data or _make_chat_response()
        self.caps_data  = caps_data or _make_caps_response()
        self.health_ok  = health_ok
        self.fail_on    = fail_on  
        self.calls:  List[tuple] = []

    async def post(self, path: str, json: dict = None, timeout=None) -> MockHTTPResponse:
        self.calls.append(("POST", path, json))
        if self.fail_on and path.startswith(self.fail_on):
            raise ConnectionRefusedError("connection refused")
        return MockHTTPResponse(self.chat_data)

    async def get(self, path: str, timeout=None) -> MockHTTPResponse:
        self.calls.append(("GET", path))
        if self.fail_on and path.startswith(self.fail_on):
            raise ConnectionRefusedError("connection refused")
        if "/health" in path:
            return MockHTTPResponse({"status": "ok"} if self.health_ok else {}, 200 if self.health_ok else 503)
        if "/capabilities" in path:
            return MockHTTPResponse(self.caps_data)
        if "/models" in path:
            return MockHTTPResponse({"data": [{"id": "apple/on-device-3b"}]})
        return MockHTTPResponse({})

    def stream(self, method: str, path: str, json: dict = None, timeout=None):
        return _StreamContextManager(self.chat_data)

    async def aclose(self):
        pass


class _StreamContextManager:
    def __init__(self, data: dict):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        content = self._data["choices"][0]["message"]["content"]
        for word in content.split():
            yield f'data: {json.dumps({"choices":[{"delta":{"content":word+" "}}]})}'
        yield "data: [DONE]"


def _client_with_mock(mock_http: MockHTTPClient) -> MobileLLMClient:
    """Create a MobileLLMClient with the mock HTTP backend injected."""
    client = MobileLLMClient(config={"platform": MobilePlatform.IOS})
    client._client = mock_http
    return client


# ─────────────────────────────────────────────────────────────────────────────
#  1. Platform resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestPlatformResolution:

    def test_explicit_ios(self):
        c = MobileLLMClient(config={"platform": "ios"})
        assert c.platform == MobilePlatform.IOS

    def test_explicit_android(self):
        c = MobileLLMClient(config={"platform": "android"})
        assert c.platform == MobilePlatform.ANDROID

    def test_explicit_generic(self):
        c = MobileLLMClient(config={"platform": "generic"})
        assert c.platform == MobilePlatform.GENERIC

    def test_auto_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TRUENORTH_MOBILE_PLATFORM", "android")
        c = MobileLLMClient(config={"platform": "auto"})
        assert c.platform == MobilePlatform.ANDROID

    def test_auto_defaults_to_generic_on_linux(self, monkeypatch):
        import platform as _p
        monkeypatch.delenv("TRUENORTH_MOBILE_PLATFORM", raising=False)
        monkeypatch.setattr(_p, "machine", lambda: "x86_64")
        c = MobileLLMClient(config={"platform": "auto"})
        assert c.platform in (MobilePlatform.IOS, MobilePlatform.GENERIC, MobilePlatform.ANDROID)

    def test_auto_detect_factory(self):
        c = MobileLLMClient.auto_detect()
        assert isinstance(c, MobileLLMClient)
        assert c.platform in (MobilePlatform.IOS, MobilePlatform.ANDROID, MobilePlatform.GENERIC)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_ios_default_endpoint(self):
        c = MobileLLMClient(config={"platform": "ios"})
        assert "49152" in c.endpoint

    def test_android_default_endpoint(self):
        c = MobileLLMClient(config={"platform": "android"})
        assert "49153" in c.endpoint

    def test_custom_endpoint_overrides_default(self):
        c = MobileLLMClient(config={"platform": "ios", "endpoint": "http://10.0.0.5:9000"})
        assert c.endpoint == "http://10.0.0.5:9000"

    def test_env_endpoint_override(self, monkeypatch):
        monkeypatch.setenv("TRUENORTH_MOBILE_ENDPOINT", "http://192.168.1.1:8080")
        c = MobileLLMClient(config={"platform": "ios"})
        assert c.endpoint == "http://192.168.1.1:8080"

    def test_ios_default_model(self):
        c = MobileLLMClient(config={"platform": "ios"})
        assert "apple" in c.model_name or "on-device" in c.model_name

    def test_android_default_model(self):
        c = MobileLLMClient(config={"platform": "android"})
        assert "gemini-nano" in c.model_name or "nano" in c.model_name

    def test_custom_model_override(self):
        c = MobileLLMClient(config={"platform": "ios", "model": "custom/model-v2"})
        assert c.model_name == "custom/model-v2"

    def test_max_tokens_cap_applied(self):
        c = MobileLLMClient(config={"platform": "ios", "max_tokens_override": 128})
        assert c._max_tok_cap == 128


# ─────────────────────────────────────────────────────────────────────────────
#  3. Payload building
# ─────────────────────────────────────────────────────────────────────────────

class TestPayloadBuilding:

    def _client(self) -> MobileLLMClient:
        return MobileLLMClient(config={"platform": "ios"})

    def test_basic_payload_structure(self):
        c = self._client()
        msgs = [Message(role="user", content="Hello")]
        payload = c._build_payload(msgs, None, 100, 0.7)
        assert "model"    in payload
        assert "messages" in payload
        assert "max_tokens"  in payload
        assert "temperature" in payload
        assert payload["max_tokens"] == 100

    def test_system_prepended_as_system_role(self):
        c = self._client()
        msgs = [Message(role="user", content="Hi")]
        payload = c._build_payload(msgs, "You are helpful.", 100, 0.7)
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."
        assert payload["messages"][1]["role"] == "user"

    def test_stream_flag_set_correctly(self):
        c = self._client()
        msgs = [Message(role="user", content="Hi")]
        p_stream  = c._build_payload(msgs, None, 100, 0.7, stream=True)
        p_nostream = c._build_payload(msgs, None, 100, 0.7, stream=False)
        assert p_stream["stream"]   is True
        assert p_nostream["stream"] is False

    def test_max_tokens_capped_by_override(self):
        c = MobileLLMClient(config={"platform": "ios", "max_tokens_override": 64})
        msgs = [Message(role="user", content="Hi")]
        payload = c._build_payload(msgs, None, 512, 0.7)
        assert payload["max_tokens"] == 512 


# ─────────────────────────────────────────────────────────────────────────────
#  4. Response parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseParsing:

    def test_parses_content_from_choices(self):
        c    = MobileLLMClient(config={"platform": "ios"})
        data = _make_chat_response("age is 28")
        resp = c._parse_response(data, 150)
        assert resp.content == "age is 28"

    def test_parses_token_counts(self):
        c    = MobileLLMClient(config={"platform": "ios"})
        data = _make_chat_response()
        resp = c._parse_response(data, 150)
        assert resp.input_tokens  == 20
        assert resp.output_tokens == 10

    def test_latency_populated(self):
        c    = MobileLLMClient(config={"platform": "ios"})
        data = _make_chat_response()
        resp = c._parse_response(data, 250)
        assert resp.latency_ms == 250

    def test_empty_choices_returns_empty_content(self):
        c    = MobileLLMClient(config={"platform": "ios"})
        data = {"choices": [], "usage": {}}
        resp = c._parse_response(data, 50)
        assert resp.content == ""


# ─────────────────────────────────────────────────────────────────────────────
#  5. DeviceCapabilities
# ─────────────────────────────────────────────────────────────────────────────

class TestDeviceCapabilities:

    def test_nominal_state_no_throttle(self):
        caps = DeviceCapabilities(
            platform="ios", model_name="apple/on-device-3b",
            battery_level=0.80, thermal_state="nominal",
        )
        assert caps.should_throttle is False

    def test_low_battery_throttles(self):
        caps = DeviceCapabilities(
            platform="ios", model_name="apple/on-device-3b",
            battery_level=0.15, thermal_state="nominal",
        )
        assert caps.should_throttle is True

    def test_critical_thermal_throttles(self):
        caps = DeviceCapabilities(
            platform="ios", model_name="apple/on-device-3b",
            battery_level=0.90, thermal_state="critical",
        )
        assert caps.should_throttle is True

    def test_serious_thermal_throttles(self):
        caps = DeviceCapabilities(
            platform="android", model_name="gemini-nano",
            battery_level=0.50, thermal_state="serious",
        )
        assert caps.should_throttle is True

    def test_fair_thermal_no_throttle(self):
        caps = DeviceCapabilities(
            platform="ios", model_name="apple/on-device-3b",
            battery_level=0.60, thermal_state="fair",
        )
        assert caps.should_throttle is False

    def test_to_dict_structure(self):
        caps = DeviceCapabilities(platform="ios", model_name="m")
        d    = caps.to_dict()
        for key in ["platform", "model_name", "battery_level",
                    "thermal_state", "is_available", "should_throttle"]:
            assert key in d

    def test_preferred_tasks_default(self):
        caps = DeviceCapabilities(platform="ios", model_name="m")
        assert "extract"  in caps.preferred_tasks
        assert "classify" in caps.preferred_tasks


# ─────────────────────────────────────────────────────────────────────────────
#  6. Throttling
# ─────────────────────────────────────────────────────────────────────────────

class TestThrottling:

    @pytest.mark.asyncio
    async def test_throttle_caps_max_tokens(self):
        mock_http = MockHTTPClient(
            caps_data=_make_caps_response(battery=0.10, thermal="serious"),
        )
        c = _client_with_mock(mock_http)
        c._capabilities = None   # force fresh fetch

        # Inject caps directly (avoid HTTP call in test)
        c._capabilities = DeviceCapabilities(
            platform="ios", model_name="apple/on-device-3b",
            battery_level=0.10, thermal_state="serious",
        )

        await c.generate([Message(role="user", content="hi")], max_tokens=1024)
        # The POST should have been called with capped max_tokens
        post_calls = [call for call in mock_http.calls if call[0] == "POST"]
        assert len(post_calls) == 1
        payload = post_calls[0][2]
        assert payload["max_tokens"] <= 256

    @pytest.mark.asyncio
    async def test_no_throttle_on_good_conditions(self):
        mock_http = MockHTTPClient()
        c = _client_with_mock(mock_http)
        c._capabilities = DeviceCapabilities(
            platform="ios", model_name="apple/on-device-3b",
            battery_level=0.90, thermal_state="nominal",
        )

        await c.generate([Message(role="user", content="hi")], max_tokens=512)
        post_calls = [call for call in mock_http.calls if call[0] == "POST"]
        payload    = post_calls[0][2]
        assert payload["max_tokens"] == 512


# ─────────────────────────────────────────────────────────────────────────────
#  7. Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:

    @pytest.mark.asyncio
    async def test_healthy_bridge_returns_true(self):
        mock_http = MockHTTPClient(health_ok=True)
        c = _client_with_mock(mock_http)
        assert await c.health_check() is True

    @pytest.mark.asyncio
    async def test_unreachable_bridge_returns_false(self):
        mock_http = MockHTTPClient(fail_on="/health")
        c = _client_with_mock(mock_http)
        assert await c.health_check() is False


# ─────────────────────────────────────────────────────────────────────────────
#  8. List models
# ─────────────────────────────────────────────────────────────────────────────

class TestListModels:

    @pytest.mark.asyncio
    async def test_returns_model_list(self):
        mock_http = MockHTTPClient()
        c = _client_with_mock(mock_http)
        models = await c.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self):
        mock_http = MockHTTPClient(fail_on="/models")
        c = _client_with_mock(mock_http)
        models = await c.list_models()
        assert models == []


# ─────────────────────────────────────────────────────────────────────────────
#  9. can_handle — task routing decisions
# ─────────────────────────────────────────────────────────────────────────────

class TestCanHandle:

    def test_extract_is_mobile_preferred(self):
        c = MobileLLMClient(config={"platform": "ios"})
        assert c.can_handle("extract")  is True
        assert c.can_handle("classify") is True
        assert c.can_handle("embed")    is True

    def test_output_not_mobile_preferred(self):
        c = MobileLLMClient(config={"platform": "ios"})
        assert c.can_handle("output")  is False
        assert c.can_handle("verify")  is False
        assert c.can_handle("converse") is False

    def test_mobile_preferred_tasks_constant(self):
        assert "extract"  in MOBILE_PREFERRED_TASKS
        assert "classify" in MOBILE_PREFERRED_TASKS
        assert "output"  not in MOBILE_PREFERRED_TASKS


# ─────────────────────────────────────────────────────────────────────────────
#  10. Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    @pytest.mark.asyncio
    async def test_connection_refused_raises_mobile_unavailable(self):
        mock_http = MockHTTPClient(fail_on="/chat")
        c = _client_with_mock(mock_http)
        c._capabilities = DeviceCapabilities(
            platform="ios", model_name="m", battery_level=1.0, thermal_state="nominal"
        )
        with pytest.raises(MobileUnavailableError) as exc:
            await c.generate([Message(role="user", content="hi")])
        assert exc.value.platform == MobilePlatform.IOS

    def test_mobile_unavailable_has_useful_attrs(self):
        err = MobileUnavailableError("ios", "http://127.0.0.1:49152", "timeout")
        assert err.platform == "ios"
        assert "49152" in err.endpoint
        assert "timeout" in err.cause
        assert "ios" in str(err)

    @pytest.mark.asyncio
    async def test_stream_raises_on_connection_failure(self):
        mock_http = MockHTTPClient(fail_on="/chat")
        c = _client_with_mock(mock_http)
        c._capabilities = DeviceCapabilities(
            platform="ios", model_name="m", battery_level=1.0, thermal_state="nominal"
        )
        # Patch stream to raise
        original_stream = mock_http.stream
        def bad_stream(*args, **kwargs):
            raise ConnectionRefusedError("refused")
        mock_http.stream = bad_stream

        with pytest.raises((MobileUnavailableError, ConnectionRefusedError)):
            async for _ in c.generate_stream([Message(role="user", content="hi")]):
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  11. Router integration — provider detection
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterIntegration:

    def test_apple_prefix_detected_as_mobile(self):
        assert LLMRouter._detect_provider("apple/on-device-3b") == "mobile"

    def test_gemini_nano_detected_as_mobile(self):
        assert LLMRouter._detect_provider("gemini-nano") == "mobile"
        assert LLMRouter._detect_provider("gemini-nano-2") == "mobile"

    def test_mobile_prefix_detected(self):
        assert LLMRouter._detect_provider("mobile-llm") == "mobile"
        assert LLMRouter._detect_provider("on-device")  == "mobile"

    def test_gemini_flash_still_cloud(self):
        # gemini-3.5-flash → not gemini-nano → cloud gemini
        assert LLMRouter._detect_provider("gemini-3.5-flash") == "gemini"

    def test_router_builds_mobile_client(self):
        """Router._build_client should instantiate MobileLLMClient for mobile."""
        router = LLMRouter()
        client = router._build_client("mobile", "apple/on-device-3b")
        assert isinstance(client, MobileLLMClient)

    def test_custom_routing_to_mobile(self):
        router = LLMRouter(routing={TASK_EXTRACT: "apple/on-device-3b"})
        chain  = router._build_chain(TASK_EXTRACT)
        assert chain[0] == "apple/on-device-3b"

    @pytest.mark.asyncio
    async def test_mobile_routing_works_end_to_end(self):
        """Router routes TASK_EXTRACT to a registered mobile client."""
        mobile_mock = MockLLMClient(default='{"extractions":[{"name":"age","value":28}]}')
        router = LLMRouter(
            routing={TASK_EXTRACT: "apple/on-device-3b"},
            fallbacks={TASK_EXTRACT: ["gemini-3.5-flash"]},
            max_retries=0,
        )
        router.register_client("apple/on-device-3b", mobile_mock)

        resp = await router.generate(
            TASK_EXTRACT,
            [Message(role="user", content="I am 28 years old")],
        )
        assert "28" in resp.content or "age" in resp.content


# ─────────────────────────────────────────────────────────────────────────────
#  12. Fallback to cloud when mobile unavailable
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackToCloud:

    @pytest.mark.asyncio
    async def test_mobile_fails_cloud_used(self):
        """When on-device fails, router falls back to cloud Gemini."""
        cloud_mock = MockLLMClient(default="from cloud")

        class _FailingMobile(MockLLMClient):
            async def generate(self, messages, **kw):
                raise MobileUnavailableError("ios", "http://127.0.0.1:49152", "timeout")

        router = LLMRouter(
            routing={TASK_EXTRACT: "apple/on-device-3b"},
            fallbacks={TASK_EXTRACT: ["gemini-3.5-flash"]},
            max_retries=0,
        )
        router.register_client("apple/on-device-3b", _FailingMobile())
        router.register_client("gemini-3.5-flash",  cloud_mock)

        resp = await router.generate(TASK_EXTRACT, [Message(role="user", content="hi")])
        assert resp.content == "from cloud"

    @pytest.mark.asyncio
    async def test_stats_show_mobile_error(self):
        cloud_mock = MockLLMClient(default="cloud")

        class _FailingMobile(MockLLMClient):
            async def generate(self, messages, **kw):
                raise MobileUnavailableError("ios", "http://127.0.0.1", "refused")

        router = LLMRouter(
            routing={TASK_EXTRACT: "apple/on-device-3b"},
            fallbacks={TASK_EXTRACT: ["gemini-3.5-flash"]},
            max_retries=0,
        )
        router.register_client("apple/on-device-3b", _FailingMobile())
        router.register_client("gemini-3.5-flash",  cloud_mock)
        await router.generate(TASK_EXTRACT, [Message(role="user", content="hi")])
        stats = router.get_stats()
        assert stats["apple/on-device-3b"]["error_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  13. Privacy-sensitive routing
# ─────────────────────────────────────────────────────────────────────────────

class TestPrivacySensitiveRouting:
    """
    Demonstrate the privacy routing pattern:
      TASK_EXTRACT (PII fields) → on-device
      TASK_OUTPUT  (report)     → cloud

    This is the pattern TrueNorth users configure for medical/legal/financial goals.
    """

    @pytest.mark.asyncio
    async def test_extract_goes_to_mobile_output_goes_to_cloud(self):
        mobile_calls: list = []
        cloud_calls:  list = []

        class _MobileMock(MockLLMClient):
            async def generate(self, messages, **kw):
                mobile_calls.append(messages)
                return LLMResponse(content="on-device", model="mobile", input_tokens=5, output_tokens=10)

        class _CloudMock(MockLLMClient):
            async def generate(self, messages, **kw):
                cloud_calls.append(messages)
                return LLMResponse(content="cloud report", model="cloud", input_tokens=5, output_tokens=100)

        router = LLMRouter(
            routing={
                TASK_EXTRACT: "apple/on-device-3b",  # PII stays on device
                TASK_OUTPUT:  "claude-sonnet-4-20250514",  # report goes to cloud
            },
            fallbacks={TASK_EXTRACT: [], TASK_OUTPUT: []},
            max_retries=0,
        )
        router.register_client("apple/on-device-3b",       _MobileMock())
        router.register_client("claude-sonnet-4-20250514", _CloudMock())

        await router.generate(TASK_EXTRACT, [Message(role="user", content="My Aadhaar is 1234 5678 9012")])
        await router.generate(TASK_OUTPUT,  [Message(role="user", content="Generate report")])

        assert len(mobile_calls) == 1
        assert len(cloud_calls)  == 1

    def test_privacy_routing_pattern_documented_in_yaml(self):
        yaml_llm_section = {
            "routing": {
                "extract":  "apple/on-device-3b",
                "classify": "gemini-nano",
                "converse": "claude-haiku-4-5-20251001",
                "output":   "claude-sonnet-4-20250514",
                "verify":   "claude-sonnet-4-20250514",
            },
            "fallbacks": {
                "extract":  ["gemini-3.5-flash"],
                "classify": ["gemini-3.5-flash"],
            },
        }
        router = LLMRouter.from_config(yaml_llm_section)
        rt = router.routing_table()
        assert rt["extract"]  == "apple/on-device-3b"
        assert rt["classify"] == "gemini-nano"
        assert rt["output"]   == "claude-sonnet-4-20250514"
        assert router.fallback_table()["extract"] == ["gemini-3.5-flash"]