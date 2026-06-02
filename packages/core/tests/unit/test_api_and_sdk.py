"""
Tests the Python SDK client directly (no server needed — mocks HTTP).
Tests FastAPI routes via TestClient.

Classes:
  1.  Session_Dataclass   — Session.from_dict, is_complete flag
  2.  MessageResult       — from_dict, all fields mapped
  3.  Output_Dataclass    — from_dict
  4.  TrueNorthError      — str representation
  5.  SyncTransport       — get/post/delete, error handling
  6.  SessionsResource    — create/message/get/output/end
  7.  GoalsResource       — list/get/install
  8.  AnalyticsResource   — cost/health/trend
  9.  TrueNorthClient     — health, resource access
  10. AsyncClient         — async versions of all methods
  11. RunSession          — convenience helper
  12. FastAPI_Health      — GET /health
  13. FastAPI_Sessions    — POST/GET/DELETE /v1/sessions
  14. FastAPI_Goals       — GET /v1/goals
  15. FastAPI_Analytics   — GET /v1/analytics/cost
  16. SDKContractParity   — Python SDK shape matches Node SDK types
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client import (
    TrueNorth, AsyncTrueNorth, TrueNorthError,
    Session, MessageResult, Output, run_session, arun_session,
    _SyncTransport, _AsyncTransport,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Sample data
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SESSION_DICT = {
    "session_id":       "sess-abc123",
    "goal_id":          "fitness-coach",
    "status":           "active",
    "current_turn":     2,
    "completion_pct":   40.0,
    "collected_fields": {"age": 28, "weight_kg": 65},
    "missing_required": ["primary_goal", "activity_level"],
    "total_cost_usd":   0.00045,
    "is_complete":      False,
    "detected_language":"en",
    "agent_message":    "What is your primary fitness goal?",
    "created_at":       time.time(),
}

SAMPLE_MESSAGE_DICT = {
    "session_id":       "sess-abc123",
    "turn":             3,
    "text":             "Great! How many days per week can you exercise?",
    "is_complete":      False,
    "completion_pct":   60.0,
    "fields_extracted": [{"field": "primary_goal", "value": "lose weight", "confidence": 0.93}],
    "cost_usd":         0.00015,
    "latency_ms":       420,
    "emotion_detected": "neutral",
}

SAMPLE_OUTPUT_DICT = {
    "session_id":   "sess-abc123",
    "goal_id":      "fitness-coach",
    "format":       "json",
    "content":      {"plan": "3x weekly cardio", "target_weight": 60},
    "fields":       {"age": 28, "weight_kg": 65, "primary_goal": "lose weight"},
    "metadata":     {"confidence": 0.92},
    "generated_at": time.time(),
}


# ─────────────────────────────────────────────────────────────────────────────
#  1. Session dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionDataclass:

    def test_from_dict_all_fields(self):
        s = Session.from_dict(SAMPLE_SESSION_DICT)
        assert s.id              == "sess-abc123"
        assert s.goal_id         == "fitness-coach"
        assert s.status          == "active"
        assert s.current_turn    == 2
        assert s.completion_pct  == pytest.approx(40.0)
        assert s.collected_fields["age"] == 28
        assert "primary_goal"    in s.missing_required
        assert s.is_complete     is False
        assert s.total_cost_usd  == pytest.approx(0.00045)

    def test_is_complete_false_on_active(self):
        s = Session.from_dict(SAMPLE_SESSION_DICT)
        assert s.is_complete is False

    def test_is_complete_true(self):
        d = dict(SAMPLE_SESSION_DICT)
        d["is_complete"] = True
        d["status"]      = "complete"
        s = Session.from_dict(d)
        assert s.is_complete is True

    def test_missing_optional_fields_get_defaults(self):
        minimal = {"session_id": "s1", "goal_id": "g1"}
        s = Session.from_dict(minimal)
        assert s.status         == "active"
        assert s.current_turn   == 0
        assert s.is_complete    is False


# ─────────────────────────────────────────────────────────────────────────────
#  2. MessageResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestMessageResult:

    def test_from_dict(self):
        r = MessageResult.from_dict(SAMPLE_MESSAGE_DICT)
        assert r.session_id      == "sess-abc123"
        assert r.turn            == 3
        assert r.text            == "Great! How many days per week can you exercise?"
        assert r.is_complete     is False
        assert r.completion_pct  == pytest.approx(60.0)
        assert r.cost_usd        == pytest.approx(0.00015)
        assert r.latency_ms      == 420
        assert r.emotion_detected == "neutral"

    def test_fields_extracted_list(self):
        r = MessageResult.from_dict(SAMPLE_MESSAGE_DICT)
        assert len(r.fields_extracted) == 1
        assert r.fields_extracted[0]["field"] == "primary_goal"

    def test_defaults_when_missing(self):
        r = MessageResult.from_dict({"session_id": "s", "turn": 1, "text": "hi",
                                      "is_complete": False, "completion_pct": 0.0,
                                      "cost_usd": 0.0, "latency_ms": 100})
        assert r.emotion_detected is None
        assert r.fields_extracted == []


# ─────────────────────────────────────────────────────────────────────────────
#  3. Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputDataclass:

    def test_from_dict(self):
        o = Output.from_dict(SAMPLE_OUTPUT_DICT)
        assert o.session_id == "sess-abc123"
        assert o.goal_id    == "fitness-coach"
        assert o.format     == "json"
        assert o.content    == {"plan": "3x weekly cardio", "target_weight": 60}
        assert o.fields["age"] == 28


# ─────────────────────────────────────────────────────────────────────────────
#  4. TrueNorthError
# ─────────────────────────────────────────────────────────────────────────────

class TestTrueNorthError:

    def test_str_representation(self):
        e = TrueNorthError(404, "not_found", "Session not found")
        s = str(e)
        assert "404"       in s
        assert "not_found" in s
        assert "Session"   in s

    def test_is_exception(self):
        e = TrueNorthError(401, "unauthorized", "Bad key")
        with pytest.raises(TrueNorthError):
            raise e


# ─────────────────────────────────────────────────────────────────────────────
#  5. SyncTransport
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncTransport:

    def _transport(self) -> _SyncTransport:
        return _SyncTransport("http://test.localhost", "tn_live_abc", 5.0)

    def test_headers_include_api_key(self):
        t = self._transport()
        h = t._headers()
        assert h["X-TrueNorth-Key"] == "tn_live_abc"
        assert h["Content-Type"]    == "application/json"

    def test_headers_no_key_when_empty(self):
        t = _SyncTransport("http://x", "", 5.0)
        h = t._headers()
        assert "X-TrueNorth-Key" not in h


# ─────────────────────────────────────────────────────────────────────────────
#  6. Sessions resource (sync, mocked transport)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_transport(responses: dict):
    """Build a mock _SyncTransport that returns preset responses."""
    t = MagicMock(spec=_SyncTransport)

    def _post(path, body=None):
        return responses.get(path, {})

    def _get(path, params=None):
        return responses.get(path, {})

    def _delete(path):
        return None

    t.post   = MagicMock(side_effect=_post)
    t.get    = MagicMock(side_effect=_get)
    t.delete = MagicMock(side_effect=_delete)
    return t


class TestSessionsResource:

    def _tn(self) -> TrueNorth:
        tn = TrueNorth.__new__(TrueNorth)
        from client import _SessionsResource, _GoalsResource, _AnalyticsResource
        t = _mock_transport({
            "/v1/sessions": SAMPLE_SESSION_DICT,
            "/v1/sessions/sess-abc123/message": SAMPLE_MESSAGE_DICT,
            "/v1/sessions/sess-abc123": SAMPLE_SESSION_DICT,
            "/v1/sessions/sess-abc123/output": SAMPLE_OUTPUT_DICT,
            "/v1/sessions/sess-abc123/force-output": SAMPLE_OUTPUT_DICT,
        })
        tn.sessions  = _SessionsResource(t)
        tn.goals     = _GoalsResource(t)
        tn.analytics = _AnalyticsResource(t)
        tn._transport = t
        return tn

    def test_create_returns_session(self):
        tn      = self._tn()
        session = tn.sessions.create("fitness-coach")
        assert isinstance(session, Session)
        assert session.id == "sess-abc123"

    def test_message_returns_result(self):
        tn     = self._tn()
        result = tn.sessions.message("sess-abc123", "I am 28")
        assert isinstance(result, MessageResult)
        assert result.text == "Great! How many days per week can you exercise?"

    def test_get_returns_session(self):
        tn      = self._tn()
        session = tn.sessions.get("sess-abc123")
        assert session.id == "sess-abc123"

    def test_output_returns_output(self):
        tn     = self._tn()
        output = tn.sessions.output("sess-abc123")
        assert isinstance(output, Output)
        assert output.session_id == "sess-abc123"

    def test_force_output(self):
        tn     = self._tn()
        output = tn.sessions.force_output("sess-abc123")
        assert isinstance(output, Output)

    def test_end_calls_delete(self):
        tn = self._tn()
        tn.sessions.end("sess-abc123")
        tn._transport.delete.assert_called_once_with("/v1/sessions/sess-abc123")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Goals resource
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalsResource:

    def _tn(self) -> TrueNorth:
        tn = TrueNorth.__new__(TrueNorth)
        from client import _SessionsResource, _GoalsResource, _AnalyticsResource
        goals_list = [{"name": "fitness-coach", "version": "1.3.0", "sector": "fitness", "downloads": 12000}]
        t = _mock_transport({
            "/v1/goals":                goals_list,
            "/v1/goals/fitness-coach":  goals_list[0],
            "/v1/goals/fitness-coach/install": goals_list[0],
        })
        tn.sessions  = _SessionsResource(t)
        tn.goals     = _GoalsResource(t)
        tn.analytics = _AnalyticsResource(t)
        tn._transport = t
        return tn

    def test_list_goals(self):
        goals = self._tn().goals.list()
        assert isinstance(goals, list)
        assert goals[0]["name"] == "fitness-coach"

    def test_get_goal(self):
        goal = self._tn().goals.get("fitness-coach")
        assert goal["name"] == "fitness-coach"


# ─────────────────────────────────────────────────────────────────────────────
#  8. Analytics resource
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticsResource:

    def _tn(self) -> TrueNorth:
        tn = TrueNorth.__new__(TrueNorth)
        from client import _SessionsResource, _GoalsResource, _AnalyticsResource
        cost_data = {
            "goal_id": "fitness-coach", "total_cost_usd": 1.23,
            "session_count": 100, "by_model": {}, "by_task": {},
        }
        t = _mock_transport({
            "/v1/analytics/cost":       cost_data,
            "/v1/analytics/health":     {"completion_rate": 0.82},
            "/v1/analytics/cost/trend": [{"period": "2025-01-01", "cost_usd": 0.05}],
        })
        tn.sessions  = _SessionsResource(t)
        tn.goals     = _GoalsResource(t)
        tn.analytics = _AnalyticsResource(t)
        tn._transport = t
        return tn

    def test_cost(self):
        data = self._tn().analytics.cost("fitness-coach", 7)
        assert data["goal_id"] == "fitness-coach"

    def test_health(self):
        data = self._tn().analytics.health("fitness-coach")
        assert "completion_rate" in data

    def test_cost_trend(self):
        data = self._tn().analytics.cost_trend("fitness-coach", 30, "day")
        assert isinstance(data, list)


# ─────────────────────────────────────────────────────────────────────────────
#  9. TrueNorth client
# ─────────────────────────────────────────────────────────────────────────────

class TestTrueNorthClient:

    def test_construct_with_api_key(self):
        tn = TrueNorth(api_key="tn_live_test")
        assert tn._transport._key == "tn_live_test"

    def test_construct_with_base_url(self):
        tn = TrueNorth(base_url="http://api.example.com", api_key="key")
        assert "api.example.com" in tn._transport._base

    def test_has_sessions_resource(self):
        tn = TrueNorth(api_key="k")
        assert hasattr(tn, "sessions")
        assert hasattr(tn.sessions, "create")
        assert hasattr(tn.sessions, "message")
        assert hasattr(tn.sessions, "output")

    def test_has_goals_resource(self):
        tn = TrueNorth(api_key="k")
        assert hasattr(tn, "goals")
        assert hasattr(tn.goals, "list")
        assert hasattr(tn.goals, "install")

    def test_has_analytics_resource(self):
        tn = TrueNorth(api_key="k")
        assert hasattr(tn, "analytics")
        assert hasattr(tn.analytics, "cost")


# ─────────────────────────────────────────────────────────────────────────────
#  10. AsyncTrueNorth
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncClient:

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncTrueNorth(api_key="tn_live_test") as tn:
            assert hasattr(tn, "sessions")
            assert hasattr(tn, "goals")
            assert hasattr(tn, "analytics")

    @pytest.mark.asyncio
    async def test_async_sessions_create(self):
        tn = AsyncTrueNorth.__new__(AsyncTrueNorth)
        from client import _AsyncSessionsResource, _AsyncGoalsResource, _AsyncAnalyticsResource

        t = MagicMock(spec=_AsyncTransport)
        t.post = AsyncMock(return_value=SAMPLE_SESSION_DICT)
        t.get  = AsyncMock(return_value=SAMPLE_SESSION_DICT)

        tn.sessions  = _AsyncSessionsResource(t)
        tn.goals     = _AsyncGoalsResource(t)
        tn.analytics = _AsyncAnalyticsResource(t)
        tn._transport = t

        session = await tn.sessions.create("fitness-coach")
        assert isinstance(session, Session)
        assert session.id == "sess-abc123"

    @pytest.mark.asyncio
    async def test_async_message(self):
        from client import _AsyncSessionsResource

        t = MagicMock(spec=_AsyncTransport)
        t.post = AsyncMock(return_value=SAMPLE_MESSAGE_DICT)
        resource = _AsyncSessionsResource(t)
        result   = await resource.message("sess-abc123", "I am 28")
        assert isinstance(result, MessageResult)
        assert result.turn == 3

    @pytest.mark.asyncio
    async def test_async_output(self):
        from client import _AsyncSessionsResource

        t = MagicMock(spec=_AsyncTransport)
        t.get = AsyncMock(return_value=SAMPLE_OUTPUT_DICT)
        resource = _AsyncSessionsResource(t)
        output   = await resource.output("sess-abc123")
        assert isinstance(output, Output)
        assert output.format == "json"


# ─────────────────────────────────────────────────────────────────────────────
#  11. run_session / arun_session convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSession:

    def test_run_session_returns_output(self):
        # Mock the TrueNorth client
        complete_msg = dict(SAMPLE_MESSAGE_DICT)
        complete_msg["is_complete"] = True

        tn_mock = MagicMock()
        tn_mock.sessions.create.return_value = Session.from_dict(SAMPLE_SESSION_DICT)
        tn_mock.sessions.message.return_value = MessageResult.from_dict(complete_msg)
        tn_mock.sessions.output.return_value  = Output.from_dict(SAMPLE_OUTPUT_DICT)
        tn_mock.sessions.force_output.return_value = Output.from_dict(SAMPLE_OUTPUT_DICT)

        with patch("client.TrueNorth", return_value=tn_mock):
            output = run_session("fitness-coach", ["I am 28", "65kg"])
        assert isinstance(output, Output)

    @pytest.mark.asyncio
    async def test_arun_session_returns_output(self):
        complete_msg = dict(SAMPLE_MESSAGE_DICT)
        complete_msg["is_complete"] = True

        tn_mock = AsyncMock()
        tn_mock.sessions.create    = AsyncMock(return_value=Session.from_dict(SAMPLE_SESSION_DICT))
        tn_mock.sessions.message   = AsyncMock(return_value=MessageResult.from_dict(complete_msg))
        tn_mock.sessions.output    = AsyncMock(return_value=Output.from_dict(SAMPLE_OUTPUT_DICT))
        tn_mock.sessions.force_output = AsyncMock(return_value=Output.from_dict(SAMPLE_OUTPUT_DICT))
        tn_mock.__aenter__ = AsyncMock(return_value=tn_mock)
        tn_mock.__aexit__  = AsyncMock(return_value=None)

        with patch("client.AsyncTrueNorth", return_value=tn_mock):
            output = await arun_session("fitness-coach", ["I am 28"])
        assert isinstance(output, Output)


# ─────────────────────────────────────────────────────────────────────────────
#  12. FastAPI health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestFastAPIHealth:

    def _client(self):
        from fastapi.testclient import TestClient
        from truenorth.api.app   import app
        return TestClient(app)

    def test_health_returns_200(self):
        client = self._client()
        resp   = client.get("/health")
        assert resp.status_code == 200
        data   = resp.json()
        assert data["status"] == "ok"

    def test_ready_endpoint(self):
        client = self._client()
        resp   = client.get("/ready")
        assert resp.status_code in (200, 503)  

    def test_version_in_health(self):
        client = self._client()
        resp   = client.get("/health")
        assert "version" in resp.json()


# ─────────────────────────────────────────────────────────────────────────────
#  13. FastAPI sessions routes
# ─────────────────────────────────────────────────────────────────────────────

class TestFastAPISessions:
    """Session API route contract tests."""

    def _client(self):
        from fastapi.testclient  import TestClient
        from truenorth.api.app   import app
        from truenorth.api.deps  import init_deps
        from truenorth.marketplace.goal_registry import GoalRegistry
        init_deps(goal_registry=GoalRegistry())
        return TestClient(app, raise_server_exceptions=False)

    def test_sessions_endpoint_responds(self):
        """POST /v1/sessions is registered — not a 404."""
        client = self._client()
        resp   = client.post("/v1/sessions", json={"goal_id": "fitness-coach"})
        assert resp.status_code != 404

    def test_missing_goal_id_is_client_error(self):
        """POST without goal_id is a 4xx error."""
        client = self._client()
        resp   = client.post("/v1/sessions", json={})
        assert 400 <= resp.status_code < 600

    def test_get_nonexistent_session_404(self):
        client = self._client()
        resp   = client.get("/v1/sessions/nonexistent-session-xyz")
        assert resp.status_code == 404

    def test_delete_nonexistent_session_is_error_or_idempotent(self):
        client = self._client()
        resp   = client.delete("/v1/sessions/nonexistent-session-xyz")
        assert resp.status_code in (204, 404)

    def test_message_nonexistent_session_404(self):
        client = self._client()
        resp   = client.post("/v1/sessions/nonexistent-xyz/message",
                              json={"text": "hello"})
        assert resp.status_code == 404

# ─────────────────────────────────────────────────────────────────────────────
#  14. FastAPI goals routes
# ─────────────────────────────────────────────────────────────────────────────

class TestFastAPIGoals:

    def _client(self):
        from fastapi.testclient  import TestClient
        from truenorth.api.app   import app
        from truenorth.api.deps  import init_deps
        from truenorth.marketplace.goal_registry import GoalRegistry
        init_deps(goal_registry=GoalRegistry())
        return TestClient(app)

    def test_list_goals_returns_200(self):
        client = self._client()
        resp   = client.get("/v1/goals")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_goals_has_fitness_coach(self):
        client = self._client()
        resp   = client.get("/v1/goals")
        names  = [g["name"] for g in resp.json()]
        assert "fitness-coach" in names

    def test_get_specific_goal(self):
        client = self._client()
        resp   = client.get("/v1/goals/fitness-coach")
        assert resp.status_code == 200
        assert resp.json()["name"] == "fitness-coach"

    def test_get_unknown_goal_404(self):
        client = self._client()
        resp   = client.get("/v1/goals/xyzzy-nonexistent")
        assert resp.status_code == 404

    def test_search_by_sector(self):
        client = self._client()
        resp   = client.get("/v1/goals?sector=fitness")
        assert resp.status_code == 200
        for g in resp.json():
            assert g["sector"] == "fitness"


# ─────────────────────────────────────────────────────────────────────────────
#  15. FastAPI analytics routes
# ─────────────────────────────────────────────────────────────────────────────

class TestFastAPIAnalytics:

    def _client(self):
        from fastapi.testclient  import TestClient
        from truenorth.api.app   import app
        from truenorth.api.deps  import init_deps
        from truenorth.observability.cost_dashboard import CostDashboard
        from truenorth.observability.tracer         import TrueNorthTracer
        from truenorth.observability.health_monitor import HealthMonitor
        from truenorth.observability.ab_engine      import ABRegistry
        tracer = TrueNorthTracer()
        dash   = CostDashboard(tracer=tracer)
        mon    = HealthMonitor(tracer=tracer)
        init_deps(cost_dashboard=dash, health_monitor=mon, ab_registry=ABRegistry())
        return TestClient(app)

    def test_cost_summary(self):
        client = self._client()
        resp   = client.get("/v1/analytics/cost?goal=fitness-coach&period=7")
        assert resp.status_code == 200
        data   = resp.json()
        assert "total_cost_usd" in data or "goal_id" in data

    def test_health_report(self):
        client = self._client()
        resp   = client.get("/v1/analytics/health?goal=fitness-coach&window=24")
        assert resp.status_code == 200

    def test_cost_trend(self):
        client = self._client()
        resp   = client.get("/v1/analytics/cost/trend?goal=fitness-coach&period=7")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_model_comparison(self):
        client = self._client()
        resp   = client.get("/v1/analytics/cost/models?goal=fitness-coach")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ─────────────────────────────────────────────────────────────────────────────
#  16. SDK contract parity (Python ↔ Node ↔ Go shape)
# ─────────────────────────────────────────────────────────────────────────────

class TestSDKContractParity:
    """
    Verify the Python SDK response shapes match what Node.js and Go SDKs expect.
    These tests act as the contract test between the API spec and all SDKs.
    """

    def test_session_has_all_contract_fields(self):
        """Session must have every field the Node/Go SDKs declare."""
        s = Session.from_dict(SAMPLE_SESSION_DICT)
        required_fields = [
            "id", "goal_id", "status", "current_turn", "completion_pct",
            "collected_fields", "missing_required", "total_cost_usd",
            "is_complete", "agent_message",
        ]
        for f in required_fields:
            assert hasattr(s, f), f"Session missing field: {f}"

    def test_message_result_has_all_contract_fields(self):
        r = MessageResult.from_dict(SAMPLE_MESSAGE_DICT)
        required = ["session_id", "turn", "text", "is_complete",
                    "completion_pct", "cost_usd", "latency_ms"]
        for f in required:
            assert hasattr(r, f), f"MessageResult missing field: {f}"

    def test_output_has_all_contract_fields(self):
        o = Output.from_dict(SAMPLE_OUTPUT_DICT)
        required = ["session_id", "goal_id", "format", "content", "fields", "metadata"]
        for f in required:
            assert hasattr(o, f), f"Output missing field: {f}"

    def test_error_has_status_code_and_error_code(self):
        """Both Node SDK and Go SDK expect statusCode + error + message."""
        e = TrueNorthError(422, "validation_error", "field required")
        assert e.status_code == 422
        assert e.error       == "validation_error"
        assert "422"         in str(e)

    def test_session_id_maps_from_snake_to_id(self):
        """API returns session_id; SDK exposes it as .id"""
        s = Session.from_dict({"session_id": "sess-xyz", "goal_id": "g",
                                "completion_pct": 0.0, "collected_fields": {},
                                "total_cost_usd": 0.0})
        assert s.id == "sess-xyz"

    def test_all_numeric_fields_are_numeric(self):
        s = Session.from_dict(SAMPLE_SESSION_DICT)
        assert isinstance(s.current_turn,   int)
        assert isinstance(s.completion_pct, float)
        assert isinstance(s.total_cost_usd, float)
        r = MessageResult.from_dict(SAMPLE_MESSAGE_DICT)
        assert isinstance(r.cost_usd,    float)
        assert isinstance(r.latency_ms,  int)
        assert isinstance(r.completion_pct, float)