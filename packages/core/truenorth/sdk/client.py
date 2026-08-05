"""
TrueNorth Python SDK — sync and async clients.

Install from PyPI:
    pip install truenorth-sdk

Usage (sync):
    from truenorth_sdk import TrueNorth

    tn      = TrueNorth(api_key="tn_live_...", base_url="http://localhost:8000")
    session = tn.sessions.create("fitness-coach")

    while not session.is_complete:
        user_input = input(session.agent_message + "\\n> ")
        session    = tn.sessions.message(session.id, user_input)

    output = tn.sessions.output(session.id)
    print(output.content)

Same interface as sdk-node and sdk-go — one mental model across all languages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Session:
    """A TrueNorth conversation session."""
    id:               str
    goal_id:          str
    status:           str
    current_turn:     int
    completion_pct:   float
    collected_fields: Dict[str, Any]
    missing_required: List[str]
    total_cost_usd:   float
    is_complete:      bool
    detected_language: Optional[str]      = None
    agent_message:    str                 = ""
    created_at:       float               = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        return cls(
            id               = d["session_id"],
            goal_id          = d["goal_id"],
            status           = d.get("status", "active"),
            current_turn     = d.get("current_turn", 0),
            completion_pct   = d.get("completion_pct", 0.0),
            collected_fields = d.get("collected_fields", {}),
            missing_required = d.get("missing_required", []),
            total_cost_usd   = d.get("total_cost_usd", 0.0),
            is_complete      = d.get("is_complete", False),
            detected_language= d.get("detected_language"),
            created_at       = d.get("created_at", 0.0),
        )

@dataclass
class MessageResult:
    """Result of sending a message to a session."""
    session_id:       str
    turn:             int
    text:             str
    is_complete:      bool
    completion_pct:   float
    fields_extracted: List[dict]
    cost_usd:         float
    latency_ms:       int
    emotion_detected: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "MessageResult":
        return cls(
            session_id      = d["session_id"],
            turn            = d["turn"],
            text            = d.get("text", ""),
            is_complete     = d.get("is_complete", False),
            completion_pct  = d.get("completion_pct", 0.0),
            fields_extracted= d.get("fields_extracted", []),
            cost_usd        = d.get("cost_usd", 0.0),
            latency_ms      = d.get("latency_ms", 0),
            emotion_detected= d.get("emotion_detected"),
        )

@dataclass
class Output:
    """The final structured output from a completed session."""
    session_id:  str
    goal_id:     str
    format:      str
    content:     Any
    fields:      Dict[str, Any]
    metadata:    Dict[str, Any]
    generated_at: float

    @classmethod
    def from_dict(cls, d: dict) -> "Output":
        return cls(
            session_id   = d["session_id"],
            goal_id      = d["goal_id"],
            format       = d.get("format", "json"),
            content      = d.get("content"),
            fields       = d.get("fields", {}),
            metadata     = d.get("metadata", {}),
            generated_at = d.get("generated_at", 0.0),
        )

@dataclass
class TrueNorthError(Exception):
    """API error with HTTP status and structured body."""
    status_code: int
    error:       str
    message:     str

    def __str__(self) -> str:
        return f"TrueNorthError({self.status_code}): {self.error} — {self.message}"

class _SyncTransport:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self._base    = base_url.rstrip("/")
        self._key     = api_key
        self._timeout = timeout

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            h["X-TrueNorth-Key"] = self._key
        return h

    def _raise(self, status: int, body: bytes) -> None:
        try:
            d = json.loads(body)
            raise TrueNorthError(status, d.get("error", "unknown"), d.get("message", body.decode()[:200]))
        except (json.JSONDecodeError, TypeError):
            raise TrueNorthError(status, "http_error", body.decode()[:200])

    def get(self, path: str, params: dict = None) -> dict:
        import urllib.request
        import urllib.parse
        url = self._base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            self._raise(e.code, e.read())

    def post(self, path: str, body: dict = None) -> dict:
        import urllib.request
        data = json.dumps(body or {}).encode()
        req  = urllib.request.Request(self._base + path, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            self._raise(e.code, e.read())

    def delete(self, path: str) -> None:
        import urllib.request
        req = urllib.request.Request(self._base + path, headers=self._headers(), method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=self._timeout)
        except urllib.error.HTTPError as e:
            if e.code != 204:
                self._raise(e.code, e.read())

class _AsyncTransport:
    def __init__(self, base_url: str, api_key: str, timeout: float):
        self._base    = base_url.rstrip("/")
        self._key     = api_key
        self._timeout = timeout
        self._client  = None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            h["X-TrueNorth-Key"] = self._key
        return h

    async def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    headers = self._headers(),
                    timeout = self._timeout,
                )
            except ImportError:
                raise ImportError("httpx required for async SDK: pip install httpx")
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def _raise(self, resp) -> None:
        try:
            d = resp.json()
            raise TrueNorthError(resp.status_code, d.get("error","unknown"), d.get("message",""))
        except Exception:
            raise TrueNorthError(resp.status_code, "http_error", resp.text[:200])

    async def get(self, path: str, params: dict = None) -> dict:
        client = await self._get_client()
        resp   = await client.get(self._base + path, params=params)
        if resp.status_code >= 400:
            await self._raise(resp)
        return resp.json()

    async def post(self, path: str, body: dict = None) -> dict:
        client = await self._get_client()
        resp   = await client.post(self._base + path, json=body or {})
        if resp.status_code >= 400:
            await self._raise(resp)
        return resp.json()

    async def delete(self, path: str) -> None:
        client = await self._get_client()
        await client.delete(self._base + path)

class _SessionsResource:
    def __init__(self, transport): self._t = transport

    def create(
        self,
        goal_id:    str,
        user_id:    Optional[str]            = None,
        session_id: Optional[str]            = None,
        budget_usd: Optional[float]          = None,
        seed_fields: Optional[Dict[str, Any]] = None,
        language:   Optional[str]            = None,
    ) -> Session:
        resp = self._t.post("/sessions", {
            "goal_id": goal_id, "user_id": user_id,
            "session_id": session_id, "budget_usd": budget_usd,
            "seed_fields": seed_fields, "language": language,
        })
        return Session.from_dict(resp)

    def message(self, session_id: str, text: str) -> MessageResult:
        resp = self._t.post(f"/sessions/{session_id}/message", {"text": text})
        return MessageResult.from_dict(resp)

    def get(self, session_id: str) -> Session:
        return Session.from_dict(self._t.get(f"/sessions/{session_id}"))

    def output(self, session_id: str) -> Output:
        return Output.from_dict(self._t.get(f"/sessions/{session_id}/output"))

    def force_output(self, session_id: str) -> Output:
        return Output.from_dict(self._t.post(f"/sessions/{session_id}/force-output"))

    def end(self, session_id: str) -> None:
        self._t.delete(f"/sessions/{session_id}")

class _AsyncSessionsResource:
    def __init__(self, transport): self._t = transport

    async def create(
        self,
        goal_id:    str,
        user_id:    Optional[str]            = None,
        session_id: Optional[str]            = None,
        budget_usd: Optional[float]          = None,
        seed_fields: Optional[Dict[str, Any]] = None,
        language:   Optional[str]            = None,
    ) -> Session:
        resp = await self._t.post("/sessions", {
            "goal_id": goal_id, "user_id": user_id,
            "session_id": session_id, "budget_usd": budget_usd,
            "seed_fields": seed_fields, "language": language,
        })
        return Session.from_dict(resp)

    async def message(self, session_id: str, text: str) -> MessageResult:
        resp = await self._t.post(f"/sessions/{session_id}/message", {"text": text})
        return MessageResult.from_dict(resp)

    async def get(self, session_id: str) -> Session:
        return Session.from_dict(await self._t.get(f"/sessions/{session_id}"))

    async def output(self, session_id: str) -> Output:
        return Output.from_dict(await self._t.get(f"/sessions/{session_id}/output"))

    async def force_output(self, session_id: str) -> Output:
        return Output.from_dict(await self._t.post(f"/sessions/{session_id}/force-output"))

    async def end(self, session_id: str) -> None:
        await self._t.delete(f"/sessions/{session_id}")

class _GoalsResource:
    def __init__(self, transport): self._t = transport

    def list(self, q: str = "", sector: str = None, limit: int = 20) -> List[dict]:
        return self._t.get("/goals", {"q": q, "sector": sector, "limit": limit})

    def get(self, name: str, version: str = "latest") -> dict:
        return self._t.get(f"/goals/{name}", {"version": version})

    def install(self, name: str, version: str = "latest") -> dict:
        return self._t.post(f"/goals/{name}/install", {"version": version})

class _AsyncGoalsResource:
    def __init__(self, transport): self._t = transport

    async def list(self, q: str = "", sector: str = None, limit: int = 20) -> List[dict]:
        return await self._t.get("/goals", {"q": q, "sector": sector, "limit": limit})

    async def get(self, name: str, version: str = "latest") -> dict:
        return await self._t.get(f"/goals/{name}", {"version": version})

    async def install(self, name: str, version: str = "latest") -> dict:
        return await self._t.post(f"/goals/{name}/install", {"version": version})

class _AnalyticsResource:
    def __init__(self, transport): self._t = transport

    def cost(self, goal: str, period: int = 7) -> dict:
        return self._t.get("/analytics/cost", {"goal": goal, "period": period})

    def health(self, goal: str, window: int = 24) -> dict:
        return self._t.get("/analytics/health", {"goal": goal, "window": window})

    def cost_trend(self, goal: str, period: int = 30, granularity: str = "day") -> List[dict]:
        return self._t.get("/analytics/cost/trend",
                           {"goal": goal, "period": period, "granularity": granularity})

class _AsyncAnalyticsResource:
    def __init__(self, transport): self._t = transport

    async def cost(self, goal: str, period: int = 7) -> dict:
        return await self._t.get("/analytics/cost", {"goal": goal, "period": period})

    async def health(self, goal: str, window: int = 24) -> dict:
        return await self._t.get("/analytics/health", {"goal": goal, "window": window})

    async def cost_trend(self, goal: str, period: int = 30, granularity: str = "day") -> List[dict]:
        return await self._t.get("/analytics/cost/trend",
                                 {"goal": goal, "period": period, "granularity": granularity})

class TrueNorth:
    """
    Synchronous TrueNorth SDK client.

    Usage:
        from truenorth_sdk import TrueNorth

        tn      = TrueNorth(api_key="tn_live_...")
        session = tn.sessions.create("fitness-coach")

        while not session.is_complete:
            text    = input("> ")
            result  = tn.sessions.message(session.id, text)
            print(result.text)

        output = tn.sessions.output(session.id)
        print(output.content)
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: str           = "http://localhost:8000",
        timeout:  float         = 60.0,
    ):
        key = api_key or os.environ.get("TRUENORTH_API_KEY", "")
        t   = _SyncTransport(base_url, key, timeout)
        self.sessions  = _SessionsResource(t)
        self.goals     = _GoalsResource(t)
        self.analytics = _AnalyticsResource(t)
        self._transport = t

    def health(self) -> dict:
        return self._transport.get("/health")

class AsyncTrueNorth:
    """
    Async TrueNorth SDK client.

    Usage:
        from truenorth_sdk import AsyncTrueNorth

        async with AsyncTrueNorth(api_key="tn_live_...") as tn:
            session = await tn.sessions.create("fitness-coach")
            result  = await tn.sessions.message(session.id, "I am 28")
            output  = await tn.sessions.output(session.id)
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: str           = "http://localhost:8000",
        timeout:  float         = 60.0,
    ):
        key = api_key or os.environ.get("TRUENORTH_API_KEY", "")
        t   = _AsyncTransport(base_url, key, timeout)
        self.sessions  = _AsyncSessionsResource(t)
        self.goals     = _AsyncGoalsResource(t)
        self.analytics = _AsyncAnalyticsResource(t)
        self._transport = t

    async def health(self) -> dict:
        return await self._transport.get("/health")

    async def __aenter__(self) -> "AsyncTrueNorth":
        return self

    async def __aexit__(self, *args) -> None:
        await self._transport.close()

def run_session(
    goal_id:   str,
    messages:  List[str],
    api_key:   Optional[str] = None,
    base_url:  str           = "http://localhost:8000",
    budget_usd: Optional[float] = None,
) -> Output:
    """
    Run a full conversation session with a fixed list of messages.
    Useful for testing and batch processing.

    Returns the final output when the session completes or
    all messages are exhausted (force-output).
    """
    tn      = TrueNorth(api_key=api_key, base_url=base_url)
    session = tn.sessions.create(goal_id, budget_usd=budget_usd)
    for msg in messages:
        result = tn.sessions.message(session.id, msg)
        if result.is_complete:
            return tn.sessions.output(session.id)
    return tn.sessions.force_output(session.id)

async def arun_session(
    goal_id:   str,
    messages:  List[str],
    api_key:   Optional[str] = None,
    base_url:  str           = "http://localhost:8000",
    budget_usd: Optional[float] = None,
) -> Output:
    """Async version of run_session()."""
    async with AsyncTrueNorth(api_key=api_key, base_url=base_url) as tn:
        session = await tn.sessions.create(goal_id, budget_usd=budget_usd)
        for msg in messages:
            result = await tn.sessions.message(session.id, msg)
            if result.is_complete:
                return await tn.sessions.output(session.id)
        return await tn.sessions.force_output(session.id)
