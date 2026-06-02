"""
Google Agent-to-Agent (A2A) Protocol bridge for TrueNorth.

A2A is Google's open standard (https://google.github.io/A2A) for agents
to communicate with each other across frameworks. TrueNorth implements it
so that:
  1. A TrueNorth agent can receive tasks from any A2A-compliant agent
     (Google ADK, AutoGen, CrewAI with A2A adapters, etc.)
  2. A TrueNorth agent can dispatch tasks TO external A2A agents
     and receive structured results back.

Protocol overview:
  - Agents expose an HTTP endpoint (the "A2A endpoint")
  - Messages use JSON-RPC 2.0 with standardised method names
  - Core methods:
      tasks/send        — send a task to an agent
      tasks/get         — poll task status
      tasks/cancel      — cancel a running task
      tasks/sendSubscribe — streaming (SSE) task updates
  - Task states: submitted → working → completed | failed | cancelled
  - Each task has an AgentCard (capabilities, skills, auth) at /.well-known/agent.json

TrueNorth implementation:
  - A2AClient     — sends tasks to external A2A agents, maps results to AgentResponse
  - A2AServer     — exposes TrueNorth agents as an A2A endpoint
  - A2ATaskBridge — wraps TrueNorth AgentMessage ↔ A2A Task format

Sector-agnostic: a medical TrueNorth agent can receive tasks from a
Google ADK scheduling agent, a legal agent can delegate research to a
specialist A2A agent, HR agent can pipe to a background-check service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  A2A Task states (mirrors the spec)
# ─────────────────────────────────────────────────────────────────────────────

class A2ATaskState(str, Enum):
    SUBMITTED  = "submitted"
    WORKING    = "working"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    INPUT_REQUIRED = "input-required"   


# ─────────────────────────────────────────────────────────────────────────────
#  A2A data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class A2APart:
    """One part of a multi-part message (text, data, file)."""
    type:     str          # "text" | "data" | "file"
    content:  Any          # str | dict | bytes
    mime_type: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.type == "text":
            d["text"] = str(self.content)
        elif self.type == "data":
            d["data"] = self.content if isinstance(self.content, dict) else {"value": self.content}
        return d

    @classmethod
    def text(cls, content: str) -> "A2APart":
        return cls(type="text", content=content)

    @classmethod
    def data(cls, content: dict) -> "A2APart":
        return cls(type="data", content=content)


@dataclass
class A2AMessage:
    """A message in an A2A conversation."""
    role:   str              # "user" | "agent"
    parts:  List[A2APart]

    def to_dict(self) -> dict:
        return {"role": self.role, "parts": [p.to_dict() for p in self.parts]}

    @classmethod
    def from_dict(cls, d: dict) -> "A2AMessage":
        parts = [
            A2APart(
                type    = p.get("type", "text"),
                content = p.get("text") or p.get("data") or p.get("bytes", ""),
            )
            for p in d.get("parts", [])
        ]
        return cls(role=d.get("role", "user"), parts=parts)

    def text_content(self) -> str:
        """Concatenate all text parts."""
        return " ".join(
            str(p.content) for p in self.parts if p.type == "text"
        )


@dataclass
class A2ATask:
    """An A2A task — the unit of work between two agents."""
    id:           str
    state:        A2ATaskState
    messages:     List[A2AMessage] = field(default_factory=list)
    artifacts:    List[dict]       = field(default_factory=list)  # outputs
    metadata:     Dict[str, Any]   = field(default_factory=dict)
    session_id:   Optional[str]    = None
    created_at:   float            = field(default_factory=time.time)
    updated_at:   float            = field(default_factory=time.time)
    error:        Optional[str]    = None

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "state":      self.state.value,
            "messages":   [m.to_dict() for m in self.messages],
            "artifacts":  self.artifacts,
            "metadata":   self.metadata,
            "sessionId":  self.session_id,
            "error":      self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "A2ATask":
        return cls(
            id        = d.get("id", str(uuid.uuid4())[:12]),
            state     = A2ATaskState(d.get("state", "submitted")),
            messages  = [A2AMessage.from_dict(m) for m in d.get("messages", [])],
            artifacts = d.get("artifacts", []),
            metadata  = d.get("metadata", {}),
            session_id= d.get("sessionId"),
        )


@dataclass
class AgentCard:
    """
    A2A AgentCard — machine-readable capability descriptor served at
    /.well-known/agent.json by any A2A-compliant agent.
    """
    name:         str
    description:  str
    url:          str           # the agent's A2A endpoint
    version:      str = "1.0.0"
    skills:       List[dict] = field(default_factory=list)
    capabilities: dict         = field(default_factory=dict)
    auth:         Optional[dict] = None    # auth requirements

    def to_dict(self) -> dict:
        d = {
            "name":         self.name,
            "description":  self.description,
            "url":          self.url,
            "version":      self.version,
            "skills":       self.skills,
            "capabilities": self.capabilities,
        }
        if self.auth:
            d["authentication"] = self.auth
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentCard":
        return cls(
            name         = d.get("name", "unknown"),
            description  = d.get("description", ""),
            url          = d.get("url", ""),
            version      = d.get("version", "1.0.0"),
            skills       = d.get("skills", []),
            capabilities = d.get("capabilities", {}),
            auth         = d.get("authentication"),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  A2ATaskBridge — converts TrueNorth ↔ A2A formats
# ─────────────────────────────────────────────────────────────────────────────

class A2ATaskBridge:
    """
    Bidirectional translation between TrueNorth's AgentMessage/AgentResponse
    and A2A's Task/Message format.
    """

    @staticmethod
    def agent_message_to_a2a(message) -> A2ATask:
        task_id = message.message_id
        text    = message.task
        parts   = [A2APart.text(text)]
        if message.payload:
            parts.append(A2APart.data(message.payload))

        return A2ATask(
            id         = task_id,
            state      = A2ATaskState.SUBMITTED,
            messages   = [A2AMessage(role="user", parts=parts)],
            session_id = message.session_id,
            metadata   = {
                "sender":   message.sender,
                "priority": message.priority.value,
                "turn":     message.turn,
            },
        )

    @staticmethod
    def a2a_task_to_agent_response(task: A2ATask) -> "Any":
        """Convert a completed A2A Task to a TrueNorth AgentResponse."""
        from truenorth.agents.messages import AgentResponse, TaskStatus

        status_map = {
            A2ATaskState.COMPLETED:  TaskStatus.COMPLETED,
            A2ATaskState.FAILED:     TaskStatus.FAILED,
            A2ATaskState.CANCELLED:  TaskStatus.CANCELLED,
            A2ATaskState.WORKING:    TaskStatus.RUNNING,
            A2ATaskState.SUBMITTED:  TaskStatus.PENDING,
        }
        status = status_map.get(task.state, TaskStatus.FAILED)

        result: Any = None
        for msg in reversed(task.messages):
            if msg.role == "agent":
                result = msg.text_content() or (
                    msg.parts[0].content if msg.parts else None
                )
                break
        if result is None and task.artifacts:
            result = task.artifacts[-1]

        return AgentResponse(
            message_id = task.id,
            agent_id   = "a2a_external",
            status     = status,
            result     = result,
            confidence = 0.85,   
            error      = task.error,
            metadata   = task.metadata,
        )

    @staticmethod
    def text_to_a2a_task(task_text: str, session_id: str = "") -> A2ATask:
        """Create a simple A2A task from a text instruction."""
        return A2ATask(
            id         = str(uuid.uuid4())[:12],
            state      = A2ATaskState.SUBMITTED,
            messages   = [A2AMessage(role="user", parts=[A2APart.text(task_text)])],
            session_id = session_id or str(uuid.uuid4())[:8],
        )


# ─────────────────────────────────────────────────────────────────────────────
#  A2AClient — calls external A2A agents
# ─────────────────────────────────────────────────────────────────────────────

class A2AClient:
    """
    Client for calling any external A2A-compliant agent.

    Supports:
      - tasks/send         (submit + poll until done)
      - tasks/get          (poll task status)
      - tasks/cancel       (cancel a running task)
      - tasks/sendSubscribe (SSE streaming updates)
      - /.well-known/agent.json (fetch agent capabilities)
    """

    def __init__(
        self,
        endpoint:      str,
        timeout_s:     float          = 30.0,
        poll_interval: float          = 0.5,
        max_polls:     int            = 60,
        headers:       Optional[dict] = None,
        api_key:       Optional[str]  = None,
    ):
        self._endpoint      = endpoint.rstrip("/")
        self._timeout       = timeout_s
        self._poll_interval = poll_interval
        self._max_polls     = max_polls
        self._headers       = headers or {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client        = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_agent_card(self) -> Optional[AgentCard]:
        """Fetch the agent's capability descriptor."""
        client = self._get_client()
        try:
            resp = await client.get(
                f"{self._endpoint}/.well-known/agent.json",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return AgentCard.from_dict(resp.json())
        except Exception as e:
            logger.warning("a2a: failed to fetch agent card from %s: %s", self._endpoint, e)
            return None

    async def send_task(
        self,
        task_text:  str,
        session_id: str = "",
        payload:    Optional[dict] = None,
        stream:     bool = False,
    ) -> A2ATask:
        """
        Send a task to the external agent and wait for completion.
        Returns the completed A2ATask.
        """
        task = A2ATaskBridge.text_to_a2a_task(task_text, session_id)
        if payload:
            task.messages[0].parts.append(A2APart.data(payload))

        if stream:
            return await self._send_subscribe(task)
        return await self._send_and_poll(task)

    async def send_agent_message(self, message) -> "Any":
        """
        Send a TrueNorth AgentMessage to an external A2A agent.
        Returns a TrueNorth AgentResponse.
        """
        a2a_task = A2ATaskBridge.agent_message_to_a2a(message)
        completed = await self._send_and_poll(a2a_task)
        return A2ATaskBridge.a2a_task_to_agent_response(completed)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task. Returns True if cancelled successfully."""
        client = self._get_client()
        try:
            resp = await client.post(
                f"{self._endpoint}",
                json=self._jsonrpc("tasks/cancel", {"id": task_id}),
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning("a2a: cancel_task failed: %s", e)
            return False

    async def get_task(self, task_id: str) -> Optional[A2ATask]:
        """Poll the current state of a task."""
        client = self._get_client()
        try:
            resp = await client.post(
                f"{self._endpoint}",
                json=self._jsonrpc("tasks/get", {"id": task_id}),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result", {})
            return A2ATask.from_dict(result)
        except Exception as e:
            logger.warning("a2a: get_task failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    async def _send_and_poll(self, task: A2ATask) -> A2ATask:
        """Submit a task then poll until it reaches a terminal state."""
        client = self._get_client()
        try:
            resp = await client.post(
                f"{self._endpoint}",
                json=self._jsonrpc("tasks/send", {"task": task.to_dict()}),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data    = resp.json()
            current = A2ATask.from_dict(data.get("result", {}))
        except Exception as e:
            logger.error("a2a: tasks/send failed for endpoint=%s: %s", self._endpoint, e)
            return A2ATask(
                id=task.id, state=A2ATaskState.FAILED,
                messages=task.messages, error=str(e),
            )

        # Poll until terminal
        terminal = {A2ATaskState.COMPLETED, A2ATaskState.FAILED, A2ATaskState.CANCELLED}
        for _ in range(self._max_polls):
            if current.state in terminal:
                break
            await asyncio.sleep(self._poll_interval)
            polled = await self.get_task(current.id)
            if polled:
                current = polled

        logger.info("a2a: task=%s final_state=%s", current.id, current.state.value)
        return current

    async def _send_subscribe(self, task: A2ATask) -> A2ATask:
        """Submit with SSE streaming. Accumulates chunks until done."""
        client = self._get_client()
        messages: List[A2AMessage] = list(task.messages)
        try:
            async with client.stream(
                "POST", f"{self._endpoint}",
                json=self._jsonrpc("tasks/sendSubscribe", {"task": task.to_dict()}),
                timeout=self._timeout + 60,
            ) as resp:
                resp.raise_for_status()
                final_state = A2ATaskState.WORKING
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        ev_data = json.loads(line[5:].strip())
                        result  = ev_data.get("result", {})
                        state   = A2ATaskState(result.get("state", "working"))
                        # Collect any new messages/artifacts
                        for m in result.get("messages", []):
                            messages.append(A2AMessage.from_dict(m))
                        final_state = state
                        if state in (A2ATaskState.COMPLETED, A2ATaskState.FAILED):
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            logger.error("a2a: streaming failed: %s", e)
            final_state = A2ATaskState.FAILED

        return A2ATask(
            id       = task.id,
            state    = final_state,
            messages = messages,
            session_id = task.session_id,
        )

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    headers = {"Content-Type": "application/json", **self._headers},
                    timeout = httpx.Timeout(
                        connect=5.0, read=self._timeout, write=10.0, pool=5.0
                    ),
                )
            except ImportError as e:
                raise RuntimeError("httpx not installed. Run: pip install httpx") from e
        return self._client

    @staticmethod
    def _jsonrpc(method: str, params: dict) -> dict:
        return {
            "jsonrpc": "2.0",
            "id":      str(uuid.uuid4())[:8],
            "method":  method,
            "params":  params,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  A2AServer — exposes TrueNorth agents as an A2A endpoint
# ─────────────────────────────────────────────────────────────────────────────

class A2AServer:
    """
    Wraps a TrueNorth BaseAgent and exposes it as an A2A-compliant HTTP endpoint.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        card:  Optional[AgentCard] = None,
    ):
        self._agent  = agent
        self._card   = card or self._default_card(agent)
        self._tasks: Dict[str, A2ATask] = {}

    # ------------------------------------------------------------------
    # Core handlers (framework-agnostic)
    # ------------------------------------------------------------------

    async def handle_send(self, task_dict: dict) -> dict:
        """Handle tasks/send — run the agent and return the completed task."""
        task = A2ATask.from_dict(task_dict)
        task.state = A2ATaskState.WORKING
        self._tasks[task.id] = task

        # Convert to TrueNorth AgentMessage
        from truenorth.agents.messages import AgentMessage
        text    = " ".join(m.text_content() for m in task.messages if m.role == "user")
        payload = {}
        for msg in task.messages:
            for part in msg.parts:
                if part.type == "data" and isinstance(part.content, dict):
                    payload.update(part.content)

        tn_msg = AgentMessage.create(
            sender     = "a2a_client",
            recipient  = self._agent.agent_id,
            task       = text,
            payload    = payload,
            session_id = task.session_id or "",
        )

        response = await self._agent.execute(tn_msg)
        result_text  = response.result_text or response.error or ""
        result_parts = [A2APart.text(result_text)]
        if isinstance(response.result, dict):
            result_parts.append(A2APart.data(response.result))

        task.messages.append(A2AMessage(role="agent", parts=result_parts))
        task.state    = (A2ATaskState.COMPLETED if response.is_success
                         else A2ATaskState.FAILED)
        task.error    = response.error
        task.artifacts = (
            [{"type": "data", "parts": [A2APart.data(response.result).to_dict()]}]
            if isinstance(response.result, dict) else []
        )
        task.updated_at = time.time()
        self._tasks[task.id] = task

        return {"jsonrpc": "2.0", "id": "1", "result": task.to_dict()}

    async def handle_get(self, task_id: str) -> dict:
        """Handle tasks/get — return current task state."""
        task = self._tasks.get(task_id)
        if task is None:
            return {
                "jsonrpc": "2.0", "id": "1",
                "error": {"code": -32001, "message": f"Task {task_id!r} not found"},
            }
        return {"jsonrpc": "2.0", "id": "1", "result": task.to_dict()}

    async def handle_cancel(self, task_id: str) -> dict:
        """Handle tasks/cancel."""
        task = self._tasks.get(task_id)
        if task and task.state == A2ATaskState.WORKING:
            task.state = A2ATaskState.CANCELLED
            task.updated_at = time.time()
        return {"jsonrpc": "2.0", "id": "1", "result": {"id": task_id, "state": "cancelled"}}

    def agent_card_dict(self) -> dict:
        """Return the AgentCard as a dict (for /.well-known/agent.json)."""
        return self._card.to_dict()

    # ------------------------------------------------------------------
    # FastAPI integration
    # ------------------------------------------------------------------

    def fastapi_router(self):
        """
        Return a FastAPI APIRouter that exposes this agent as an A2A endpoint.
        Attach it with: app.include_router(server.fastapi_router())
        """
        try:
            from fastapi import APIRouter, Request
            from fastapi.responses import JSONResponse
        except ImportError:
            raise RuntimeError("fastapi not installed. Run: pip install fastapi")

        router = APIRouter()

        @router.get("/.well-known/agent.json")
        async def agent_card():
            return JSONResponse(self.agent_card_dict())

        @router.post("/")
        async def a2a_endpoint(request: Request):
            body   = await request.json()
            method = body.get("method", "")
            params = body.get("params", {})

            if method == "tasks/send":
                result = await self.handle_send(params.get("task", {}))
            elif method == "tasks/get":
                result = await self.handle_get(params.get("id", ""))
            elif method == "tasks/cancel":
                result = await self.handle_cancel(params.get("id", ""))
            else:
                result = {
                    "jsonrpc": "2.0", "id": body.get("id", "1"),
                    "error": {"code": -32601, "message": f"Method {method!r} not found"},
                }
            return JSONResponse(result)

        return router

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_card(agent: "BaseAgent") -> AgentCard:
        return AgentCard(
            name        = agent.agent_id,
            description = f"TrueNorth {agent.role.value} agent",
            url         = "http://localhost:8000",
            skills      = [
                {"id": cap, "name": cap, "description": f"Handles {cap} tasks"}
                for cap in (agent.capabilities or {"general"})
            ],
            capabilities = {"streaming": False, "pushNotifications": False},
        )