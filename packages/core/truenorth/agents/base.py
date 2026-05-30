"""
BaseAgent — abstract base class for all TrueNorth agents.

Every specialist agent inherits from this. The contract:
  1. Implement handle(message) → AgentResponse
  2. Set agent_id, role, capabilities
  3. Optionally override can_handle(message) for routing decisions

Built-in behaviours (all agents get these for free):
  - Timeout enforcement per message
  - Retry tracking
  - Health / readiness reporting
  - Metrics: call count, error rate, avg latency
  - Structured logging with agent_id + session_id

Domain-agnostic: same base class for a medical extraction agent,
a legal research agent, an HR validation agent, or a fitness planner.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from truenorth.agents.messages import (
    AgentMessage, AgentResponse, AgentRole,
    MessageType, TaskStatus, Priority,
)

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    agent_id:       str
    call_count:     int   = 0
    success_count:  int   = 0
    error_count:    int   = 0
    timeout_count:  int   = 0
    total_latency_ms: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.call_count, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.success_count, 1)

    def record(self, status: TaskStatus, latency_ms: int) -> None:
        self.call_count     += 1
        self.total_latency_ms += latency_ms
        if status == TaskStatus.COMPLETED:
            self.success_count += 1
        elif status == TaskStatus.FAILED:
            self.error_count += 1
        elif status == TaskStatus.CANCELLED:
            self.timeout_count += 1

    def to_dict(self) -> dict:
        return {
            "agent_id":     self.agent_id,
            "call_count":   self.call_count,
            "success_rate": round(self.success_rate, 3),
            "error_count":  self.error_count,
            "timeout_count":self.timeout_count,
            "avg_latency_ms": round(self.avg_latency_ms),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BaseAgent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all TrueNorth specialist agents.

    Subclass this to create:
      - Domain-specific extraction agents (medical, legal, HR)
      - Tool-calling research agents
      - Output generation agents
      - Validation / quality agents

    Minimal implementation:

        class MyAgent(BaseAgent):
            agent_id     = "my_agent"
            role         = AgentRole.CUSTOM
            capabilities = {"extract", "classify"}

            async def handle(self, message: AgentMessage) -> AgentResponse:
                # do the work
                return AgentResponse(
                    message_id = message.message_id,
                    agent_id   = self.agent_id,
                    status     = TaskStatus.COMPLETED,
                    result     = {"answer": "42"},
                )
    """

    agent_id:     str      = "base_agent"
    role:         AgentRole = AgentRole.CUSTOM
    capabilities: Set[str] = set()   

    # Optional overrides
    max_retries:  int   = 2
    default_timeout_s: float = 30.0

    def __init__(
        self,
        router: Optional["LLMRouter"] = None,
        config: Optional[dict]        = None,
    ):
        self._router  = router
        self._config  = config or {}
        self._metrics = AgentMetrics(agent_id=self.agent_id)
        self._ready   = True

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def handle(self, message: AgentMessage) -> AgentResponse:
        """
        Process an AgentMessage and return an AgentResponse.
        This is the main work method. Keep it focused.
        """

    # ------------------------------------------------------------------
    # Routing interface — subclasses MAY override
    # ------------------------------------------------------------------

    def can_handle(self, message: AgentMessage) -> bool:
        """
        Return True if this agent can handle the given message.
        Default: check if the task matches any capability keyword.
        Override for more sophisticated routing.
        """
        if not self.capabilities:
            return True
        task_lower = message.task.lower()
        return any(cap in task_lower for cap in self.capabilities)

    # ------------------------------------------------------------------
    # Lifecycle — call execute() from Orchestrator (not handle() directly)
    # ------------------------------------------------------------------

    async def execute(self, message: AgentMessage) -> AgentResponse:
        """
        Safely execute a message with timeout, retry, and metric tracking.
        The Orchestrator calls this — never call handle() directly.
        """
        timeout = message.timeout_s or self.default_timeout_s
        t0      = time.perf_counter()

        logger.info(
            "agent[%s]: executing task=%r session=%s turn=%d priority=%s",
            self.agent_id, message.task[:60],
            message.session_id, message.turn, message.priority.value,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.handle(message),
                    timeout=timeout,
                )
                latency = int((time.perf_counter() - t0) * 1000)
                response.latency_ms = latency
                self._metrics.record(response.status, latency)

                logger.info(
                    "agent[%s]: done status=%s confidence=%.2f latency=%dms",
                    self.agent_id, response.status.value,
                    response.confidence, latency,
                )
                return response

            except asyncio.TimeoutError:
                latency = int((time.perf_counter() - t0) * 1000)
                logger.warning(
                    "agent[%s]: timeout after %.1fs (attempt %d/%d)",
                    self.agent_id, timeout, attempt + 1, self.max_retries + 1,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                self._metrics.record(TaskStatus.CANCELLED, latency)
                return self._timeout_response(message, timeout)

            except Exception as e:
                latency = int((time.perf_counter() - t0) * 1000)
                logger.error(
                    "agent[%s]: error on attempt %d: %s: %s",
                    self.agent_id, attempt + 1, type(e).__name__, e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                self._metrics.record(TaskStatus.FAILED, latency)
                return self._error_response(message, e)

        # Should not reach here
        return self._error_response(message, RuntimeError("max retries exceeded"))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._ready

    def health(self) -> dict:
        return {
            "agent_id":   self.agent_id,
            "role":       self.role.value,
            "ready":      self._ready,
            "metrics":    self._metrics.to_dict(),
        }

    # ------------------------------------------------------------------
    # Response factories
    # ------------------------------------------------------------------

    @staticmethod
    def ok(message: AgentMessage, result: Any, confidence: float = 1.0) -> AgentResponse:
        return AgentResponse(
            message_id = message.message_id,
            agent_id   = message.recipient,
            status     = TaskStatus.COMPLETED,
            result     = result,
            confidence = confidence,
        )

    @staticmethod
    def fail(message: AgentMessage, error: str) -> AgentResponse:
        return AgentResponse(
            message_id = message.message_id,
            agent_id   = message.recipient,
            status     = TaskStatus.FAILED,
            result     = None,
            error      = error,
        )

    def _timeout_response(self, message: AgentMessage, timeout: float) -> AgentResponse:
        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.CANCELLED,
            result     = None,
            error      = f"Task timed out after {timeout}s",
        )

    def _error_response(self, message: AgentMessage, exc: Exception) -> AgentResponse:
        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.FAILED,
            result     = None,
            error      = f"{type(exc).__name__}: {exc}",
        )