"""
Responsibilities:
  1. Register and manage a pool of specialist agents
  2. Route incoming tasks to the right agent (capability-based routing)
  3. Coordinate multi-step workflows (sequential + parallel)
  4. Collect and merge results from multiple agents
  5. Escalate failures to the Supervisor
  6. Track the full execution plan for audit

Routing algorithm:
  For each task, the orchestrator:
    1. Checks capability registry — which agents declared they can handle this?
    2. Applies priority filter — if CRITICAL, only high-confidence agents
    3. Picks the agent with the best success_rate × (1 − avg_latency_factor)
    4. Falls back to a default agent if no match
    5. Returns a combined AgentResponse to the caller

Workflow types:
  Sequential: task A must complete before task B starts
  Parallel:   tasks A, B, C run concurrently; results merged

Domain-agnostic: the orchestrator doesn't know or care whether
the domain is medical, legal, HR, or fitness. It only routes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from truenorth.agents.messages import (
    AgentMessage, AgentResponse, AgentRole,
    MessageType, TaskStatus, Priority,
)
from truenorth.agents.base import BaseAgent

if TYPE_CHECKING:
    from truenorth.agents.supervisor import AgentSupervisor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Execution plan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionStep:
    """One step in an orchestration plan."""
    task:        str
    payload:     Dict[str, Any]
    agent_id:    Optional[str] = None  
    priority:    Priority = Priority.NORMAL
    depends_on:  List[str] = field(default_factory=list)  

    def to_dict(self) -> dict:
        return {
            "task":       self.task,
            "agent_id":   self.agent_id,
            "priority":   self.priority.value,
            "depends_on": self.depends_on,
        }


@dataclass
class OrchestrationResult:
    """Combined result of a multi-step orchestration."""
    session_id:    str
    steps_total:   int
    steps_ok:      int
    steps_failed:  int
    results:       List[AgentResponse]
    merged:        Dict[str, Any]      
    latency_ms:    int
    plan:          List[ExecutionStep]

    @property
    def success(self) -> bool:
        return self.steps_failed == 0

    @property
    def partial_success(self) -> bool:
        return self.steps_ok > 0

    def to_dict(self) -> dict:
        return {
            "session_id":   self.session_id,
            "success":      self.success,
            "steps_total":  self.steps_total,
            "steps_ok":     self.steps_ok,
            "steps_failed": self.steps_failed,
            "latency_ms":   self.latency_ms,
            "merged":       self.merged,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  AgentOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Routes tasks to specialist agents and coordinates multi-agent workflows.

    Sector-agnostic: the same orchestrator runs a medical intake,
    a legal case intake, an HR screen, or a fitness plan. What
    changes is which agents are registered.
    """

    def __init__(
        self,
        supervisor:     Optional["AgentSupervisor"] = None,
        config:         Optional[dict]               = None,
        max_parallel:   int   = 5,
        default_timeout: float = 30.0,
    ):
        self._agents:     Dict[str, BaseAgent]  = {}
        self._default:    Optional[BaseAgent]   = None
        self._supervisor  = supervisor
        self._config      = config or {}
        self._max_parallel = max_parallel
        self._default_timeout = default_timeout

        # Execution log — all tasks dispatched this session
        self._log: List[dict] = []

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register(self, agent: BaseAgent, is_default: bool = False) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.agent_id] = agent
        if is_default:
            self._default = agent
        logger.info(
            "orchestrator: registered agent=%s role=%s caps=%s",
            agent.agent_id, agent.role.value, agent.capabilities,
        )

    def set_default(self, agent: BaseAgent) -> None:
        """Set the fallback agent used when no capability match is found."""
        self._default = agent
        if agent.agent_id not in self._agents:
            self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            if self._default and self._default.agent_id == agent_id:
                self._default = None
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[dict]:
        return [a.health() for a in self._agents.values()]

    # ------------------------------------------------------------------
    # Single task dispatch
    # ------------------------------------------------------------------

    async def run_task(
        self,
        task:       str,
        payload:    Dict[str, Any],
        session_id: str         = "",
        turn:       int         = 0,
        priority:   Priority    = Priority.NORMAL,
        agent_id:   Optional[str] = None,   
        timeout_s:  Optional[float] = None,
    ) -> AgentResponse:
        """
        Route one task to the best available agent and return its response.

        Args:
            task:       Natural-language task description (used for routing)
            payload:    Task-specific data dict
            session_id: For logging and audit
            turn:       Conversation turn number
            priority:   Execution priority
            agent_id:   Force a specific agent (bypasses routing)
            timeout_s:  Per-task timeout override
        """
        message = AgentMessage.create(
            sender    = AgentRole.ORCHESTRATOR.value,
            recipient = agent_id or "",
            task      = task,
            payload   = payload,
            session_id = session_id,
            turn       = turn,
            priority   = priority,
            timeout_s  = timeout_s or self._default_timeout,
        )

        agent = self._route(message, agent_id)
        if agent is None:
            logger.warning(
                "orchestrator: no agent available for task=%r session=%s",
                task[:60], session_id,
            )
            return AgentResponse(
                message_id = message.message_id,
                agent_id   = "orchestrator",
                status     = TaskStatus.FAILED,
                result     = None,
                error      = f"No agent available for task: {task!r}",
            )

        message.recipient = agent.agent_id
        response = await agent.execute(message)
        if (
            self._supervisor is not None
            and priority in (Priority.HIGH, Priority.CRITICAL)
            and response.is_success
        ):
            verdict = await self._supervisor.review(response, context=payload)
            if not verdict.approved and verdict.retry:
                logger.info(
                    "orchestrator: supervisor rejected result, retrying agent=%s",
                    agent.agent_id,
                )
                response = await agent.execute(message)

        self._log.append({
            "task":      task[:100],
            "agent_id":  agent.agent_id,
            "status":    response.status.value,
            "latency_ms": response.latency_ms,
            "session_id": session_id,
            "turn":       turn,
            "timestamp":  time.time(),
        })

        return response

    # ------------------------------------------------------------------
    # Parallel task dispatch
    # ------------------------------------------------------------------

    async def run_parallel(
        self,
        tasks:      List[Tuple[str, Dict[str, Any]]],
        session_id: str      = "",
        turn:       int      = 0,
        priority:   Priority = Priority.NORMAL,
    ) -> OrchestrationResult:
        """
        Run multiple independent tasks concurrently.

        Args:
            tasks:      List of (task_description, payload) tuples
            session_id: Session ID for all tasks
            turn:       Conversation turn

        Returns:
            OrchestrationResult with all responses and merged output
        """
        t0       = time.perf_counter()
        sem      = asyncio.Semaphore(self._max_parallel)
        plan     = [
            ExecutionStep(task=t, payload=p, priority=priority)
            for t, p in tasks
        ]

        async def _run_one(step: ExecutionStep) -> AgentResponse:
            async with sem:
                return await self.run_task(
                    task       = step.task,
                    payload    = step.payload,
                    session_id = session_id,
                    turn       = turn,
                    priority   = step.priority,
                )

        responses = await asyncio.gather(
            *[_run_one(step) for step in plan],
            return_exceptions=False,
        )

        ok     = [r for r in responses if r.is_success]
        failed = [r for r in responses if not r.is_success]
        merged = self._merge_results(ok)

        return OrchestrationResult(
            session_id   = session_id,
            steps_total  = len(tasks),
            steps_ok     = len(ok),
            steps_failed = len(failed),
            results      = list(responses),
            merged       = merged,
            latency_ms   = int((time.perf_counter() - t0) * 1000),
            plan         = plan,
        )

    # ------------------------------------------------------------------
    # Sequential workflow
    # ------------------------------------------------------------------

    async def run_sequential(
        self,
        steps:      List[ExecutionStep],
        session_id: str = "",
        turn:       int = 0,
        stop_on_failure: bool = True,
    ) -> OrchestrationResult:
        """
        Run steps in order. Each step can access previous results via payload.
        Stops on first failure if stop_on_failure=True.
        """
        t0        = time.perf_counter()
        responses: List[AgentResponse] = []
        context:   Dict[str, Any]      = {}

        for i, step in enumerate(steps):
            enriched_payload = {**step.payload, "__prior_results": context}
            resp = await self.run_task(
                task       = step.task,
                payload    = enriched_payload,
                session_id = session_id,
                turn       = turn,
                priority   = step.priority,
                agent_id   = step.agent_id,
            )
            responses.append(resp)

            if resp.is_success and isinstance(resp.result, dict):
                context.update(resp.result)

            if not resp.is_success and stop_on_failure:
                logger.warning(
                    "orchestrator: sequential step %d/%d failed — stopping. task=%r",
                    i + 1, len(steps), step.task[:60],
                )
                break

        ok     = [r for r in responses if r.is_success]
        failed = [r for r in responses if not r.is_success]

        return OrchestrationResult(
            session_id   = session_id,
            steps_total  = len(steps),
            steps_ok     = len(ok),
            steps_failed = len(failed),
            results      = responses,
            merged       = context,
            latency_ms   = int((time.perf_counter() - t0) * 1000),
            plan         = steps,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self,
        message:  AgentMessage,
        agent_id: Optional[str] = None,
    ) -> Optional[BaseAgent]:
        """
        Select the best agent for this message.
        Explicit agent_id → capability match → best by success_rate → default.
        """
        if agent_id and agent_id in self._agents:
            return self._agents[agent_id]
        candidates = [
            a for a in self._agents.values()
            if a.is_ready() and a.can_handle(message)
        ]

        if not candidates:
            return self._default

        def _score(agent: BaseAgent) -> float:
            m = agent._metrics
            return m.success_rate - (m.avg_latency_ms / 100_000)

        return max(candidates, key=_score)

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_results(responses: List[AgentResponse]) -> Dict[str, Any]:
        """
        Merge results from multiple agents into one dict.
        Later agents override earlier ones on key conflicts.
        Non-dict results stored under their agent_id.
        """
        merged: Dict[str, Any] = {}
        for resp in responses:
            if isinstance(resp.result, dict):
                merged.update(resp.result)
            elif resp.result is not None:
                merged[resp.agent_id] = resp.result
        return merged

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def execution_log(self) -> List[dict]:
        return list(self._log)

    def stats(self) -> dict:
        total = len(self._log)
        ok    = sum(1 for e in self._log if e["status"] == "completed")
        return {
            "total_tasks":   total,
            "success_rate":  round(ok / max(total, 1), 3),
            "agents":        len(self._agents),
            "log":           self._log[-20:],   
        }