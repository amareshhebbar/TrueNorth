"""
ResearchAgent — calls MCP tools to gather supplementary information.

Used when the agent needs to look something up to answer or validate a claim:
  - Web search for current information
  - Calculator for derived values (BMI, TDEE, compound interest, debt ratios)
  - Datetime for scheduling, age verification, deadline calculations
  - Custom MCP server calls

Domain-agnostic: a medical research agent uses the same code to look up
drug interactions as a legal agent uses to look up case law precedents.
What differs is which MCP tools are registered.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from truenorth.agents.base import BaseAgent
from truenorth.agents.messages import AgentMessage, AgentResponse, AgentRole, TaskStatus

if TYPE_CHECKING:
    from truenorth.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Calls registered MCP tools to gather supplementary information.

    Payload keys expected:
      tool_name  : str  — which MCP tool to call
      arguments  : dict — tool arguments
      question   : str  — what we're trying to find out (for logging)

    Result shape:
      The raw tool result (dict or str from MCP tool)
    """

    agent_id     = "research_agent"
    role         = AgentRole.RESEARCHER
    capabilities = {"search", "lookup", "calculate", "research", "find", "compute"}

    def __init__(
        self,
        registry: Optional["MCPRegistry"] = None,
        router=None,
        config: Optional[dict] = None,
    ):
        super().__init__(router=router, config=config)
        self._registry = registry

    async def handle(self, message: AgentMessage) -> AgentResponse:
        payload   = message.payload
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments", {})
        question  = payload.get("question", task := message.task)

        if not tool_name:
            tool_name = self._infer_tool(message.task, payload)

        if tool_name is None:
            return self.fail(message, "No tool_name specified and cannot be inferred")

        if self._registry is None:
            return self.fail(message, "No MCP registry configured for ResearchAgent")

        logger.info(
            "research_agent: calling tool=%s question=%r",
            tool_name, (question or message.task)[:60],
        )

        result = await self._registry.call_tool(
            tool_name  = tool_name,
            arguments  = arguments,
            session_id = message.session_id,
            turn       = message.turn,
        )

        if not result.is_success:
            return AgentResponse(
                message_id = message.message_id,
                agent_id   = self.agent_id,
                status     = TaskStatus.FAILED,
                result     = None,
                error      = result.error or f"Tool {tool_name!r} failed",
            )

        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.COMPLETED,
            result     = result.content,
            confidence = 0.90,
            metadata   = {
                "tool":       tool_name,
                "latency_ms": result.latency_ms,
            },
        )

    @staticmethod
    def _infer_tool(task: str, payload: dict) -> Optional[str]:
        """Infer tool_name from task description."""
        task_lower = task.lower()
        if any(k in task_lower for k in ("calculat", "bmi", "formula", "compute", "math")):
            return "calculator"
        if any(k in task_lower for k in ("date", "time", "when", "timezone")):
            return "datetime_tool"
        if any(k in task_lower for k in ("search", "find", "lookup", "web")):
            return "web_search"
        return None