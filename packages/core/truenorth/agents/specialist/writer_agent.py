"""
WriterAgent — generates structured final output from collected fields.

Wraps TrueNorth's OutputGenerator with agent-protocol messaging.
Called by the Orchestrator as the last step of a multi-agent workflow,
after all fields are collected and validated.

Domain-agnostic: same WriterAgent produces a medical summary, a legal
intake report, an HR screening summary, or a fitness plan. The
goal_config in the payload carries the output template.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

from truenorth.agents.base import BaseAgent
from truenorth.agents.messages import AgentMessage, AgentResponse, AgentRole, TaskStatus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """
    Generates final structured output from collected field values.
    """

    agent_id     = "writer_agent"
    role         = AgentRole.WRITER
    capabilities = {"write", "generate", "output", "report", "summarise", "summarize"}

    async def handle(self, message: AgentMessage) -> AgentResponse:
        payload          = message.payload
        collected_fields = payload.get("collected_fields", {})
        goal_config      = payload.get("goal_config", {})
        session_id       = payload.get("session_id", message.session_id)

        if not collected_fields:
            return self.fail(message, "No collected_fields provided for output generation")

        if self._router is not None:
            result = await self._llm_generate(
                collected_fields, goal_config, session_id
            )
        else:
            result = self._template_generate(collected_fields, goal_config)

        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.COMPLETED,
            result     = result,
            confidence = 0.85 if self._router else 0.60,
            metadata   = {
                "format":    goal_config.get("output", {}).get("format", "text"),
                "field_count": len(collected_fields),
            },
        )

    async def _llm_generate(
        self,
        collected_fields: Dict[str, Any],
        goal_config:      Dict[str, Any],
        session_id:       str,
    ) -> Dict[str, Any]:
        from truenorth.output.generator import OutputGenerator
        from truenorth.core.graph_state import GraphState

        gen   = OutputGenerator(router=self._router)
        state = GraphState.__new__(GraphState)
        state.session_id       = session_id
        state.goal_id          = goal_config.get("id", "writer_task")
        state.fields_config    = {f["name"]: f for f in goal_config.get("fields", [])}
        state.collected_fields = collected_fields
        state.field_confidences= {k: 0.85 for k in collected_fields}
        state.turn_history     = []
        state.current_turn     = 1
        state.total_cost_usd   = 0.0
        state.completion_pct   = 100.0

        return await gen.generate(state)

    @staticmethod
    def _template_generate(
        collected_fields: Dict[str, Any],
        goal_config:      Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple template fill without LLM."""
        template = goal_config.get("output", {}).get("template", "")
        fmt      = goal_config.get("output", {}).get("format", "text")
        if template:
            try:
                content = template.format(**collected_fields)
            except KeyError:
                content = str(collected_fields)
        else:
            lines = [f"{k}: {v}" for k, v in collected_fields.items()]
            content = "\n".join(lines)
        return {"content": content, "format": fmt, "fields": collected_fields}