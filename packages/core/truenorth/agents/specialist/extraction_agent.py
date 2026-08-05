"""
ExtractionAgent — specialist for extracting structured fields from text.

Wraps TrueNorth's FieldExtractor with agent-protocol messaging.
Used by the Orchestrator when a field extraction subtask is needed
in a multi-agent workflow.

Works for any domain: medical field extraction, legal entity extraction,
HR candidate data extraction, financial parameter extraction, etc.
The field config in the payload determines what gets extracted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from truenorth.llm.base import Message

from truenorth.agents.base import BaseAgent
from truenorth.agents.messages import AgentMessage, AgentResponse, AgentRole, TaskStatus

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)

class ExtractionAgent(BaseAgent):
    """
    Extracts structured field values from raw text using the LLM.
    """

    agent_id     = "extraction_agent"
    role         = AgentRole.EXTRACTOR
    capabilities = {"extract", "parse", "field", "entity", "structured"}

    def __init__(self, router: Optional["LLMRouter"] = None, config: Optional[dict] = None):
        super().__init__(router=router, config=config)

    async def handle(self, message: AgentMessage) -> AgentResponse:
        payload       = message.payload
        text          = payload.get("text", "")
        fields_config = payload.get("fields_config", {})
        target_fields = payload.get("target_fields")

        if not text:
            return self.fail(message, "No text provided for extraction")
        if self._router is not None:
            result = await self._llm_extract(text, fields_config, target_fields)
        else:
            result = self._rule_extract(text, fields_config, target_fields)

        if not result:
            return AgentResponse(
                message_id = message.message_id,
                agent_id   = self.agent_id,
                status     = TaskStatus.COMPLETED,
                result     = {},
                confidence = 0.40,
                metadata   = {"extracted_count": 0},
            )

        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.COMPLETED,
            result     = result,
            confidence = self._score_confidence(result, fields_config),
            metadata   = {"extracted_count": len(result)},
        )

    async def _llm_extract(
        self,
        text:          str,
        fields_config: Dict[str, Any],
        target_fields: Optional[list],
    ) -> Dict[str, Any]:
        """Extract using TrueNorth's FieldExtractor."""
        from truenorth.core.field_extractor import FieldExtractor
        from truenorth.core.graph_state    import GraphState

        extractor = FieldExtractor(router=self._router)

        state = GraphState.__new__(GraphState)
        state.session_id       = "agent_extract"
        state.fields_config    = fields_config
        state.collected_fields = {}
        state.turn_history     = []
        state.current_turn     = 0
        state.skipped_fields   = set()

        result = await extractor.extract(
            message = Message(role="user", content=text),
            state   = state,
        )

        extracted: Dict[str, Any] = {}
        for ef in result.fields:
            if target_fields is None or ef.name in target_fields:
                extracted[ef.name] = ef.value
        return extracted

    @staticmethod
    def _rule_extract(
        text:          str,
        fields_config: Dict[str, Any],
        target_fields: Optional[list],
    ) -> Dict[str, Any]:
        """
        Simple regex fallback when no LLM is available.
        Handles numbers, dates, and short text values.
        """
        import re
        extracted: Dict[str, Any] = {}
        text_lower = text.lower()

        for fname, cfg in fields_config.items():
            if target_fields and fname not in target_fields:
                continue
            ftype = cfg.get("type", "text")
            if ftype in ("integer", "int", "number", "float"):
                nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
                if nums:
                    try:
                        val = int(nums[0]) if ftype in ("integer", "int") else float(nums[0])
                        extracted[fname] = val
                    except ValueError:
                        pass
            elif ftype == "text":
                for av in cfg.get("allowed_values", []):
                    if str(av).lower() in text_lower:
                        extracted[fname] = av
                        break
        return extracted

    @staticmethod
    def _score_confidence(
        result: Dict[str, Any],
        fields_config: Dict[str, Any],
    ) -> float:
        if not fields_config:
            return 0.70
        required = {k for k, v in fields_config.items() if v.get("required", True)}
        if not required:
            return 0.80
        hit = len(required & set(result.keys()))
        return round(0.50 + 0.50 * (hit / len(required)), 2)
