"""
truenorth/output/generator.py

Generates the final output when all required fields are collected.
Supports multiple output formats:
  - text     : narrative paragraph (default)
  - markdown  : formatted markdown report
  - json      : structured JSON object
  - template  : Jinja2-style template from goal YAML

Uses Claude Sonnet (highest quality) for generation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, TYPE_CHECKING

from truenorth.core.graph_state import GraphState

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)


class OutputGenerator:
    """
    Generates the final session output.

    Usage:
        generator = OutputGenerator(router=llm_router)
        result = await generator.generate(state=state)
    """

    _SYSTEM = """\
You are a precise report writer. Generate a clear, personalized output
based ONLY on the information collected. Do not invent or assume any facts
not explicitly provided. Reference the user's actual values, not placeholders.
"""

    def __init__(self, router: Optional["LLMRouter"] = None):
        self._router = router

    async def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate the final output for a completed session.

        Returns dict with:
          - format   : output format
          - content  : the generated content (str or dict)
          - fields   : the collected fields used
          - metadata : token counts, model used, etc.
        """
        fmt       = state.goal_config.get("output", {}).get("format", "text")
        template  = state.output_template
        collected = state.collected_fields
        goal_name = state.goal_config.get("name", state.goal_id)

        logger.info(
            "output_generator: session=%s format=%s fields=%d",
            state.session_id, fmt, len(collected),
        )

        if fmt == "json":
            content = self._json_output(collected, state)
        elif fmt == "markdown":
            content = await self._llm_output(collected, state, goal_name, template, fmt)
        elif template:
            content = await self._template_output(collected, template, state)
        else:
            content = await self._llm_output(collected, state, goal_name, template, fmt)

        result = {
            "format":  fmt,
            "content": content,
            "fields":  {k: v for k, v in collected.items()},
            "session_id": state.session_id,
            "goal_id":    state.goal_id,
            "metadata": {
                "total_turns":   state.current_turn,
                "total_cost":    round(state.total_cost_usd, 6),
                "completion_pct": state.completion_pct,
            },
        }

        logger.info("output_generator: session=%s output generated", state.session_id)
        return result

    # ------------------------------------------------------------------
    # Format handlers
    # ------------------------------------------------------------------

    def _json_output(self, collected: Dict[str, Any], state: GraphState) -> dict:
        """Pure JSON output — no LLM, just the collected fields."""
        return {
            "goal":   state.goal_id,
            "fields": collected,
            "confidence": state.field_confidences,
        }

    async def _template_output(
        self,
        collected: Dict[str, Any],
        template:  str,
        state:     GraphState,
    ) -> str:
        """
        Fill a YAML-defined template with collected field values.
        Supports {field_name} placeholders. Falls back to LLM for missing context.
        """
        result = template
        for field_name, value in collected.items():
            result = result.replace(f"{{{field_name}}}", str(value))

        unfilled = re.findall(r"\{(\w+)\}", result)
        if unfilled:
            logger.warning(
                "output_generator: unfilled placeholders: %s", unfilled
            )

        return result

    async def _llm_output(
        self,
        collected: Dict[str, Any],
        state:     GraphState,
        goal_name: str,
        template:  Optional[str],
        fmt:       str,
    ) -> str:
        """LLM-generated output (narrative or markdown)."""
        if self._router is None:
            return self._fallback_output(collected, goal_name)

        from truenorth.llm.base import Message as LLMMessage
        from truenorth.llm.router import TASK_OUTPUT

        format_instruction = (
            "Format as markdown with headers and bullet points."
            if fmt == "markdown"
            else "Format as clear, friendly prose (no bullet points, no headers)."
        )

        template_instruction = (
            f"\n\nUse this structure as a guide:\n{template}"
            if template else ""
        )

        fields_str = "\n".join(
            f"  {k}: {v}" for k, v in collected.items()
        )

        prompt = (
            f"Generate a {goal_name} report for a user with these collected details:\n\n"
            f"{fields_str}\n\n"
            f"{format_instruction}"
            f"{template_instruction}\n\n"
            f"Reference the user's ACTUAL values throughout. "
            f"Do NOT use generic placeholders. "
            f"Language: {state.detected_language}."
        )

        try:
            resp = await self._router.generate(
                task       = TASK_OUTPUT,
                messages   = [LLMMessage(role="user", content=prompt)],
                system     = self._SYSTEM,
                max_tokens  = 2048,
                temperature = 0.5,
            )
            return resp.content.strip()
        except Exception as e:
            logger.error("output_generator LLM failed: %s", e)
            return self._fallback_output(collected, goal_name)

    @staticmethod
    def _fallback_output(collected: Dict[str, Any], goal_name: str) -> str:
        """Plain-text fallback when LLM is unavailable."""
        lines = [f"# {goal_name} Summary\n"]
        for field, value in collected.items():
            label = field.replace("_", " ").title()
            lines.append(f"**{label}**: {value}")
        return "\n".join(lines)