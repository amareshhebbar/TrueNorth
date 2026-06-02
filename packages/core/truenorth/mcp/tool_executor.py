"""
truenorth/mcp/tool_executor.py

ToolExecutor — Stage 13 of the TrueNorth engine pipeline.

After the ConversationPlanner generates a response, the ToolExecutor:
  1. Scans the response for TOOL_CALL patterns
  2. Validates the tool call against the registered schema
  3. Executes the tool via the MCPRegistry
  4. Injects the result back into the response
  5. Logs the call in the session audit trail

Tool call syntax (in LLM response):
  TOOL_CALL: web_search({"query": "BMI for 28 years old"})
  TOOL_CALL: calculator({"expression": "65 / (1.63 ** 2)"})
  TOOL_CALL: datetime_tool({"timezone": "Asia/Kolkata"})

After execution, the TOOL_CALL line is replaced with the result:
  [Tool result: {"results": [...]}]

Security:
  - Tool calls are validated against the registered schema before execution
  - Arguments are sanitized to prevent injection
  - Results are length-capped before LLM context injection (avoid context poisoning)
  - Tool calls that reference unregistered tools are silently skipped
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from truenorth.mcp.types import ToolResult, ToolResultStatus

if TYPE_CHECKING:
    from truenorth.mcp.registry import MCPRegistry
    from truenorth.core.graph_state import GraphState

logger = logging.getLogger(__name__)

# Maximum result length injected into LLM context (chars)
_MAX_RESULT_LENGTH = 2000

# Regex to detect TOOL_CALL patterns in LLM output
_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((\{.*?\})\)",
    re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

class ToolValidationError(Exception):
    """Raised when tool arguments don't match the declared schema."""


def _validate_arguments(
    arguments:    Dict[str, Any],
    input_schema: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Validate arguments against JSON Schema (subset: required, types).
    Returns (is_valid, error_message_or_None).
    """
    required = input_schema.get("required", [])
    for req in required:
        if req not in arguments:
            return False, f"Required parameter '{req}' is missing"

    props = input_schema.get("properties", {})
    for arg_name, arg_val in arguments.items():
        prop = props.get(arg_name)
        if prop is None:
            continue  # extra args are allowed by default
        expected_type = prop.get("type")
        if expected_type == "string"  and not isinstance(arg_val, str):
            return False, f"'{arg_name}' must be a string"
        if expected_type == "integer" and not isinstance(arg_val, int):
            return False, f"'{arg_name}' must be an integer"
        if expected_type == "number"  and not isinstance(arg_val, (int, float)):
            return False, f"'{arg_name}' must be a number"
        if expected_type == "boolean" and not isinstance(arg_val, bool):
            return False, f"'{arg_name}' must be a boolean"
        if expected_type == "array"   and not isinstance(arg_val, list):
            return False, f"'{arg_name}' must be an array"

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
#  ToolExecutionLog — audit entry per tool call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolExecutionLog:
    """One tool call execution, stored in session state for audit."""
    session_id:  str
    turn:        int
    tool_name:   str
    arguments:   Dict[str, Any]
    result:      ToolResult
    called_at:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn":       self.turn,
            "tool":       self.tool_name,
            "arguments":  {k: str(v)[:100] for k, v in self.arguments.items()},
            "status":     self.result.status.value,
            "latency_ms": self.result.latency_ms,
            "error":      self.result.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  ToolExecutor
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Scans LLM responses for TOOL_CALL patterns, executes them, and
    injects results back into the conversation.
    """

    def __init__(
        self,
        registry:    "MCPRegistry",
        max_per_turn: int   = 3,    
        max_result:   int   = _MAX_RESULT_LENGTH,
        validate:     bool  = True,
    ):
        self._registry    = registry
        self._max_per_turn = max_per_turn
        self._max_result   = max_result
        self._validate     = validate

    # ------------------------------------------------------------------
    # Main entry point — called from engine.py
    # ------------------------------------------------------------------

    async def run(
        self,
        response_text: str,
        state:         "GraphState",
    ) -> Tuple[str, List[ToolExecutionLog]]:
        """
        Scan response for tool calls, execute them, inject results.

        Args:
            response_text: LLM response that may contain TOOL_CALL: patterns
            state:         Current session state (for session_id, turn, etc.)

        Returns:
            (modified_response_text, list_of_execution_logs)
        """
        if not response_text or "TOOL_CALL:" not in response_text:
            return response_text, []

        session_id = state.session_id
        turn       = state.current_turn
        calls      = self._extract_calls(response_text)

        if not calls:
            return response_text, []
        if len(calls) > self._max_per_turn:
            logger.warning(
                "tool_executor: %d tool calls in one turn (max %d) — truncating",
                len(calls), self._max_per_turn,
            )
            calls = calls[:self._max_per_turn]

        logs: List[ToolExecutionLog] = []
        modified = response_text

        for tool_name, arguments, original_match in calls:
            result = await self.execute_single(
                tool_name  = tool_name,
                arguments  = arguments,
                session_id = session_id,
                turn       = turn,
            )
            log = ToolExecutionLog(
                session_id = session_id,
                turn       = turn,
                tool_name  = tool_name,
                arguments  = arguments,
                result     = result,
            )
            logs.append(log)
            result_text = self._format_result(result)
            modified = modified.replace(original_match, result_text, 1)

            logger.info(
                "tool_executor: session=%s turn=%d tool=%s status=%s latency=%dms",
                session_id, turn, tool_name,
                result.status.value, result.latency_ms,
            )

        return modified, logs

    async def execute_single(
        self,
        tool_name:  str,
        arguments:  Dict[str, Any],
        session_id: str = "",
        turn:       int = 0,
        timeout:    float = 30.0,
    ) -> ToolResult:
        """
        Execute one tool call directly (without scanning response text).
        Used by tests and the CLI.
        """
        rt = self._registry.get_tool(tool_name)
        if rt is None:
            logger.warning("tool_executor: unknown tool '%s'", tool_name)
            return ToolResult(
                call_id   = "",
                tool_name = tool_name,
                status    = ToolResultStatus.SKIPPED,
                content   = None,
                error     = f"Tool '{tool_name}' not registered",
            )

        if self._validate:
            ok, err = _validate_arguments(arguments, rt.tool.input_schema)
            if not ok:
                logger.warning(
                    "tool_executor: validation failed tool=%s: %s", tool_name, err
                )
                return ToolResult(
                    call_id   = "",
                    tool_name = tool_name,
                    status    = ToolResultStatus.ERROR,
                    content   = None,
                    error     = f"Argument validation failed: {err}",
                )

        return await self._registry.call_tool(
            tool_name  = tool_name,
            arguments  = arguments,
            session_id = session_id,
            turn       = turn,
            timeout    = timeout,
        )

    # ------------------------------------------------------------------
    # Tool call detection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_calls(
        text: str,
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Find all TOOL_CALL patterns in text.
        Returns list of (tool_name, arguments, original_match_text).
        """
        results = []
        for match in _TOOL_CALL_RE.finditer(text):
            tool_name    = match.group(1)
            args_str     = match.group(2)
            original     = match.group(0)
            try:
                arguments = json.loads(args_str)
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError as e:
                logger.warning(
                    "tool_executor: invalid JSON in TOOL_CALL for '%s': %s",
                    tool_name, e,
                )
                arguments = {}
            results.append((tool_name, arguments, original))
        return results

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    def _format_result(self, result: ToolResult) -> str:
        """Format a ToolResult for injection into the response text."""
        if not result.is_success:
            status = result.status.value
            err    = result.error or "unknown error"
            return f"[Tool '{result.tool_name}' {status}: {err}]"

        text = result.text
        if len(text) > self._max_result:
            text = text[:self._max_result] + "...[truncated]"

        return f"[Tool '{result.tool_name}' result: {text}]"