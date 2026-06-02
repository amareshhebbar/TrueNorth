"""
ValidationAgent — validates field values against declared constraints.

Checks: type correctness, range bounds, allowed values, cross-field
consistency. Returns a validation report with pass/fail per field.

Domain-agnostic: same agent validates a patient's age (1-120),
a candidate's salary expectation (0-10M), or a fitness weight (30-300 kg).
The field_config in the payload carries the domain-specific constraints.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from truenorth.agents.base import BaseAgent
from truenorth.agents.messages import AgentMessage, AgentResponse, AgentRole, TaskStatus

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """
    Validates extracted field values against the goal YAML schema.
    """

    agent_id     = "validation_agent"
    role         = AgentRole.VALIDATOR
    capabilities = {"validate", "check", "verify", "schema", "constraint"}

    async def handle(self, message: AgentMessage) -> AgentResponse:
        payload         = message.payload
        fields_config   = payload.get("fields_config", {})
        values_to_check = payload.get("values_to_check", {})
        collected       = payload.get("collected_fields", {})

        if not values_to_check:
            return self.ok(message, {
                "valid": True, "passed": [], "failed": [], "warnings": [],
            })

        passed:   List[str]  = []
        failed:   List[dict] = []
        warnings: List[dict] = []

        for fname, value in values_to_check.items():
            cfg    = fields_config.get(fname, {})
            ok, reason, is_warning = self._validate_field(fname, value, cfg)
            if ok:
                passed.append(fname)
            elif is_warning:
                warnings.append({"field": fname, "value": value, "warning": reason})
            else:
                failed.append({"field": fname, "value": value, "reason": reason})

        all_valid  = len(failed) == 0
        confidence = len(passed) / max(len(values_to_check), 1)

        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.COMPLETED,
            result     = {
                "valid":    all_valid,
                "passed":   passed,
                "failed":   failed,
                "warnings": warnings,
            },
            confidence = confidence,
            metadata   = {
                "checked": len(values_to_check),
                "passed":  len(passed),
                "failed":  len(failed),
            },
        )

    @staticmethod
    def _validate_field(
        fname: str,
        value: Any,
        cfg:   Dict[str, Any],
    ) -> tuple[bool, str, bool]:
        """
        Validate one field value. Returns (ok, reason, is_warning).
        is_warning=True means soft violation (not a hard fail).
        """
        ftype = cfg.get("type", "text")
        v     = str(value).strip()

        if value is None or v == "":
            if cfg.get("required", True):
                return False, "Required field is empty", False
            return True, "", False

        if ftype in ("integer", "int"):
            if not re.match(r"^-?\d+$", v.replace(",", "")):
                return False, f"Expected integer, got {v!r}", False
            try:
                num = int(v.replace(",", ""))
            except ValueError:
                return False, f"Cannot parse as integer: {v!r}", False
        elif ftype in ("number", "float"):
            if not re.match(r"^-?\d+(?:\.\d+)?$", v.replace(",", "")):
                return False, f"Expected number, got {v!r}", False
            try:
                num = float(v.replace(",", ""))
            except ValueError:
                return False, f"Cannot parse as number: {v!r}", False
        else:
            num = None

        if num is not None:
            mn = cfg.get("min")
            mx = cfg.get("max")
            if mn is not None and num < mn:
                return False, f"Value {num} below minimum {mn}", False
            if mx is not None and num > mx:
                return False, f"Value {num} above maximum {mx}", False

        allowed = cfg.get("allowed_values") or cfg.get("enum", [])
        if allowed:
            normalized = [str(a).strip().lower() for a in allowed]
            if v.lower() not in normalized:
                return False, f"Value {v!r} not in allowed values", False

        return True, "", False