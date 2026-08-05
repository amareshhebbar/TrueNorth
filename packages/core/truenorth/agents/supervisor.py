"""
AgentSupervisor — quality control layer for multi-agent workflows.

The Supervisor sits between the Orchestrator and the user. It:
  1. Reviews high-stakes agent results before they reach the user
  2. Checks confidence thresholds — rejects low-confidence results
  3. Detects hallucinations in agent outputs (uses HallucinationFirewall)
  4. Decides whether to approve, retry, or escalate to human review
  5. Maintains a verdict log for audit trail

Supervision levels:
  OFF      — no supervision (development mode)
  LIGHT    — only check confidence threshold
  STANDARD — confidence + basic consistency checks
  STRICT   — confidence + firewall + cross-agent consistency

The Supervisor is OPTIONAL. If not registered with the Orchestrator,
agents run unsupervised. For healthcare or legal goals, STRICT is recommended.

Sector-agnostic: same Supervisor for a medical agent, a legal agent,
a financial agent, or a fitness agent. The quality criteria are generic.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from truenorth.agents.messages import AgentResponse, SupervisorVerdict, TaskStatus

if TYPE_CHECKING:
    from truenorth.safety.hallucination_firewall import HallucinationFirewall

logger = logging.getLogger(__name__)

class SupervisionLevel(str, Enum):
    OFF      = "off"
    LIGHT    = "light"
    STANDARD = "standard"
    STRICT   = "strict"

class _ConfidenceCheck:
    """Reject results below the minimum confidence threshold."""

    def __init__(self, min_confidence: float = 0.60):
        self._min = min_confidence

    def check(
        self, response: AgentResponse, context: Dict[str, Any]
    ) -> tuple[bool, str]:
        if response.confidence < self._min:
            return False, (
                f"Confidence {response.confidence:.2f} below "
                f"threshold {self._min:.2f}"
            )
        return True, ""

class _ConsistencyCheck:
    def check(
        self, response: AgentResponse, context: Dict[str, Any]
    ) -> tuple[bool, str]:
        if not isinstance(response.result, dict):
            return True, ""

        collected = context.get("collected_fields", {})
        issues: list[str] = []

        for key, val in response.result.items():
            if key in collected:
                existing = str(collected[key]).strip().lower()
                new_val  = str(val).strip().lower()
                if existing and new_val and existing != new_val:

                    try:
                        e_num = float(existing.replace(",", ""))
                        n_num = float(new_val.replace(",", ""))
                        diff  = abs(e_num - n_num) / max(abs(e_num), 1e-9)
                        if diff > 0.05:
                            issues.append(
                                f"Field '{key}': agent says {val!r} "
                                f"but collected value is {collected[key]!r}"
                            )
                    except (ValueError, TypeError):
                        if existing != new_val:
                            issues.append(
                                f"Field '{key}': agent says {val!r} "
                                f"but collected value is {collected[key]!r}"
                            )

        if issues:
            return False, " | ".join(issues)
        return True, ""

class _LengthCheck:
    """Reject empty or suspiciously short results."""

    def __init__(self, min_length: int = 1):
        self._min = min_length

    def check(
        self, response: AgentResponse, context: Dict[str, Any]
    ) -> tuple[bool, str]:
        text = response.result_text.strip()
        if len(text) < self._min:
            return False, f"Result too short ({len(text)} chars)"
        return True, ""

class AgentSupervisor:
    """
    Quality-control supervisor for the multi-agent pipeline.

    Reviews agent results and decides: approve / retry / escalate.

    Usage:
        supervisor = AgentSupervisor(level=SupervisionLevel.STANDARD)
        orch = AgentOrchestrator(supervisor=supervisor)

    For medical / legal goals:
        supervisor = AgentSupervisor(
            level=SupervisionLevel.STRICT,
            min_confidence=0.75,
            firewall=HallucinationFirewall(router=router),
        )
    """

    def __init__(
        self,
        level:           SupervisionLevel   = SupervisionLevel.STANDARD,
        min_confidence:  float              = 0.60,
        max_retries:     int                = 1,
        firewall:        Optional["HallucinationFirewall"] = None,
    ):
        self._level          = level
        self._min_confidence = min_confidence
        self._max_retries    = max_retries
        self._firewall       = firewall
        self._verdicts:      List[SupervisorVerdict] = []

        self._checks = self._build_checks()

    async def review(
        self,
        response: AgentResponse,
        context:  Optional[Dict[str, Any]] = None,
    ) -> SupervisorVerdict:
        """
        Review an agent's response and produce a verdict.

        Args:
            response: The agent's AgentResponse to review
            context:  Optional dict with collected_fields, goal_config, etc.
        """
        if self._level == SupervisionLevel.OFF:
            return self._approve(response, score=1.0, feedback="Supervision disabled")

        ctx = context or {}

        issues:  List[str] = []
        scores:  List[float] = []

        for check in self._checks:
            ok, reason = check.check(response, ctx)
            if not ok:
                issues.append(reason)
            scores.append(1.0 if ok else 0.0)
        if (
            self._level == SupervisionLevel.STRICT
            and self._firewall is not None
            and isinstance(response.result, str)
        ):
            fw_result = await self._firewall.check(
                output           = response.result,
                collected_fields = ctx.get("collected_fields", {}),
                fields_config    = ctx.get("fields_config", {}),
                session_id       = ctx.get("session_id", "supervisor"),
            )
            if fw_result.blocked_count > 0:
                issues.append(
                    f"Hallucination firewall blocked {fw_result.blocked_count} claim(s)"
                )
                scores.append(0.0)
            else:
                scores.append(1.0)

        overall_score = sum(scores) / max(len(scores), 1)
        approved      = len(issues) == 0 and overall_score >= 0.70

        verdict = SupervisorVerdict(
            message_id = response.message_id,
            agent_id   = response.agent_id,
            approved   = approved,
            score      = overall_score,
            feedback   = "; ".join(issues) if issues else "All checks passed",
            retry      = not approved and response.status == TaskStatus.COMPLETED,
            escalate   = not approved and overall_score < 0.30,
            issues     = issues,
        )
        self._verdicts.append(verdict)

        log_fn = logger.info if approved else logger.warning
        log_fn(
            "supervisor: agent=%s approved=%s score=%.2f issues=%d level=%s",
            response.agent_id, approved, overall_score,
            len(issues), self._level.value,
        )
        return verdict

    def verdict_log(self) -> List[dict]:
        return [v.to_dict() for v in self._verdicts]

    def approval_rate(self) -> float:
        if not self._verdicts:
            return 1.0
        ok = sum(1 for v in self._verdicts if v.approved)
        return round(ok / len(self._verdicts), 3)

    def _build_checks(self):
        checks = []
        if self._level in (SupervisionLevel.LIGHT, SupervisionLevel.STANDARD,
                            SupervisionLevel.STRICT):
            checks.append(_ConfidenceCheck(self._min_confidence))
            checks.append(_LengthCheck(min_length=1))
        if self._level in (SupervisionLevel.STANDARD, SupervisionLevel.STRICT):
            checks.append(_ConsistencyCheck())
        return checks

    @staticmethod
    def _approve(
        response: AgentResponse, score: float, feedback: str
    ) -> SupervisorVerdict:
        return SupervisorVerdict(
            message_id = response.message_id,
            agent_id   = response.agent_id,
            approved   = True,
            score      = score,
            feedback   = feedback,
            retry      = False,
            escalate   = False,
        )
