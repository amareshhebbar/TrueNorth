"""
Every user message flows through these stages in order:
  1. PII scan          — redact sensitive data before any LLM sees it
  2. Language detect   — detect + track language; auto-switch response language
  3. Field extract     — extract structured values from natural language
  4. Emotion detect    — classify user emotional state
  5. Conflict check    — detect contradictions with prior collected values
  6. State update      — persist new field values + signals to GraphState
  7. Quality check     — measure conversation health (engagement, frustration)
  8. Reason            — decide what to do next (Reasoner)
  9. Plan response     — generate natural language response (ConversationPlanner)
  10. Cost tracking    — record token usage + enforce budget cap
  11. Session save     — persist state to storage
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from truenorth.core.graph_state import GraphState
from truenorth.core.yaml_loader import YAMLLoader
from truenorth.core.reasoner import Reasoner, ReasonerAction
from truenorth.core.session_manager import SessionManager
from truenorth.core.field_extractor import FieldExtractor
from truenorth.core.conversation_planner import ConversationPlanner
from truenorth.intelligence.emotion_detector import EmotionDetector
from truenorth.intelligence.confidence_scorer import ConfidenceScorer
from truenorth.intelligence.conflict_detector import ConflictDetector
from truenorth.intelligence.conversation_quality import ConversationQualityMonitor
from truenorth.intelligence.language_detector import LanguageDetector
from truenorth.llm.cost_tracker import CostTracker, BudgetExceededError
from truenorth.privacy.pii_detector import PIIDetector
from truenorth.output.generator import OutputGenerator
from truenorth.mcp.registry     import MCPRegistry
from truenorth.agents.orchestrator import AgentOrchestrator
from truenorth.mcp.tool_executor import ToolExecutor
from truenorth.safety.hallucination_firewall import HallucinationFirewall

logger = logging.getLogger(__name__)

@dataclass
class EngineResponse:
    """Returned by process_message() after each conversation turn."""
    text:          str
    session_id:    str
    turn:          int
    action:        str
    target_field:  Optional[str]
    is_complete:   bool
    final_output:  Optional[Dict[str, Any]] = None
    state_summary: Dict[str, Any] = field(default_factory=dict)
    cost_usd:      float = 0.0
    latency_ms:    int   = 0

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "session_id":    self.session_id,
            "turn":          self.turn,
            "action":        self.action,
            "target_field":  self.target_field,
            "is_complete":   self.is_complete,
            "final_output":  self.final_output,
            "cost_usd":      round(self.cost_usd, 6),
            "latency_ms":    self.latency_ms,
            "state_summary": self.state_summary,
        }

class TrueNorthEngine:
    """
    Main TrueNorth pipeline engine.

    Instantiate via:
      - TrueNorthEngine.from_yaml(path)           — from a goal YAML file
      - TrueNorthEngine(goal_config, **kwargs)    — from a pre-loaded config dict

    The engine is stateful per-session. Create one instance per active session.
    For multi-tenant production use, instantiate from the session store via SessionManager.
    """

    VERSION = "0.1.1"

    def __init__(
        self,
        goal_config:     dict,
        session_id:      Optional[str]     = None,
        user_id:         Optional[str]     = None,
        tenant_id:       Optional[str]     = None,
        router=None,
        session_manager: Optional[SessionManager] = None,
        cost_tracker:    Optional[CostTracker]    = None,
        firewall:        Optional[HallucinationFirewall] = None,
        mcp_registry:    Optional[MCPRegistry]    = None,
        orchestrator:    Optional[AgentOrchestrator] = None,
        config:          Optional[dict]           = None,
    ):
        self._goal_config     = goal_config
        self._config          = config or {}
        self._router          = router
        self._session_manager = session_manager
        self._cost_tracker    = cost_tracker or CostTracker()
        session_id = session_id or str(uuid.uuid4())

        self.state = GraphState.from_goal_config(
            goal_config = goal_config,
            session_id  = session_id,
            user_id     = user_id,
            tenant_id   = tenant_id,
        )
        self._pii       = PIIDetector()
        self._lang      = LanguageDetector()
        self._extractor = FieldExtractor(router=router)
        self._emotion   = EmotionDetector(router=router)
        self._confidence= ConfidenceScorer()
        self._conflict  = ConflictDetector()
        self._quality   = ConversationQualityMonitor()
        self._reasoner  = Reasoner(config=self._config.get("reasoner", {}))
        self._planner   = ConversationPlanner(router=router)
        _firewall = firewall or (
            HallucinationFirewall(router=router)
            if router is not None else None
        )
        try:
            from truenorth.output.source_tracer import SourceTracer as _ST
            _tracer = _ST()
        except ImportError:
            _tracer = None
        self._output    = OutputGenerator(
            router   = router,
            firewall = _firewall,
            tracer   = _tracer,
        )

        _mcp_servers = goal_config.get("mcp_servers", [])
        _registry    = mcp_registry or MCPRegistry()
        if _mcp_servers:
            for srv in _mcp_servers:
                if srv.get("builtin"):
                    _registry.register_builtins(server_name=srv["name"])
                elif srv.get("url"):
                    _registry.register_server_config(srv)
        self._mcp_registry    = _registry
        self._tool_executor   = (
            ToolExecutor(registry=_registry)
            if (_mcp_servers or mcp_registry) else None
        )
        self._firewall     = _firewall
        self._orchestrator = orchestrator

        budget = goal_config.get("budget", {}).get("max_cost_usd")
        if budget:
            self._cost_tracker.set_budget(session_id, float(budget))

        logger.info(
            "engine: session=%s goal=%s fields=%d",
            session_id,
            goal_config.get("id", "?"),
            len(self.state.fields_config),
        )

    @classmethod
    async def from_yaml(
        cls,
        yaml_path: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> "TrueNorthEngine":
        """
        Create an engine from a goal YAML file.
        """
        config = YAMLLoader.load(yaml_path)
        return cls(goal_config=config, session_id=session_id, **kwargs)

    @classmethod
    async def from_session(
        cls,
        session_id: str,
        session_manager: SessionManager,
        **kwargs,
    ) -> Optional["TrueNorthEngine"]:
        """
        Resume an existing session from storage.
        Returns None if session not found.
        """
        state_data = await session_manager.load(session_id)
        if state_data is None:
            return None

        state = GraphState.from_dict(state_data)
        engine = cls(
            goal_config     = state.goal_config,
            session_id      = session_id,
            session_manager = session_manager,
            **kwargs,
        )
        engine.state = state
        engine.state.is_resumed = True
        return engine

    async def start(self) -> EngineResponse:
        """
        Start the conversation — returns the first agent message.
        Call this once when a session begins, before any user input.
        """
        decision = self._reasoner.decide(self.state)
        response_text = await self._planner.plan(decision, self.state)

        self.state.add_turn("assistant", response_text, metadata={
            "target_field": decision.target_field,
        })
        await self._save_state()

        return EngineResponse(
            text         = response_text,
            session_id   = self.state.session_id,
            turn         = self.state.current_turn,
            action       = decision.action.value,
            target_field = decision.target_field,
            is_complete  = False,
        )

    async def process_message(self, user_message: str) -> EngineResponse:
        """
        Process one user message through the full pipeline.

        Args:
            user_message: Raw text from the user

        Returns:
            EngineResponse with the agent's reply and session state
        """
        start_time = time.perf_counter()
        self.state.current_turn += 1
        self.state.current_input = user_message

        try:

            pii_result = self._pii.scan(user_message)
            safe_text  = pii_result.redacted

            lang_result = self._lang.detect_from_history(
                self.state.user_messages + [user_message]
            )
            self.state.detected_language = lang_result.language_code
            self.state.is_romanized      = lang_result.is_romanized

            _last_target = (
                self.state.turn_history[-1].get("target_field")
                if self.state.turn_history else None
            )
            extraction = await self._extractor.extract(
                user_message  = safe_text,
                fields_config = self.state.fields_config,
                context       = self.state.collected_fields,
                target_field  = _last_target,
            )

            emotion = await self._emotion.detect(safe_text, use_llm=self._router is not None)
            self.state.current_emotion = emotion.to_dict()

            new_values = extraction.as_map()
            _turn_map = getattr(self.state, "_field_turn_map", {})
            conflicts  = self._conflict.check(
                new_extractions   = new_values,
                collected         = self.state.collected_fields,
                fields_config     = self.state.fields_config,
                current_turn      = self.state.current_turn,
                turn_map          = _turn_map,
                field_confidences = self.state.field_confidences,
            )
            for c in conflicts:
                self.state.active_conflicts.append(c.to_dict())

            conflict_fields = {c.get("field") for c in self.state.active_conflicts if not c.get("resolved")}
            if not hasattr(self.state, "_field_turn_map"):
                self.state._field_turn_map = {}
            for ef in extraction.fields:
                if ef.name not in conflict_fields:
                    self.state.set_field(ef.name, ef.value, ef.confidence)
                    if ef.name not in self.state._field_turn_map:
                        self.state._field_turn_map[ef.name] = self.state.current_turn
            self.state.last_extraction = (
                extraction.fields[0].to_dict() if extraction.fields else None
            )

            confidence_scores = self._confidence.score_all(
                collected_fields = self.state.collected_fields,
                fields_config    = self.state.fields_config,
                extraction_meta  = {
                    ef.name: {"confidence": ef.confidence, "source_text": ef.raw_text}
                    for ef in extraction.fields
                },
            )
            self.state.field_confidences = {
                k: v.score for k, v in confidence_scores.items()
            }

            quality = self._quality.check(
                turn_number            = self.state.current_turn,
                user_message           = user_message,
                turn_history           = self.state.turn_history,
                fields_collected       = len(self.state.collected_fields),
                total_required_fields  = len(self.state.required_fields),
            )
            self.state.quality_reports.append(quality.to_dict())

            self.state.add_turn("user", user_message, metadata={
                "pii_detected":   pii_result.has_pii,
                "language":       lang_result.language_code,
                "emotion":        emotion.label,
                "fields_extracted": [ef.name for ef in extraction.fields],
            })

            decision      = self._reasoner.decide(self.state)
            response_text = await self._planner.plan(decision, self.state)

            final_output = None
            if decision.action == ReasonerAction.GENERATE_OUTPUT:
                final_output = await self._output.generate(self.state)
                self.state.is_complete  = True
                self.state.final_output = final_output
                if self._session_manager:
                    await self._session_manager.complete(
                        self.state.session_id, output=final_output
                    )

            session_cost = self._cost_tracker.get_session_cost(self.state.session_id)
            self.state.total_cost_usd = session_cost.total_cost_usd

            if self._tool_executor and response_text:
                response_text, tool_logs = await self._tool_executor.run(
                    response_text = response_text,
                    state         = self.state,
                )
                if not hasattr(self.state, "tool_call_log"):
                    self.state.tool_call_log = []
                self.state.tool_call_log.extend([t.to_dict() for t in tool_logs])

            self.state.add_turn("assistant", response_text, metadata={
                "target_field": decision.target_field,
            })
            await self._save_state()

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info(
                "engine: turn=%d session=%s action=%s field=%s "
                "lang=%s emotion=%s extracted=%d cost=$%.4f latency=%dms",
                self.state.current_turn,
                self.state.session_id,
                decision.action.value,
                decision.target_field or "-",
                lang_result.language_code,
                emotion.label,
                len(extraction.fields),
                self.state.total_cost_usd,
                latency_ms,
            )

            return EngineResponse(
                text          = response_text,
                session_id    = self.state.session_id,
                turn          = self.state.current_turn,
                action        = decision.action.value,
                target_field  = decision.target_field,
                is_complete   = self.state.is_complete,
                final_output  = final_output,
                cost_usd      = self.state.total_cost_usd,
                latency_ms    = latency_ms,
                state_summary = self._state_summary(),
            )

        except BudgetExceededError as e:
            logger.warning("engine: %s", e)
            budget_response = (
                "We've reached the session cost limit. "
                "Your progress is saved — resume with your session ID."
            )
            self.state.add_turn("assistant", budget_response)
            await self._save_state()
            if self._session_manager:
                await self._session_manager.fail(self.state.session_id, str(e))
            return EngineResponse(
                text         = budget_response,
                session_id   = self.state.session_id,
                turn         = self.state.current_turn,
                action       = "budget_exceeded",
                target_field = None,
                is_complete  = False,
            )

        except Exception as e:
            logger.exception("engine: unhandled error in process_message: %s", e)
            self.state.error = str(e)
            await self._save_state()
            raise

    async def process_message_stream(
        self,
        user_message: str,
    ) -> AsyncIterator[str]:
        """
        Streaming version of process_message().
        Yields response text token by token, then yields a sentinel JSON object
        with the full EngineResponse metadata.

        Usage:
            async for chunk in engine.process_message_stream("hello"):
                if chunk.startswith('{"session_id"'):
                    meta = json.loads(chunk)  # final metadata
                else:
                    print(chunk, end="", flush=True)  # text token
        """
        start_time = time.perf_counter()
        self.state.current_turn += 1
        self.state.current_input = user_message

        pii_result  = self._pii.scan(user_message)
        safe_text   = pii_result.redacted
        lang_result = self._lang.detect_from_history(self.state.user_messages + [user_message])
        self.state.detected_language = lang_result.language_code

        extraction = await self._extractor.extract(
            user_message  = safe_text,
            fields_config = self.state.fields_config,
            context       = self.state.collected_fields,
        )
        emotion = await self._emotion.detect(safe_text, use_llm=False)
        self.state.current_emotion = emotion.to_dict()

        for ef in extraction.fields:
            self.state.set_field(ef.name, ef.value, ef.confidence)

        self.state.add_turn("user", user_message)
        decision = self._reasoner.decide(self.state)

        if self._router and decision.action not in (
            ReasonerAction.GENERATE_OUTPUT, ReasonerAction.BUDGET_EXCEEDED
        ):
            from truenorth.llm.base import Message as LLMMessage
            from truenorth.llm.router import TASK_CONVERSE

            full_text = []
            async for chunk in self._router.generate_stream(
                task     = TASK_CONVERSE,
                messages = [LLMMessage(role="user", content=f"Turn {self.state.current_turn}")],
                max_tokens = 200,
            ):
                if chunk.delta:
                    full_text.append(chunk.delta)
                    yield chunk.delta

            response_text = "".join(full_text)
        else:
            response_text = await self._planner.plan(decision, self.state)
            yield response_text

        import json
        self.state.add_turn("assistant", response_text)
        await self._save_state()
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        meta = EngineResponse(
            text         = response_text,
            session_id   = self.state.session_id,
            turn         = self.state.current_turn,
            action       = decision.action.value,
            target_field = decision.target_field,
            is_complete  = self.state.is_complete,
            latency_ms   = latency_ms,
        )
        yield json.dumps(meta.to_dict())

    def get_state(self) -> dict:
        """Return current session state as a serializable dict."""
        return self.state.to_dict()

    def get_collected_fields(self) -> Dict[str, Any]:
        """Return all collected field values."""
        return dict(self.state.collected_fields)

    def get_missing_fields(self) -> List[str]:
        """Return names of required fields not yet collected."""
        return self.state.missing_required

    def get_session_id(self) -> str:
        return self.state.session_id

    def explain_decision(self) -> str:
        """Return a human-readable explanation of what the engine would do next (for dry-run)."""
        return self._reasoner.explain(self.state)

    async def force_output(self) -> Dict[str, Any]:
        """Force final output generation even if not all fields collected."""
        return await self._output.generate(self.state)

    async def _save_state(self) -> None:
        """Persist state to session manager if available."""
        self.state.updated_at = time.time()
        if self._session_manager:
            try:
                await self._session_manager.save(
                    self.state.session_id,
                    self.state.to_dict(),
                )
            except Exception as e:
                logger.warning("engine: state save failed: %s", e)

    def _state_summary(self) -> dict:
        """Compact state summary for API responses."""
        return {
            "collected":   list(self.state.collected_fields.keys()),
            "missing":     self.state.missing_required,
            "completion":  self.state.completion_pct,
            "turn":        self.state.current_turn,
            "language":    self.state.detected_language,
            "cost_usd":    round(self.state.total_cost_usd, 6),
        }

    def __repr__(self) -> str:
        return (
            f"TrueNorthEngine("
            f"session={self.state.session_id!r}, "
            f"goal={self.state.goal_id!r}, "
            f"turn={self.state.current_turn}, "
            f"completion={self.state.completion_pct:.0%})"
        )
