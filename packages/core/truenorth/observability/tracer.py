"""
TrueNorthTracer — per-turn structured event collection.

Records every significant action in one conversation turn as a typed
LogEvent, then flushes to configured sinks (in-memory, stdout, HTTP
endpoint, Datadog, CloudWatch, etc.).

One TurnTrace aggregates everything that happened in a single turn:
  - User input (text, language, chars)
  - PII detection result
  - Field extractions attempted and their outcomes
  - Emotion detected
  - Conflicts caught
  - Every LLM call: model, task, tokens, cost, latency
  - Hallucination firewall verdict
  - Response sent to user

A SessionTrace holds all TurnTraces for one session.

Integration with engine.py:
    tracer = TrueNorthTracer()
    engine = TrueNorthEngine(goal_config=config, tracer=tracer)
    # Engine calls tracer.record_* at each pipeline stage
    # At session end:
    summary = tracer.session_summary(session_id)

Sinks:
    tracer.add_sink(StdoutSink())             # development
    tracer.add_sink(HTTPSink("http://..."))   # production
    tracer.add_sink(MemorySink())             # testing

Sector-agnostic: same tracer for medical intake, legal case,
HR screening, financial plan, fitness coach.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from truenorth.observability.log_categories import (
    LogCategory, LogEvent, LogLevel,
    conversation_event, extraction_event, emotion_event,
    conflict_event, cost_event, hallucination_event, compliance_event,
    make_event,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TurnTrace — everything that happened in one conversation turn
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnTrace:
    """Complete trace for one conversation turn."""
    session_id:       str
    goal_id:          str
    turn:             int
    started_at:       float = field(default_factory=time.time)
    finished_at:      Optional[float] = None

    # User input
    user_text_chars:  int   = 0
    user_language:    str   = "en"

    # PII
    pii_detected:     bool  = False
    pii_types:        List[str] = field(default_factory=list)

    # Extractions
    extractions:      List[dict] = field(default_factory=list)   # {field, value, conf, success}

    # Emotion
    emotion:          Optional[str]   = None
    emotion_valence:  Optional[float] = None

    # Conflict
    conflicts:        List[dict] = field(default_factory=list)

    # LLM calls
    llm_calls:        List[dict] = field(default_factory=list)   # {model, task, tokens, cost, ms}

    # Hallucination
    fw_verdict:       Optional[str]  = None   # CLEAN | FLAGGED | BLOCKED
    fw_blocked:       int            = 0

    # Response
    response_chars:   int  = 0

    @property
    def latency_ms(self) -> int:
        if self.finished_at:
            return int((self.finished_at - self.started_at) * 1000)
        return 0

    @property
    def total_cost_usd(self) -> float:
        return sum(c.get("cost_usd", 0) for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.get("total_tokens", 0) for c in self.llm_calls)

    @property
    def extraction_success_rate(self) -> float:
        if not self.extractions:
            return 1.0
        ok = sum(1 for e in self.extractions if e.get("success"))
        return ok / len(self.extractions)

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "goal_id":       self.goal_id,
            "turn":          self.turn,
            "latency_ms":    self.latency_ms,
            "user_chars":    self.user_text_chars,
            "language":      self.user_language,
            "pii_detected":  self.pii_detected,
            "pii_types":     self.pii_types,
            "extractions":   self.extractions,
            "emotion":       self.emotion,
            "conflicts":     self.conflicts,
            "llm_calls":     self.llm_calls,
            "fw_verdict":    self.fw_verdict,
            "response_chars":self.response_chars,
            "total_cost_usd":round(self.total_cost_usd, 8),
            "total_tokens":  self.total_tokens,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  SessionTrace — all turns for one conversation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionTrace:
    """All traces for one complete session."""
    session_id:   str
    goal_id:      str
    user_id:      Optional[str] = None
    tenant_id:    Optional[str] = None
    turns:        List[TurnTrace] = field(default_factory=list)
    started_at:   float = field(default_factory=time.time)
    finished_at:  Optional[float] = None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.total_cost_usd for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def total_latency_ms(self) -> int:
        return sum(t.latency_ms for t in self.turns)

    @property
    def pii_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.pii_detected)

    @property
    def conflict_count(self) -> int:
        return sum(len(t.conflicts) for t in self.turns)

    @property
    def hallucinations_blocked(self) -> int:
        return sum(t.fw_blocked for t in self.turns)

    @property
    def extraction_count(self) -> int:
        return sum(len(t.extractions) for t in self.turns)

    def session_summary(self) -> dict:
        return {
            "session_id":         self.session_id,
            "goal_id":            self.goal_id,
            "turn_count":         self.turn_count,
            "total_cost_usd":     round(self.total_cost_usd, 6),
            "total_tokens":       self.total_tokens,
            "total_latency_ms":   self.total_latency_ms,
            "pii_turns":          self.pii_turn_count,
            "conflicts":          self.conflict_count,
            "hallucinations_blocked": self.hallucinations_blocked,
            "extractions":        self.extraction_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Sinks — where events go
# ─────────────────────────────────────────────────────────────────────────────

class TraceSink(ABC):
    """Abstract event sink. Implement emit() to route events anywhere."""

    @abstractmethod
    async def emit(self, event: LogEvent) -> None:
        """Emit one log event. Should be non-blocking (fire and forget)."""

    async def flush(self) -> None:
        """Optional: flush buffered events. Called at session end."""


class MemorySink(TraceSink):
    """Stores all events in memory. Used for testing and dry-runs."""

    def __init__(self, max_events: int = 10_000):
        self._events: List[LogEvent] = []
        self._max    = max_events

    async def emit(self, event: LogEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max:
            self._events = self._events[-self._max:]

    @property
    def events(self) -> List[LogEvent]:
        return list(self._events)

    def by_category(self, category: LogCategory) -> List[LogEvent]:
        return [e for e in self._events if e.category == category]

    def by_session(self, session_id: str) -> List[LogEvent]:
        return [e for e in self._events if e.session_id == session_id]

    def clear(self) -> None:
        self._events.clear()


class StdoutSink(TraceSink):
    """Prints JSON events to stdout. For development and debugging."""

    def __init__(self, pretty: bool = False, categories: Optional[List[LogCategory]] = None):
        self._pretty = pretty
        self._filter = set(categories) if categories else None

    async def emit(self, event: LogEvent) -> None:
        if self._filter and event.category not in self._filter:
            return
        indent = 2 if self._pretty else None
        print(json.dumps(event.to_dict(), default=str, indent=indent))


class CallbackSink(TraceSink):
    """Calls a user-provided async function with each event."""

    def __init__(self, callback: Callable[[LogEvent], None]):
        self._cb = callback

    async def emit(self, event: LogEvent) -> None:
        try:
            result = self._cb(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug("tracer: callback sink error: %s", e)


class HTTPSink(TraceSink):
    """
    POSTs events as JSON to an HTTP endpoint (Datadog, Splunk, custom).
    Buffers events and flushes in batches to reduce HTTP overhead.
    """

    def __init__(
        self,
        endpoint:   str,
        batch_size: int           = 50,
        headers:    Optional[dict] = None,
        timeout_s:  float          = 5.0,
    ):
        self._endpoint   = endpoint
        self._batch_size = batch_size
        self._headers    = headers or {"Content-Type": "application/json"}
        self._timeout    = timeout_s
        self._buffer:    List[dict] = []

    async def emit(self, event: LogEvent) -> None:
        self._buffer.append(event.to_dict())
        if len(self._buffer) >= self._batch_size:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    self._endpoint,
                    json    = {"events": batch},
                    headers = self._headers,
                    timeout = self._timeout,
                )
        except Exception as e:
            logger.warning("tracer: HTTP sink flush failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
#  TrueNorthTracer — main entry point
# ─────────────────────────────────────────────────────────────────────────────

class TrueNorthTracer:
    """
    Per-session structured event tracer for TrueNorth.

    Records every pipeline stage as a typed LogEvent and routes
    events to registered sinks. Maintains per-session TurnTraces
    for in-process analytics
    """

    def __init__(self):
        self._sinks:    List[TraceSink] = []
        self._sessions: Dict[str, SessionTrace] = {}

    # ------------------------------------------------------------------
    # Sink management
    # ------------------------------------------------------------------

    def add_sink(self, sink: TraceSink) -> "TrueNorthTracer":
        self._sinks.append(sink)
        return self

    def get_sink(self, sink_type: type) -> Optional[TraceSink]:
        for s in self._sinks:
            if isinstance(s, sink_type):
                return s
        return None

    # ------------------------------------------------------------------
    # Session / turn lifecycle
    # ------------------------------------------------------------------

    def session_start(
        self,
        session_id: str,
        goal_id:    str,
        user_id:    Optional[str]  = None,
        tenant_id:  Optional[str]  = None,
    ) -> None:
        self._sessions[session_id] = SessionTrace(
            session_id = session_id,
            goal_id    = goal_id,
            user_id    = user_id,
            tenant_id  = tenant_id,
        )
        self._safe_emit(make_event(
            LogCategory.SYSTEM, session_id, goal_id,
            data={"action": "session_start", "user_id": user_id},
        ))

    def turn_start(self, session_id: str, goal_id: str, turn: int) -> TurnTrace:
        sess = self._get_or_create_session(session_id, goal_id)
        trace = TurnTrace(session_id=session_id, goal_id=goal_id, turn=turn)
        sess.turns.append(trace)
        return trace

    def turn_end(
        self,
        session_id: str,
        turn:       int,
        response_text: str = "",
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.finished_at    = time.time()
            trace.response_chars = len(response_text)

    def session_end(self, session_id: str) -> None:
        sess = self._sessions.get(session_id)
        if sess:
            sess.finished_at = time.time()
        self._safe_emit(make_event(
            LogCategory.SYSTEM, session_id,
            goal_id = sess.goal_id if sess else "",
            data    = {"action": "session_end",
                       "summary": sess.session_summary() if sess else {}},
        ))

    # ------------------------------------------------------------------
    # Record methods — called by engine pipeline stages
    # ------------------------------------------------------------------

    def record_user_input(
        self,
        session_id: str, goal_id: str, turn: int,
        text:       str,
        language:   str = "en",
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.user_text_chars = len(text)
            trace.user_language   = language
        self._safe_emit(conversation_event(
            session_id, goal_id, turn, "user", text, language,
        ))

    def record_pii(
        self,
        session_id: str, goal_id: str, turn: int,
        pii_types:  List[str],
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.pii_detected = bool(pii_types)
            trace.pii_types    = pii_types
        if pii_types:
            self._safe_emit(compliance_event(
                session_id, goal_id, turn,
                action    = "pii_detected",
                framework = "internal",
                details   = {"types": pii_types},
                level     = LogLevel.WARNING,
            ))

    def record_extraction(
        self,
        session_id: str, goal_id: str, turn: int,
        field_name: str,
        value:      Any,
        confidence: float,
        success:    bool,
        model:      str = "",
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.extractions.append({
                "field": field_name, "value": str(value)[:80] if value is not None else None,
                "confidence": round(confidence, 3), "success": success,
            })
        self._safe_emit(extraction_event(
            session_id, goal_id, turn, field_name, value, confidence, success, model,
        ))

    def record_emotion(
        self,
        session_id: str, goal_id: str, turn: int,
        emotion:    str,
        valence:    float,
        arousal:    float,
        shifted:    bool = False,
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.emotion         = emotion
            trace.emotion_valence = valence
        self._safe_emit(emotion_event(
            session_id, goal_id, turn, emotion, valence, arousal, shifted,
        ))

    def record_conflict(
        self,
        session_id:    str, goal_id: str, turn: int,
        field_name:    str,
        prev_value:    Any,
        new_value:     Any,
        conflict_type: str,
        severity:      str,
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.conflicts.append({
                "field": field_name, "prev": str(prev_value)[:60],
                "new":   str(new_value)[:60], "type": conflict_type, "severity": severity,
            })
        self._safe_emit(conflict_event(
            session_id, goal_id, turn, field_name, prev_value, new_value, conflict_type, severity,
        ))

    def record_llm_call(
        self,
        session_id:    str, goal_id: str, turn: int,
        model:         str,
        task_type:     str,
        input_tokens:  int,
        output_tokens: int,
        cost_usd:      float,
        latency_ms:    int,
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.llm_calls.append({
                "model":          model,
                "task":           task_type,
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
                "total_tokens":   input_tokens + output_tokens,
                "cost_usd":       round(cost_usd, 8),
                "latency_ms":     latency_ms,
            })
        self._safe_emit(cost_event(
            session_id, goal_id, turn, model, task_type,
            input_tokens, output_tokens, cost_usd, latency_ms,
        ))

    def record_hallucination(
        self,
        session_id:    str, goal_id: str, turn: int,
        verdict:       str,
        claims_total:  int,
        claims_blocked: int,
        output_snippet: str = "",
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.fw_verdict = verdict
            trace.fw_blocked = claims_blocked
        self._safe_emit(hallucination_event(
            session_id, goal_id, turn, verdict, claims_total, claims_blocked, output_snippet,
        ))

    def record_response(
        self,
        session_id: str, goal_id: str, turn: int,
        text:       str,
        latency_ms: int = 0,
    ) -> None:
        trace = self._get_turn(session_id, turn)
        if trace:
            trace.response_chars = len(text)
        self._safe_emit(conversation_event(
            session_id, goal_id, turn, "assistant", text, latency_ms=latency_ms,
        ))

    def record_compliance(
        self,
        session_id: str, goal_id: str, turn: int,
        action:     str,
        framework:  str,
        user_id:    Optional[str] = None,
        details:    Optional[dict] = None,
    ) -> None:
        self._safe_emit(compliance_event(
            session_id, goal_id, turn, action, framework, user_id, details,
        ))

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------

    def session_summary(self, session_id: str) -> Optional[dict]:
        sess = self._sessions.get(session_id)
        return sess.session_summary() if sess else None

    def get_session(self, session_id: str) -> Optional[SessionTrace]:
        return self._sessions.get(session_id)

    def get_turn_trace(self, session_id: str, turn: int) -> Optional[TurnTrace]:
        return self._get_turn(session_id, turn)

    def all_session_ids(self) -> List[str]:
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe_emit(self, event: LogEvent) -> None:
        """Fire-and-forget emit that works inside and outside async contexts."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit(event))
        except RuntimeError:
            try:
                asyncio.run(self._emit(event))
            except Exception:
                pass  

    async def _emit(self, event: LogEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception as e:
                logger.debug("tracer: sink %s emit error: %s", type(sink).__name__, e)

    def _get_or_create_session(self, session_id: str, goal_id: str) -> SessionTrace:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionTrace(session_id=session_id, goal_id=goal_id)
        return self._sessions[session_id]

    def _get_turn(self, session_id: str, turn: int) -> Optional[TurnTrace]:
        sess = self._sessions.get(session_id)
        if not sess:
            return None
        for t in reversed(sess.turns):
            if t.turn == turn:
                return t
        return None