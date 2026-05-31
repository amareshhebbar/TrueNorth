"""
Structured log event types for TrueNorth.

Every significant action in the engine produces a typed LogEvent.
Events route into category-specific streams so you can subscribe to
exactly what you need — filter COST events for billing, COMPLIANCE
events for audit, HALLUCINATION events for safety review.

Seven log streams:
  CONVERSATION   — every user message + agent response turn
  EXTRACTION     — field extraction attempts and results
  EMOTION        — detected emotion + tone shift events
  CONFLICT       — contradictions caught between turns
  COST           — every LLM call with token counts and USD
  HALLUCINATION  — firewall verdicts (CLEAN / FLAGGED / BLOCKED)
  COMPLIANCE     — consent grants, rights requests, PII detections

Usage:
    from truenorth.observability.log_categories import LogCategory, LogEvent, make_event

    event = make_event(
        LogCategory.EXTRACTION,
        session_id = "sess-abc",
        goal_id    = "medical_intake",
        data = {"field": "chief_complaint", "value": "lower back pain", "confidence": 0.91},
    )
    # emit to tracer, Datadog, CloudWatch, etc.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LogCategory(str, Enum):
    CONVERSATION   = "conversation"
    EXTRACTION     = "extraction"
    EMOTION        = "emotion"
    CONFLICT       = "conflict"
    COST           = "cost"
    HALLUCINATION  = "hallucination"
    COMPLIANCE     = "compliance"
    SYSTEM         = "system"        # engine lifecycle events


class LogLevel(str, Enum):
    DEBUG   = "debug"
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


@dataclass
class LogEvent:
    """
    One structured log event. Every field has a fixed type —
    no free-form dicts at the top level.
    """
    event_id:    str
    category:    LogCategory
    level:       LogLevel
    session_id:  str
    goal_id:     str
    turn:        int
    timestamp:   float
    data:        Dict[str, Any]       # category-specific payload
    tags:        List[str] = field(default_factory=list)
    user_id:     Optional[str] = None
    tenant_id:   Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "event_id":  self.event_id,
            "category":  self.category.value,
            "level":     self.level.value,
            "session_id":self.session_id,
            "goal_id":   self.goal_id,
            "turn":      self.turn,
            "timestamp": self.timestamp,
            "user_id":   self.user_id,
            "data":      self.data,
            "tags":      self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogEvent":
        return cls(
            event_id   = d["event_id"],
            category   = LogCategory(d["category"]),
            level      = LogLevel(d.get("level", "info")),
            session_id = d["session_id"],
            goal_id    = d.get("goal_id", ""),
            turn       = d.get("turn", 0),
            timestamp  = d.get("timestamp", time.time()),
            data       = d.get("data", {}),
            tags       = d.get("tags", []),
            user_id    = d.get("user_id"),
        )


def make_event(
    category:   LogCategory,
    session_id: str,
    goal_id:    str,
    data:       Dict[str, Any],
    turn:       int             = 0,
    level:      LogLevel        = LogLevel.INFO,
    user_id:    Optional[str]   = None,
    tags:       Optional[List[str]] = None,
) -> LogEvent:
    """Factory — the one place where event IDs are generated."""
    return LogEvent(
        event_id   = str(uuid.uuid4())[:16],
        category   = category,
        level      = level,
        session_id = session_id,
        goal_id    = goal_id,
        turn       = turn,
        timestamp  = time.time(),
        data       = data,
        tags       = tags or [],
        user_id    = user_id,
    )


# ─── Typed data builders for each category ───────────────────────────────────

def conversation_event(
    session_id: str, goal_id: str, turn: int,
    role:       str,   # "user" | "assistant"
    text:       str,
    language:   str    = "en",
    latency_ms: int    = 0,
    **kw,
) -> LogEvent:
    return make_event(
        LogCategory.CONVERSATION, session_id, goal_id, turn=turn,
        data={
            "role":       role,
            "text_chars": len(text),
            "text_snippet": text[:120],
            "language":   language,
            "latency_ms": latency_ms,
        }, **kw,
    )


def extraction_event(
    session_id: str, goal_id: str, turn: int,
    field_name: str,
    value:      Any,
    confidence: float,
    success:    bool,
    model:      str = "",
    **kw,
) -> LogEvent:
    level = LogLevel.INFO if success else LogLevel.WARNING
    return make_event(
        LogCategory.EXTRACTION, session_id, goal_id, turn=turn, level=level,
        data={
            "field":      field_name,
            "value":      str(value)[:100] if value is not None else None,
            "confidence": round(confidence, 3),
            "success":    success,
            "model":      model,
        }, **kw,
    )


def emotion_event(
    session_id: str, goal_id: str, turn: int,
    emotion:    str,
    valence:    float,   # -1..1
    arousal:    float,   # 0..1
    shifted:    bool     = False,
    **kw,
) -> LogEvent:
    level = LogLevel.WARNING if emotion in ("distressed", "angry", "frustrated") else LogLevel.INFO
    return make_event(
        LogCategory.EMOTION, session_id, goal_id, turn=turn, level=level,
        data={
            "emotion": emotion,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "shifted": shifted,
        }, **kw,
    )


def conflict_event(
    session_id:   str, goal_id: str, turn: int,
    field_name:   str,
    prev_value:   Any,
    new_value:    Any,
    conflict_type: str,
    severity:     str,
    **kw,
) -> LogEvent:
    level = LogLevel.WARNING if severity in ("HIGH", "CRITICAL") else LogLevel.INFO
    return make_event(
        LogCategory.CONFLICT, session_id, goal_id, turn=turn, level=level,
        data={
            "field":        field_name,
            "prev_value":   str(prev_value)[:80],
            "new_value":    str(new_value)[:80],
            "conflict_type":conflict_type,
            "severity":     severity,
        }, **kw,
    )


def cost_event(
    session_id:    str, goal_id: str, turn: int,
    model:         str,
    task_type:     str,
    input_tokens:  int,
    output_tokens: int,
    cost_usd:      float,
    latency_ms:    int,
    **kw,
) -> LogEvent:
    return make_event(
        LogCategory.COST, session_id, goal_id, turn=turn,
        data={
            "model":         model,
            "task_type":     task_type,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  input_tokens + output_tokens,
            "cost_usd":      round(cost_usd, 8),
            "latency_ms":    latency_ms,
        }, **kw,
    )


def hallucination_event(
    session_id:    str, goal_id: str, turn: int,
    verdict:       str,   # CLEAN | FLAGGED | BLOCKED
    claims_total:  int,
    claims_blocked: int,
    output_snippet: str = "",
    **kw,
) -> LogEvent:
    level = {
        "CLEAN":   LogLevel.INFO,
        "FLAGGED": LogLevel.WARNING,
        "BLOCKED": LogLevel.ERROR,
    }.get(verdict, LogLevel.INFO)
    return make_event(
        LogCategory.HALLUCINATION, session_id, goal_id, turn=turn, level=level,
        data={
            "verdict":         verdict,
            "claims_total":    claims_total,
            "claims_blocked":  claims_blocked,
            "output_snippet":  output_snippet[:120],
        }, **kw,
    )


def compliance_event(
    session_id: str, goal_id: str, turn: int,
    action:     str,   # "consent_granted" | "pii_detected" | "erasure_requested" | etc.
    framework:  str,   # "dpdp" | "gdpr"
    user_id:    Optional[str] = None,
    details:    Optional[dict] = None,
    **kw,
) -> LogEvent:
    return make_event(
        LogCategory.COMPLIANCE, session_id, goal_id, turn=turn,
        user_id=user_id,
        data={
            "action":    action,
            "framework": framework,
            "details":   details or {},
        }, **kw,
    )