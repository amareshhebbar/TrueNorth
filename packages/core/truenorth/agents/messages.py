"""
Inter-agent message types for TrueNorth's multi-agent system.

Every message that flows between agents is a typed dataclass with
a unique ID, routing info, and a payload. This makes the system
debuggable (full audit trail), testable (deterministic serialisation),
and extensible (add new message types without breaking old agents).

Message lifecycle:
    Orchestrator  →  [AgentMessage]  →  SpecialistAgent
    SpecialistAgent  →  [AgentResponse]  →  Orchestrator
    Orchestrator  →  [SupervisorMessage]  →  Supervisor
    Supervisor  →  [SupervisorVerdict]  →  Orchestrator

Works for any sector — healthcare, legal, HR, finance, fitness.
The message schema is domain-agnostic; the payload carries domain data.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    EXTRACTOR    = "extractor"
    VALIDATOR    = "validator"
    RESEARCHER   = "researcher"
    WRITER       = "writer"
    SUPERVISOR   = "supervisor"
    CUSTOM       = "custom"

class MessageType(str, Enum):
    TASK_ASSIGN     = "task_assign"
    TASK_CANCEL     = "task_cancel"
    CONTEXT_UPDATE  = "context_update"

    TASK_RESULT     = "task_result"
    TASK_FAILED     = "task_failed"
    TASK_BLOCKED    = "task_blocked"
    PARTIAL_RESULT  = "partial_result"

    REVIEW_REQUEST  = "review_request"
    REVIEW_VERDICT  = "review_verdict"

class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    BLOCKED    = "blocked"
    CANCELLED  = "cancelled"
    REVIEWING  = "reviewing"

class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    NORMAL   = "normal"
    LOW      = "low"

@dataclass
class AgentMessage:
    """
    A task assignment from the Orchestrator to a SpecialistAgent.

    The orchestrator creates one of these for each unit of work and
    delivers it to the appropriate agent. The agent processes it and
    returns an AgentResponse.
    """
    message_id:   str
    sender:       str
    recipient:    str
    message_type: MessageType
    task:         str
    payload:      Dict[str, Any]
    session_id:   str = ""
    turn:         int = 0
    priority:     Priority = Priority.NORMAL
    parent_id:    Optional[str] = None
    timeout_s:    float = 30.0
    created_at:   float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        sender:    str,
        recipient: str,
        task:      str,
        payload:   Dict[str, Any],
        msg_type:  MessageType = MessageType.TASK_ASSIGN,
        **kwargs,
    ) -> "AgentMessage":
        return cls(
            message_id   = str(uuid.uuid4())[:12],
            sender       = sender,
            recipient    = recipient,
            message_type = msg_type,
            task         = task,
            payload      = payload,
            **kwargs,
        )

    def to_dict(self) -> dict:
        return {
            "message_id":   self.message_id,
            "sender":       self.sender,
            "recipient":    self.recipient,
            "type":         self.message_type.value,
            "task":         self.task,
            "session_id":   self.session_id,
            "turn":         self.turn,
            "priority":     self.priority.value,
            "created_at":   self.created_at,
        }

@dataclass
class AgentResponse:
    """
    Result returned by a SpecialistAgent to the Orchestrator.
    """
    message_id:    str
    agent_id:      str
    status:        TaskStatus
    result:        Any
    confidence:    float = 1.0
    error:         Optional[str] = None
    metadata:      Dict[str, Any] = field(default_factory=dict)
    latency_ms:    int = 0
    tokens_used:   int = 0
    completed_at:  float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def result_text(self) -> str:
        if isinstance(self.result, str):
            return self.result
        if isinstance(self.result, dict):
            import json
            return json.dumps(self.result, default=str)
        return str(self.result) if self.result is not None else ""

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "agent_id":   self.agent_id,
            "status":     self.status.value,
            "confidence": round(self.confidence, 3),
            "error":      self.error,
            "latency_ms": self.latency_ms,
            "tokens_used":self.tokens_used,
        }

@dataclass
class SupervisorVerdict:
    """
    Supervisor's quality verdict on an agent's result.
    """
    message_id:  str
    agent_id:    str
    approved:    bool
    score:       float
    feedback:    str
    retry:       bool = False
    escalate:    bool = False
    issues:      List[str] = field(default_factory=list)
    created_at:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "agent_id":   self.agent_id,
            "approved":   self.approved,
            "score":      round(self.score, 3),
            "feedback":   self.feedback,
            "retry":      self.retry,
            "escalate":   self.escalate,
            "issues":     self.issues,
        }
