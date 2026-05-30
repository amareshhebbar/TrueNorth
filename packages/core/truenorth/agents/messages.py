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


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"   # routes tasks, coordinates
    EXTRACTOR    = "extractor"      # field extraction specialist
    VALIDATOR    = "validator"      # validates extracted values
    RESEARCHER   = "researcher"     # web search / tool calls
    WRITER       = "writer"         # output generation specialist
    SUPERVISOR   = "supervisor"     # quality control, escalation
    CUSTOM       = "custom"         # user-defined specialist


class MessageType(str, Enum):
    TASK_ASSIGN     = "task_assign"      # assign a task to an agent
    TASK_CANCEL     = "task_cancel"      # cancel an in-progress task
    CONTEXT_UPDATE  = "context_update"   # push new context to agent

    TASK_RESULT     = "task_result"      # task completed successfully
    TASK_FAILED     = "task_failed"      # task failed, needs rerouting
    TASK_BLOCKED    = "task_blocked"     # task blocked, needs human input
    PARTIAL_RESULT  = "partial_result"   # intermediate result (streaming)

    REVIEW_REQUEST  = "review_request"   # request supervisor review
    REVIEW_VERDICT  = "review_verdict"   # supervisor's verdict


class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    BLOCKED    = "blocked"
    CANCELLED  = "cancelled"
    REVIEWING  = "reviewing"


class Priority(str, Enum):
    CRITICAL = "critical"   # must complete before next user turn
    HIGH     = "high"       # complete this turn
    NORMAL   = "normal"     # complete within 2 turns
    LOW      = "low"        # best-effort, can be deferred


# ─────────────────────────────────────────────────────────────────────────────
#  Core message types
# ─────────────────────────────────────────────────────────────────────────────

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
    task:         str               # human-readable task description
    payload:      Dict[str, Any]    # task-specific data
    session_id:   str = ""
    turn:         int = 0
    priority:     Priority = Priority.NORMAL
    parent_id:    Optional[str] = None    # if this is a sub-task
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
    result:        Any              # the actual output (str, dict, etc.)
    confidence:    float = 1.0      # 0–1 confidence in this result
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
    retry:       bool = False   # should the agent retry?
    escalate:    bool = False   # escalate to human?
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