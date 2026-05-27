"""Central state object passed through every node of the conversation graph."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FieldValue:
    """A collected field value with metadata."""
    value: Any
    confidence: float               # 0.0 to 1.0
    source: str                     # user_stated | inferred | calculated | document
    raw_text: str                   # Original user text
    timestamp: datetime = field(default_factory=datetime.utcnow)
    privacy_level: str = "low"      # low | medium | high | critical


@dataclass
class ConversationTurn:
    role: str                       # user | assistant
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens_used: int = 0


@dataclass
class GraphState:
    """
    The complete state of one TrueNorth session.
    Passed through every processing node. Immutable pattern — nodes
    return new state rather than mutating in place.
    """
    session_id: str
    goal_id: str
    config: dict                                        # Parsed YAML goal config

    # Collected data
    profile: dict[str, FieldValue] = field(default_factory=dict)

    # Conversation
    conversation: list[ConversationTurn] = field(default_factory=list)

    # Intelligence signals
    emotion_state: str = "neutral"                      # engaged|neutral|confused|frustrated|anxious|distressed
    emotion_confidence: float = 0.0
    conversation_quality_score: float = 1.0
    abandonment_risk: float = 0.0

    # Progress tracking
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    current_field_target: str | None = None
    skipped_fields: list[str] = field(default_factory=list)

    # Output
    output: dict | None = None
    output_version: int = 0

    # Session lifecycle
    completed: bool = False
    escalated: bool = False
    escalation_reason: str | None = None
    resumed: bool = False

    # Cost tracking
    cost_usd: float = 0.0
    tokens_used: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    user_id: str | None = None

    def add_turn(self, role: str, content: str, tokens: int = 0) -> "GraphState":
        """Return new state with turn appended."""
        import copy
        new = copy.deepcopy(self)
        new.conversation.append(ConversationTurn(role=role, content=content, tokens_used=tokens))
        new.updated_at = datetime.utcnow()
        return new

    def set_field(self, name: str, value: FieldValue) -> "GraphState":
        """Return new state with field set."""
        import copy
        new = copy.deepcopy(self)
        new.profile[name] = value
        if name in new.missing_required:
            new.missing_required.remove(name)
        if name in new.missing_optional:
            new.missing_optional.remove(name)
        new.updated_at = datetime.utcnow()
        return new

    @property
    def collected_fields(self) -> dict:
        """Plain dict of field_name → value for prompts."""
        return {k: v.value for k, v in self.profile.items()}

    @property
    def is_ready_for_output(self) -> bool:
        return len(self.missing_required) == 0
