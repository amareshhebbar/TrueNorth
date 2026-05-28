"""
truenorth/core/graph_state.py

Single source of truth for one conversation session.
Passed through every stage of the pipeline each turn.
Has zero imports from the rest of the truenorth package (prevents circular imports).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class GraphState:
    """
    Mutable session state. Every pipeline component reads from and writes to this.

    Lifecycle:
      1. Created by SessionManager.create() with goal config loaded from YAML
      2. Passed into engine.process_message() each turn
      3. Serialized to dict by .to_dict() for storage
      4. Restored by .from_dict() on session resume
    """

    # ------------------------------------------------------------------ identity
    session_id:   str = ""
    goal_id:      str = ""
    user_id:      Optional[str] = None
    tenant_id:    Optional[str] = None

    # ------------------------------------------------------------------ goal config (from YAML)
    goal_config:      Dict[str, Any] = field(default_factory=dict)
    fields_config:    Dict[str, Any] = field(default_factory=dict)  # field_name → field spec
    persona:          Dict[str, Any] = field(default_factory=dict)  # name, tone, language
    output_template:  str = ""

    # ------------------------------------------------------------------ collection state
    collected_fields:     Dict[str, Any] = field(default_factory=dict)   # field_name → value
    field_confidences:    Dict[str, float] = field(default_factory=dict) # field_name → 0-1
    skipped_fields:       Set[str]  = field(default_factory=set)
    asked_optional_fields: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ conversation
    turn_history:   List[Dict[str, Any]] = field(default_factory=list)
    current_turn:   int = 0
    current_input:  str = ""     # raw user message this turn
    current_output: str = ""     # agent response this turn

    # ------------------------------------------------------------------ intelligence signals
    current_emotion:   Optional[Dict[str, Any]] = None  # {label, score, raw}
    active_conflicts:  List[Dict[str, Any]] = field(default_factory=list)
    last_extraction:   Optional[Dict[str, Any]] = None  # {field, value, confidence}
    detected_language: str = "en"
    is_romanized:      bool = False
    quality_reports:   List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ cost / budget
    total_cost_usd:  float = 0.0
    cost_budget_usd: Optional[float] = None
    token_counts:    Dict[str, int] = field(default_factory=dict)  # model → total_tokens

    # ------------------------------------------------------------------ session flags
    is_complete:  bool = False
    is_resumed:   bool = False
    final_output: Optional[Dict[str, Any]] = None
    error:        Optional[str] = None
    created_at:   float = field(default_factory=time.time)
    updated_at:   float = field(default_factory=time.time)

    # ------------------------------------------------------------------ helpers

    def add_turn(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        """Append a turn to the conversation history."""
        self.turn_history.append({
            "role":      role,
            "content":   content,
            "turn":      self.current_turn,
            "timestamp": time.time(),
            **(metadata or {}),
        })

    def set_field(self, name: str, value: Any, confidence: float = 1.0) -> None:
        """Store a collected field value with its confidence score."""
        self.collected_fields[name] = value
        self.field_confidences[name] = round(confidence, 4)
        self.updated_at = time.time()

    def get_field(self, name: str, default: Any = None) -> Any:
        return self.collected_fields.get(name, default)

    @property
    def required_fields(self) -> Dict[str, Any]:
        return {k: v for k, v in self.fields_config.items() if v.get("required", False)}

    @property
    def missing_required(self) -> List[str]:
        return [
            name for name in self.required_fields
            if name not in self.collected_fields and name not in self.skipped_fields
        ]

    @property
    def completion_pct(self) -> float:
        req = len(self.required_fields)
        if req == 0:
            return 1.0
        collected = sum(1 for f in self.required_fields if f in self.collected_fields)
        return round(collected / req, 3)

    @property
    def user_messages(self) -> List[str]:
        return [t["content"] for t in self.turn_history if t.get("role") == "user"]

    @property
    def agent_messages(self) -> List[str]:
        return [t["content"] for t in self.turn_history if t.get("role") == "assistant"]

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict:
        return {
            "session_id":            self.session_id,
            "goal_id":               self.goal_id,
            "user_id":               self.user_id,
            "tenant_id":             self.tenant_id,
            "goal_config":           self.goal_config,
            "fields_config":         self.fields_config,
            "persona":               self.persona,
            "output_template":       self.output_template,
            "collected_fields":      self.collected_fields,
            "field_confidences":     self.field_confidences,
            "skipped_fields":        list(self.skipped_fields),
            "asked_optional_fields": list(self.asked_optional_fields),
            "turn_history":          self.turn_history,
            "current_turn":          self.current_turn,
            "current_emotion":       self.current_emotion,
            "active_conflicts":      self.active_conflicts,
            "last_extraction":       self.last_extraction,
            "detected_language":     self.detected_language,
            "is_romanized":          self.is_romanized,
            "quality_reports":       self.quality_reports,
            "total_cost_usd":        self.total_cost_usd,
            "cost_budget_usd":       self.cost_budget_usd,
            "token_counts":          self.token_counts,
            "is_complete":           self.is_complete,
            "is_resumed":            self.is_resumed,
            "final_output":          self.final_output,
            "error":                 self.error,
            "created_at":            self.created_at,
            "updated_at":            self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphState":
        state = cls(
            session_id            = data.get("session_id", ""),
            goal_id               = data.get("goal_id", ""),
            user_id               = data.get("user_id"),
            tenant_id             = data.get("tenant_id"),
            goal_config           = data.get("goal_config", {}),
            fields_config         = data.get("fields_config", {}),
            persona               = data.get("persona", {}),
            output_template       = data.get("output_template", ""),
            collected_fields      = data.get("collected_fields", {}),
            field_confidences     = data.get("field_confidences", {}),
            skipped_fields        = set(data.get("skipped_fields", [])),
            asked_optional_fields = set(data.get("asked_optional_fields", [])),
            turn_history          = data.get("turn_history", []),
            current_turn          = data.get("current_turn", 0),
            current_emotion       = data.get("current_emotion"),
            active_conflicts      = data.get("active_conflicts", []),
            last_extraction       = data.get("last_extraction"),
            detected_language     = data.get("detected_language", "en"),
            is_romanized          = data.get("is_romanized", False),
            quality_reports       = data.get("quality_reports", []),
            total_cost_usd        = data.get("total_cost_usd", 0.0),
            cost_budget_usd       = data.get("cost_budget_usd"),
            token_counts          = data.get("token_counts", {}),
            is_complete           = data.get("is_complete", False),
            is_resumed            = data.get("is_resumed", False),
            final_output          = data.get("final_output"),
            error                 = data.get("error"),
            created_at            = data.get("created_at", time.time()),
            updated_at            = data.get("updated_at", time.time()),
        )
        return state

    @classmethod
    def from_goal_config(cls, goal_config: dict, session_id: str, **kwargs) -> "GraphState":
        """
        Construct a fresh GraphState from a loaded goal YAML config.
        """
        fields_cfg = {}
        for f in goal_config.get("fields", []):
            fields_cfg[f["name"]] = f

        return cls(
            session_id      = session_id,
            goal_id         = goal_config.get("id", ""),
            goal_config     = goal_config,
            fields_config   = fields_cfg,
            persona         = goal_config.get("persona", {}),
            output_template = goal_config.get("output", {}).get("template", ""),
            cost_budget_usd = goal_config.get("budget", {}).get("max_cost_usd"),
            **kwargs,
        )