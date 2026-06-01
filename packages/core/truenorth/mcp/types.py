"""truenorth/mcp/types.py"""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR   = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"

@dataclass
class Tool:
    name:         str
    description:  str
    server_name:  str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    builtin:      bool           = False
    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "server_name": self.server_name, "input_schema": self.input_schema,
                "builtin": self.builtin}

@dataclass
class ToolCall:
    call_id:    str
    tool_name:  str
    arguments:  Dict[str, Any] = field(default_factory=dict)
    session_id: str            = ""
    turn:       int            = 0
    created_at: float          = field(default_factory=time.time)
    @classmethod
    def create(cls, tool_name: str, arguments: Dict[str, Any],
               session_id: str = "", turn: int = 0) -> "ToolCall":
        return cls(call_id=str(uuid.uuid4())[:12], tool_name=tool_name,
                   arguments=arguments, session_id=session_id, turn=turn)
    def to_dict(self) -> dict:
        return {"call_id": self.call_id, "tool_name": self.tool_name,
                "arguments": self.arguments, "session_id": self.session_id, "turn": self.turn}

@dataclass
class ToolResult:
    call_id:    str
    tool_name:  str
    status:     ToolResultStatus
    content:    Optional[Any] = None
    error:      Optional[str] = None
    latency_ms: int           = 0
    created_at: float         = field(default_factory=time.time)
    @property
    def is_success(self) -> bool:
        return self.status == ToolResultStatus.SUCCESS
    @property
    def result_text(self) -> str:
        if self.content is None:
            return self.error or ""
        if isinstance(self.content, str):
            return self.content
        import json
        return json.dumps(self.content, default=str)
    def to_dict(self) -> dict:
        return {"call_id": self.call_id, "tool_name": self.tool_name,
                "status": self.status.value, "content": self.content,
                "error": self.error, "latency_ms": self.latency_ms}
