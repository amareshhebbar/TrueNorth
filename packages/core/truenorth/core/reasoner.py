"""BaseReasoner + DefaultReasoner."""
from __future__ import annotations
from abc import ABC, abstractmethod
from truenorth.core.graph_state import GraphState
from truenorth.core.yaml_loader import GoalConfig
from truenorth.llm.router import LLMRouter


class BaseReasoner(ABC):
    @abstractmethod
    async def decide_next_question(self, state: GraphState, missing: list[str]) -> str: ...
    @abstractmethod
    async def generate_output(self, state: GraphState) -> dict: ...


class DefaultReasoner(BaseReasoner):
    def __init__(self, llm_router: LLMRouter, config: GoalConfig):
        self.llm = llm_router
        self.config = config

    async def decide_next_question(self, state: GraphState, missing: list[str]) -> str:
        for field in self.config.required_fields:
            if field.name in missing:
                return field.name
        for field in self.config.optional_fields:
            if field.name in missing:
                return field.name
        return missing[0] if missing else ""

    async def generate_output(self, state: GraphState) -> dict:
        return {"profile": state.get_profile_values(), "session_id": state.session_id}
