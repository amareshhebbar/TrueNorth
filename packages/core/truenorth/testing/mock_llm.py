"""
Deterministic mock LLM for tests and dry-runs.
Zero API calls. Reads responses from scenario files.
"""

from __future__ import annotations
import json
from pathlib import Path
from truenorth.llm.base import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """
    Returns scripted responses from a scenario JSON file.
    Falls back to smart default responses based on task type.
    """

    def __init__(self, scenario_path: str | None = None):
        self._scenario: list[dict] = []
        self._call_count = 0

        if scenario_path:
            path = Path(scenario_path)
            if path.exists():
                data = json.loads(path.read_text())
                self._scenario = data.get("llm_responses", [])

    async def complete(self, prompt: str, system: str = "", model: str = "",
                       temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        # Use scripted response if available
        if self._call_count < len(self._scenario):
            content = self._scenario[self._call_count].get("content", "")
            self._call_count += 1
        else:
            content = self._default_response(prompt, system)

        return LLMResponse(
            content=content, model="mock",
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()),
        )

    def _default_response(self, prompt: str, system: str) -> str:
        """Smart defaults based on what the prompt is asking for."""
        prompt_lower = prompt.lower()

        # Field extraction
        if "extract" in prompt_lower and "json" in (system or "").lower():
            return json.dumps({"extracted": {}, "user_is_correcting": False})

        # Emotion detection
        if "emotional state" in prompt_lower or "emotion" in prompt_lower:
            return json.dumps({
                "state": "neutral", "confidence": 0.8,
                "reasoning": "mock", "agent_adaptation": {}
            })

        # Conflict detection
        if "contradict" in prompt_lower or "conflict" in prompt_lower:
            return json.dumps({"conflict_type": "no_conflict"})

        # Conversation / question planning
        if "next field" in prompt_lower or "what to ask" in prompt_lower:
            return json.dumps({
                "next_field": None, "message": "Could you tell me more?",
                "is_complete": False, "reasoning": "mock"
            })

        # Welcome / completion messages
        return "Hello! I'm here to help. Tell me about yourself."
