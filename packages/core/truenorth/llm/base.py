"""Abstract base for all LLM provider clients."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        from truenorth.llm.pricing import get_cost
        return get_cost(self.model, self.input_tokens, self.output_tokens)


class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, system: str = "", model: str = "",
                       temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        pass

    async def complete_json(self, prompt: str, system: str = "",
                             model: str = "", **kwargs) -> dict:
        """Complete and parse JSON response."""
        import json, re
        response = await self.complete(
            prompt=prompt,
            system=system + "\nRespond ONLY with valid JSON. No markdown, no explanation.",
            model=model,
            temperature=0.1,
            **kwargs
        )
        text = response.content.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"```$", "", text).strip()
        return json.loads(text), response
