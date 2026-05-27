"""Anthropic Claude client."""

from __future__ import annotations
import anthropic
from truenorth.llm.base import BaseLLMClient, LLMResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str, system: str = "", model: str = "",
                       temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        model = model or "claude-3-5-haiku-20241022"
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system

        msg = await self._client.messages.create(**kwargs)
        return LLMResponse(
            content=msg.content[0].text,
            model=model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
