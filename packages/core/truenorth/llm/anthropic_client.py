"""Anthropic Claude client."""
from __future__ import annotations
import logging
from typing import AsyncGenerator
import anthropic
from truenorth.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str | None = None):
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=messages,
        )
        return response.content[0].text

    async def stream(self, prompt: str, system: str = "", max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
