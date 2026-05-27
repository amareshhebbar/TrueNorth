"""OpenAI client."""
from __future__ import annotations
import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI
from truenorth.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000, temperature: float = 0.7) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, system: str = "", max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, stream=True,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
