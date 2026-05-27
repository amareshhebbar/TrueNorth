"""Google Gemini client."""
from __future__ import annotations
import logging
from typing import AsyncGenerator
import google.generativeai as genai
from truenorth.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-2.0-flash-lite", api_key: str | None = None):
        self.model = model
        if api_key:
            genai.configure(api_key=api_key)
        self._genai_model = genai.GenerativeModel(model)

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000, temperature: float = 0.7) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = await self._genai_model.generate_content_async(
            full_prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return response.text

    async def stream(self, prompt: str, system: str = "", max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        async for chunk in await self._genai_model.generate_content_async(
            full_prompt,
            generation_config={"max_output_tokens": max_tokens},
            stream=True,
        ):
            if chunk.text:
                yield chunk.text
