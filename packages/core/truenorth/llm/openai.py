from __future__ import annotations
from openai import AsyncOpenAI
from truenorth.llm.base import BaseLLMClient, LLMResponse

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str, system: str = "", model: str = "",
                       temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        model = model or "gpt-4o-mini"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )
