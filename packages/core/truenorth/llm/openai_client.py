from __future__ import annotations

import logging
import os
from typing import AsyncIterator, List, Optional

from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk, _Timer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

class OpenAIClient(LLMBase):
    """
    OpenAI GPT adapter.

    Config keys:
      api_key  : str  — OPENAI_API_KEY
      model    : str  — default gpt-4o-mini
      base_url : str  — override for OpenAI-compatible endpoints (Groq, Together, Ollama)
    """

    supports_streaming: bool = True
    max_context_tokens: int  = 128_000

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", _DEFAULT_MODEL)
        self._api_key   = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self._base_url  = self.config.get("base_url")
        self._client    = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                kwargs = {"api_key": self._api_key or None}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = openai.AsyncOpenAI(**kwargs)
            except ImportError as e:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                ) from e
        return self._client

    async def generate(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_client()
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        with _Timer() as t:
            try:
                resp = await client.chat.completions.create(
                    model       = self.model_name,
                    messages    = api_messages,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    **kwargs,
                )
            except Exception as e:
                logger.error("openai generate error: %s", e)
                raise

        content = resp.choices[0].message.content or ""
        usage   = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = getattr(usage, "prompt_tokens", 0),
            output_tokens = getattr(usage, "completion_tokens", 0),
            latency_ms    = t.ms,
            raw           = resp,
        )

    async def generate_stream(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        try:
            stream = await client.chat.completions.create(
                model       = self.model_name,
                messages    = api_messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                stream      = True,
                **kwargs,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                if delta:
                    yield StreamChunk(delta=delta)
            yield StreamChunk(delta="", is_final=True)
        except Exception as e:
            logger.error("openai stream error: %s", e)
            raise
