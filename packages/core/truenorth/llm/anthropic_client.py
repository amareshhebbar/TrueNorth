from __future__ import annotations

import logging
import os
from typing import AsyncIterator, List, Optional

from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk, _Timer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicClient(LLMBase):
    """
    Anthropic Claude adapter.

    Config keys (all optional — fall back to env vars):
      api_key   : str   — ANTHROPIC_API_KEY
      model     : str   — default claude-sonnet-4-20250514
      max_tokens: int   — default 1024
    """

    supports_streaming: bool = True
    max_context_tokens: int  = 200_000

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", _DEFAULT_MODEL)
        self._api_key   = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client    = None  # lazy-initialised

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key or None)
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
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
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        with _Timer() as t:
            try:
                resp = await client.messages.create(
                    model       = self.model_name,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    system      = system or "",
                    messages    = api_messages,
                    **kwargs,
                )
            except Exception as e:
                logger.error("anthropic generate error: %s", e)
                raise

        content = resp.content[0].text if resp.content else ""
        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = resp.usage.input_tokens,
            output_tokens = resp.usage.output_tokens,
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
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            async with client.messages.stream(
                model       = self.model_name,
                max_tokens  = max_tokens,
                temperature = temperature,
                system      = system or "",
                messages    = api_messages,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text)
                final = await stream.get_final_message()
                yield StreamChunk(
                    delta         = "",
                    is_final      = True,
                    input_tokens  = final.usage.input_tokens,
                    output_tokens = final.usage.output_tokens,
                )
        except Exception as e:
            logger.error("anthropic stream error: %s", e)
            raise