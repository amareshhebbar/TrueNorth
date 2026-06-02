from __future__ import annotations

import logging
import os
from typing import AsyncIterator, List, Optional

from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk, _Timer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-3.5-flash"


class GeminiClient(LLMBase):
    """
    Google Gemini adapter.

    Config keys:
      api_key : str  — GEMINI_API_KEY or GOOGLE_API_KEY
      model   : str  — default gemini-3.5-flash
    """

    supports_streaming: bool = True
    max_context_tokens: int  = 1_000_000

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", _DEFAULT_MODEL)
        self._api_key   = (
            self.config.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        self._model_obj = None

    def _get_model(self):
        if self._model_obj is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key or None)
                self._model_obj = genai.GenerativeModel(self.model_name)
            except ImportError as e:
                raise RuntimeError(
                    "google-generativeai not installed. Run: pip install google-generativeai"
                ) from e
        return self._model_obj

    def _build_contents(self, messages: List[Message], system: Optional[str]) -> list:
        """Convert TrueNorth messages to Gemini content format."""
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System] {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return contents

    async def generate(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        import asyncio
        model = self._get_model()
        contents = self._build_contents(messages, system)

        gen_config = {"max_output_tokens": max_tokens, "temperature": temperature}

        with _Timer() as t:
            try:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(contents, generation_config=gen_config),
                )
            except Exception as e:
                logger.error("gemini generate error: %s", e)
                raise

        content = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        in_tok  = getattr(meta, "prompt_token_count", self._count_tokens_approx(
            " ".join(m.content for m in messages)
        ))
        out_tok = getattr(meta, "candidates_token_count", self._count_tokens_approx(content))

        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = in_tok,
            output_tokens = out_tok,
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
        import asyncio
        model = self._get_model()
        contents = self._build_contents(messages, system)
        gen_config = {"max_output_tokens": max_tokens, "temperature": temperature}

        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    contents,
                    generation_config=gen_config,
                    stream=True,
                ),
            )
            for chunk in resp:
                text = chunk.text or ""
                if text:
                    yield StreamChunk(delta=text)
            yield StreamChunk(delta="", is_final=True)
        except Exception as e:
            logger.error("gemini stream error: %s", e)
            raise