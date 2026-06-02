"""
Local LLM client — connects to Ollama (or any OpenAI-compatible local server).

Supports:
  - Ollama  : ollama serve  → http://localhost:11434
  - llama.cpp server        → http://localhost:8080
  - LM Studio               → http://localhost:1234
  - Any OpenAI-compat API   → set base_url in config

Zero cloud cost. Works fully offline. Ideal for:
  - Self-hosted deployments
  - Privacy-sensitive goals (medical, financial)
  - Development without API keys

Config keys:
  base_url  : str  — default http://localhost:11434 (Ollama)
  model     : str  — e.g. "llama3.1", "mistral", "phi3"
  provider  : str  — "ollama" | "llamacpp" | "lmstudio" | "openai_compat"
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator, List, Optional

from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk, _Timer

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS = {
    "ollama":        ("http://localhost:11434",  "/api/chat"),
    "llamacpp":      ("http://localhost:8080",   "/chat/completions"),
    "lmstudio":      ("http://localhost:1234",   "/chat/completions"),
    "openai_compat": ("http://localhost:8000",   "/chat/completions"),
}


class LocalLLMClient(LLMBase):
    """
    Local LLM via Ollama or OpenAI-compatible HTTP API.

    Usage (Ollama):
        client = LocalLLMClient(config={"model": "llama3.1", "provider": "ollama"})
        response = await client.generate([Message(role="user", content="Hello")])

    Usage (llama.cpp):
        client = LocalLLMClient(config={
            "model":    "mistral-7b",
            "provider": "llamacpp",
            "base_url": "http://localhost:8080",
        })
    """

    supports_streaming: bool = True
    max_context_tokens: int  = 32_768

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._provider  = self.config.get("provider", "ollama")
        self.model_name = self.config.get("model", "llama3.1")

        default_url, _ = _PROVIDER_DEFAULTS.get(self._provider, ("http://localhost:11434", ""))
        self._base_url  = (
            self.config.get("base_url")
            or os.environ.get("LOCAL_LLM_URL")
            or default_url
        )
        self._client    = None   # lazy: httpx.AsyncClient

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url = self._base_url,
                    timeout  = httpx.Timeout(120.0),
                )
            except ImportError as e:
                raise RuntimeError("httpx not installed. Run: pip install httpx") from e
        return self._client

    async def generate(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if self._provider == "ollama":
            return await self._ollama_generate(messages, system, max_tokens, temperature)
        return await self._openai_compat_generate(messages, system, max_tokens, temperature)

    async def generate_stream(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        if self._provider == "ollama":
            async for chunk in self._ollama_stream(messages, system, max_tokens, temperature):
                yield chunk
        else:
            async for chunk in self._openai_compat_stream(messages, system, max_tokens, temperature):
                yield chunk

    # ------------------------------------------------------------------
    # Ollama API
    # ------------------------------------------------------------------

    async def _ollama_generate(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
    ) -> LLMResponse:
        client = self._get_client()
        payload = self._ollama_payload(messages, system, max_tokens, temperature, stream=False)
        with _Timer() as t:
            try:
                resp = await client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error("ollama generate error: %s", e)
                raise

        content    = data.get("message", {}).get("content", "")
        in_tokens  = data.get("prompt_eval_count", self._count_tokens_approx(
            self._messages_to_str(messages)
        ))
        out_tokens = data.get("eval_count", self._count_tokens_approx(content))

        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = in_tokens,
            output_tokens = out_tokens,
            latency_ms    = t.ms,
            raw           = data,
        )

    async def _ollama_stream(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        import json as _json
        client  = self._get_client()
        payload = self._ollama_payload(messages, system, max_tokens, temperature, stream=True)
        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                in_tok = out_tok = 0
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data  = _json.loads(line)
                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        yield StreamChunk(delta=delta)
                    if data.get("done"):
                        in_tok  = data.get("prompt_eval_count", 0)
                        out_tok = data.get("eval_count", 0)
                yield StreamChunk(delta="", is_final=True,
                                  input_tokens=in_tok, output_tokens=out_tok)
        except Exception as e:
            logger.error("ollama stream error: %s", e)
            raise

    def _ollama_payload(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
        stream:      bool,
    ) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        return {
            "model":   self.model_name,
            "messages": msgs,
            "stream":  stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

    # ------------------------------------------------------------------
    # OpenAI-compatible API (llama.cpp, LM Studio, etc.)
    # ------------------------------------------------------------------

    async def _openai_compat_generate(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
    ) -> LLMResponse:
        client = self._get_client()
        payload = self._openai_payload(messages, system, max_tokens, temperature, stream=False)
        with _Timer() as t:
            try:
                resp = await client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error("local openai_compat generate error: %s", e)
                raise

        choice    = data["choices"][0]
        content   = choice["message"]["content"] or ""
        usage     = data.get("usage", {})
        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
            latency_ms    = t.ms,
            raw           = data,
        )

    async def _openai_compat_stream(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        import json as _json
        client  = self._get_client()
        payload = self._openai_payload(messages, system, max_tokens, temperature, stream=True)
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data  = _json.loads(line)
                        delta = data["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            yield StreamChunk(delta=delta)
                    except (_json.JSONDecodeError, KeyError):
                        continue
            yield StreamChunk(delta="", is_final=True)
        except Exception as e:
            logger.error("local openai_compat stream error: %s", e)
            raise

    def _openai_payload(
        self,
        messages:    List[Message],
        system:      Optional[str],
        max_tokens:  int,
        temperature: float,
        stream:      bool,
    ) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += [{"role": m.role, "content": m.content} for m in messages]
        return {
            "model":       self.model_name,
            "messages":    msgs,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      stream,
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if the local LLM server is reachable."""
        client = self._get_client()
        try:
            if self._provider == "ollama":
                resp = await client.get("/api/tags")
            else:
                resp = await client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def list_ollama_models(cls, base_url: str = "http://localhost:11434") -> list[str]:
        """Return list of locally available Ollama models."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{base_url}/api/tags")
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []