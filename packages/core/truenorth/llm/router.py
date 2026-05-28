"""
truenorth/llm/router.py

Multi-model router. Dispatches each task type to the cheapest/best model:
  extract   → Gemini Flash   (cheap, fast, good at structured output)
  converse  → Claude Haiku   (natural language, empathetic)
  output    → Claude Sonnet  (highest quality final report)
  classify  → Gemini Flash   (fast classification tasks)

The routing table is fully config-driven — override any assignment via
goal YAML or environment variables.

Usage:
    router = LLMRouter.from_env()
    response = await router.generate(task="converse", messages=[...], system="...")
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Dict, List, Optional

from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk

logger = logging.getLogger(__name__)

TASK_EXTRACT  = "extract"
TASK_CONVERSE = "converse"
TASK_OUTPUT   = "output"
TASK_CLASSIFY = "classify"
TASK_EMBED    = "embed"
TASK_OTHER    = "other"


# ---------------------------------------------------------------------------
# Default routing table (model strings)
# ---------------------------------------------------------------------------

_DEFAULT_ROUTING: Dict[str, str] = {
    TASK_EXTRACT:  "gemini-1.5-flash",
    TASK_CONVERSE: "claude-haiku-4-5-20251001",
    TASK_OUTPUT:   "claude-sonnet-4-20250514",
    TASK_CLASSIFY: "gemini-1.5-flash",
    TASK_OTHER:    "claude-haiku-4-5-20251001",
}

_MODEL_PROVIDER: Dict[str, str] = {
    "claude":  "anthropic",
    "gpt":     "openai",
    "gemini":  "gemini",
    "ollama":  "local",
    "local":   "local",
    "llama":   "local",
    "mistral": "local",
}


class RouterError(Exception):
    """Raised when routing or provider initialisation fails."""


class LLMRouter:
    """
    Routes LLM calls to the appropriate provider+model for each task type.

    Supports:
      - Per-task model assignment (routing table)
      - Fallback chain (primary → fallback → emergency)
      - Lazy client initialisation (providers only instantiated when first used)
    """

    def __init__(
        self,
        routing:  Optional[Dict[str, str]] = None,
        clients:  Optional[Dict[str, LLMBase]] = None,
        config:   Optional[dict] = None,
    ):
        """
        Args:
            routing : {task_type: model_string} overrides.
            clients : Pre-built {provider: LLMBase} instances (for testing).
            config  : Full router config dict.
        """
        self._config   = config or {}
        self._routing  = {**_DEFAULT_ROUTING, **(routing or {})}
        self._clients: Dict[str, LLMBase] = clients or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, extra_config: Optional[dict] = None) -> "LLMRouter":
        """
        Build a router from environment variables.
        Each task type can be overridden via env var:
          TRUENORTH_MODEL_EXTRACT=gemini-1.5-flash
          TRUENORTH_MODEL_CONVERSE=claude-haiku-4-5-20251001
          TRUENORTH_MODEL_OUTPUT=claude-sonnet-4-20250514
        """
        routing = dict(_DEFAULT_ROUTING)
        for task in (TASK_EXTRACT, TASK_CONVERSE, TASK_OUTPUT, TASK_CLASSIFY):
            env_key = f"TRUENORTH_MODEL_{task.upper()}"
            if env_val := os.environ.get(env_key):
                routing[task] = env_val
                logger.info("router: %s → %s (from env)", task, env_val)

        return cls(routing=routing, config=extra_config or {})

    @classmethod
    def from_config(cls, config: dict) -> "LLMRouter":
        """Build a router from a goal YAML `llm:` section."""
        routing = dict(_DEFAULT_ROUTING)
        routing.update(config.get("routing", {}))
        return cls(routing=routing, config=config)

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def generate(
        self,
        task:        str,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        model:       Optional[str] = None,   
        **kwargs,
    ) -> LLMResponse:
        """
        Route a generation request to the appropriate provider.

        Args:
            task:       Task type (use TASK_* constants)
            messages:   Conversation history
            system:     System prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            model:      Override model (bypasses routing table)
        """
        target_model = model or self._routing.get(task, self._routing[TASK_OTHER])
        client = self._get_client(target_model)

        logger.debug("router: task=%s model=%s", task, target_model)

        try:
            return await client.generate(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            logger.error("router: %s failed with %s: %s", target_model, type(e).__name__, e)
            # Try fallback if available
            fallback = self._config.get("fallback_model")
            if fallback and fallback != target_model:
                logger.warning("router: falling back to %s", fallback)
                fallback_client = self._get_client(fallback)
                return await fallback_client.generate(
                    messages=messages, system=system,
                    max_tokens=max_tokens, temperature=temperature,
                )
            raise

    async def generate_stream(
        self,
        task:        str,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        model:       Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming version of generate()."""
        target_model = model or self._routing.get(task, self._routing[TASK_OTHER])
        client = self._get_client(target_model)
        logger.debug("router: stream task=%s model=%s", task, target_model)
        async for chunk in client.generate_stream(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_client(self, model: str) -> LLMBase:
        """Return (or create) the provider client for a model string."""
        if model in self._clients:
            return self._clients[model]

        provider = self._detect_provider(model)
        client   = self._build_client(provider, model)
        self._clients[model] = client
        return client

    @staticmethod
    def _detect_provider(model: str) -> str:
        for prefix, provider in _MODEL_PROVIDER.items():
            if model.startswith(prefix):
                return provider
        return "anthropic"  # safe default

    def _build_client(self, provider: str, model: str) -> LLMBase:
        """Lazily instantiate a provider client."""
        model_config = {"model": model, **self._config.get("provider_config", {}).get(provider, {})}

        if provider == "anthropic":
            from truenorth.llm.anthropic_client import AnthropicClient
            return AnthropicClient(config=model_config)

        if provider == "openai":
            from truenorth.llm.openai_client import OpenAIClient
            return OpenAIClient(config=model_config)

        if provider == "gemini":
            from truenorth.llm.gemini_client import GeminiClient
            return GeminiClient(config=model_config)

        if provider == "local":
            from truenorth.llm.local_llm import LocalLLMClient  # Phase 2 task
            return LocalLLMClient(config=model_config)

        raise RouterError(f"Unknown provider '{provider}' for model '{model}'")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def routing_table(self) -> Dict[str, str]:
        """Return current task → model assignments."""
        return dict(self._routing)

    def set_routing(self, task: str, model: str) -> None:
        """Override routing for a specific task at runtime."""
        self._routing[task] = model
        logger.info("router: %s now routes to %s", task, model)

    def register_client(self, model: str, client: LLMBase) -> None:
        """Register a pre-built client (useful for tests / mock LLMs)."""
        self._clients[model] = client