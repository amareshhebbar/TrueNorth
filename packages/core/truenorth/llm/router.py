"""
LLM Router — routes different tasks to the most cost-efficient model.

Field extraction  → gemini-2.0-flash-lite  (cheap, fast)
Emotion detection → gemini-2.0-flash-lite
Conversation      → claude-3-5-haiku       (quality response)
Output generation → claude-sonnet-4        (best quality)
Vision/document   → claude-sonnet-4        (needs vision)
"""

from __future__ import annotations
import os
from functools import lru_cache
from truenorth.llm.base import BaseLLMClient, LLMResponse

TASK_MODELS = {
    "field_extraction":    os.getenv("EXTRACTION_MODEL",    "gemini-2.0-flash-lite"),
    "emotion_detection":   os.getenv("EXTRACTION_MODEL",    "gemini-2.0-flash-lite"),
    "conflict_detection":  os.getenv("EXTRACTION_MODEL",    "gemini-2.0-flash-lite"),
    "conversation":        os.getenv("CONVERSATION_MODEL",  "claude-3-5-haiku-20241022"),
    "output_generation":   os.getenv("OUTPUT_MODEL",        "claude-sonnet-4-20250514"),
    "vision_extraction":   os.getenv("OUTPUT_MODEL",        "claude-sonnet-4-20250514"),
}


class LLMRouter:
    def __init__(self, max_cost_usd: float = 0.05):
        self.max_cost_usd = max_cost_usd
        self._clients: dict[str, BaseLLMClient] = {}
        self._session_cost = 0.0
        self._session_tokens = 0
        self._setup_clients()

    def _setup_clients(self):
        if key := os.getenv("ANTHROPIC_API_KEY"):
            from truenorth.llm.anthropic import AnthropicClient
            self._clients["anthropic"] = AnthropicClient(key)

        if key := os.getenv("OPENAI_API_KEY"):
            from truenorth.llm.openai import OpenAIClient
            self._clients["openai"] = OpenAIClient(key)

        if key := os.getenv("GOOGLE_API_KEY"):
            from truenorth.llm.gemini import GeminiClient
            self._clients["gemini"] = GeminiClient(key)

    def _get_client_for_model(self, model: str) -> BaseLLMClient:
        if "claude" in model and "anthropic" in self._clients:
            return self._clients["anthropic"]
        if ("gpt" in model or "o1" in model) and "openai" in self._clients:
            return self._clients["openai"]
        if "gemini" in model and "gemini" in self._clients:
            return self._clients["gemini"]

        # Fallback to first available
        if self._clients:
            return next(iter(self._clients.values()))
        raise RuntimeError("No LLM clients configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY.")

    async def complete(self, task: str, prompt: str, system: str = "",
                       temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        model = TASK_MODELS.get(task, TASK_MODELS["conversation"])

        # Downgrade if near budget
        if self._session_cost > self.max_cost_usd * 0.8:
            model = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash-lite")

        client = self._get_client_for_model(model)
        response = await client.complete(prompt=prompt, system=system,
                                          model=model, temperature=temperature,
                                          max_tokens=max_tokens)
        self._session_cost += response.cost_usd
        self._session_tokens += response.total_tokens
        return response

    async def complete_json(self, task: str, prompt: str, system: str = "", **kwargs) -> tuple[dict, LLMResponse]:
        model = TASK_MODELS.get(task, TASK_MODELS["conversation"])
        client = self._get_client_for_model(model)
        result, response = await client.complete_json(prompt=prompt, system=system,
                                                       model=model, **kwargs)
        self._session_cost += response.cost_usd
        self._session_tokens += response.total_tokens
        return result, response

    @property
    def session_cost(self) -> float:
        return self._session_cost

    @property
    def session_tokens(self) -> int:
        return self._session_tokens

    def reset_session(self):
        self._session_cost = 0.0
        self._session_tokens = 0
