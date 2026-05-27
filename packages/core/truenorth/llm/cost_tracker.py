"""Token and cost tracking per session."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Approximate costs per 1M tokens (input + output blended) as of 2025
MODEL_COSTS_PER_1M = {
    "claude-sonnet-4-20250514": 15.0,
    "claude-3-5-sonnet-20241022": 15.0,
    "claude-3-5-haiku-20241022": 1.25,
    "gpt-4o": 7.50,
    "gpt-4o-mini": 0.60,
    "gemini-2.0-flash-lite": 0.10,
    "gemini-2.0-flash": 0.40,
}


@dataclass
class CostTracker:
    session_id: str
    budget_usd: float = 0.05
    tokens_used: int = 0
    cost_usd: float = 0.0
    _model_usage: dict = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        total_tokens = input_tokens + output_tokens
        cost_per_1m = MODEL_COSTS_PER_1M.get(model, 1.0)
        cost = (total_tokens / 1_000_000) * cost_per_1m
        self.tokens_used += total_tokens
        self.cost_usd += cost
        self._model_usage[model] = self._model_usage.get(model, 0) + total_tokens
        logger.debug(f"[{self.session_id}] {model}: +{total_tokens} tokens, +${cost:.5f} (total: ${self.cost_usd:.4f})")

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_usd - self.cost_usd)

    @property
    def budget_exceeded(self) -> bool:
        return self.cost_usd >= self.budget_usd

    @property
    def budget_warning(self) -> bool:
        return self.cost_usd >= self.budget_usd * 0.8
