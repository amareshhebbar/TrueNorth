"""

Abstract base class that every LLM provider adapter must implement.
The router dispatches calls to concrete subclasses (Anthropic, OpenAI, Gemini, Local).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

@dataclass
class Message:
    """One message in a conversation."""
    role:    str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

@dataclass
class LLMResponse:
    """Returned by every LLM call."""
    content:       str
    model:         str
    input_tokens:  int    = 0
    output_tokens: int    = 0
    latency_ms:    int    = 0
    raw:           Any    = field(default=None, repr=False)
    metadata:      Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

@dataclass
class StreamChunk:
    """One chunk from a streaming response."""
    delta:         str
    is_final:      bool = False
    input_tokens:  int  = 0
    output_tokens: int  = 0

class LLMBase(ABC):
    """
    Abstract base class for all TrueNorth LLM providers.

    Subclasses must implement:
      - generate()        — single-turn completion
      - generate_stream() — streaming completion

    Optional override:
      - supports_streaming — set to False for providers without streaming
      - max_context_tokens — context window size
    """

    model_name:         str  = "unknown"
    supports_streaming: bool = True
    max_context_tokens: int  = 200_000

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    async def generate(
        self,
        messages:     List[Message],
        system:       Optional[str] = None,
        max_tokens:   int = 1024,
        temperature:  float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages:    Conversation history. Last message is the user turn.
            system:      Optional system prompt.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature (0 = deterministic).
            **kwargs:    Provider-specific options.

        Returns:
            LLMResponse
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages:     List[Message],
        system:       Optional[str] = None,
        max_tokens:   int = 1024,
        temperature:  float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Streaming version of generate(). Yields StreamChunk objects.
        The final chunk has is_final=True and populated token counts.
        """
        ...

    def _measure(self) -> "_Timer":
        return _Timer()

    def _count_tokens_approx(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token. Use only when exact count unavailable."""
        return max(1, len(text) // 4)

    def _messages_to_str(self, messages: List[Message]) -> str:
        """Join messages into a single string (for token estimation)."""
        return " ".join(m.content for m in messages)

    async def health_check(self) -> bool:
        """
        Verify the provider is reachable. Returns True if healthy.
        Default implementation sends a minimal test request.
        """
        try:
            resp = await self.generate(
                messages=[Message(role="user", content="ping")],
                max_tokens=5,
                temperature=0.0,
            )
            return bool(resp.content)
        except Exception:
            return False

class _Timer:
    """Context manager that measures elapsed milliseconds."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)

    @property
    def ms(self) -> int:
        return getattr(self, "elapsed_ms", 0)

BaseLLMClient = LLMBase
