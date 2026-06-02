"""
truenorth/testing/mock_llm.py
Mock LLM client for tests. Returns canned responses without network calls.
"""
from __future__ import annotations
from typing import AsyncIterator, Dict, List, Optional
from truenorth.llm.base import LLMBase, LLMResponse, Message, StreamChunk


class MockLLMClient(LLMBase):
    """
    Deterministic mock LLM for tests.
    Pass `responses` dict keyed by keywords in the prompt.
    When a keyword matches, returns the associated response.
    Otherwise returns `default`.
    """
    model_name = "mock"

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default:   str = "mock response",
    ):
        super().__init__()
        self._responses  = responses or {}
        self._default    = default
        self.call_count  = 0
        self.last_prompt = ""

    def _match(self, messages: List[Message]) -> str:
        text = " ".join(m.content for m in messages if hasattr(m, "content")).lower()
        self.last_prompt = text
        for kw, resp in self._responses.items():
            if kw.lower() in text:
                return resp
        return self._default

    async def generate(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        self.call_count += 1
        content = self._match(messages)
        return LLMResponse(
            content       = content,
            model         = self.model_name,
            input_tokens  = 10,
            output_tokens = 20,
        )

    async def generate_stream(
        self,
        messages:    List[Message],
        system:      Optional[str] = None,
        max_tokens:  int   = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        self.call_count += 1
        content = self._match(messages)
        for word in content.split():
            yield StreamChunk(delta=word + " ")
        yield StreamChunk(delta="", is_final=True, input_tokens=10, output_tokens=20)
