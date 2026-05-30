from __future__ import annotations
import google.generativeai as genai
from truenorth.llm.base import BaseLLMClient, LLMResponse


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    async def complete(self, prompt: str, system: str = "", model: str = "",
                       temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse:
        model = model or "gemini-2.0-flash-lite"
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt

        gemini = genai.GenerativeModel(model)
        config = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        resp = await gemini.generate_content_async(full_prompt, generation_config=config)

        in_tokens = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
        out_tokens = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0

        return LLMResponse(
            content=resp.text,
            model=model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
