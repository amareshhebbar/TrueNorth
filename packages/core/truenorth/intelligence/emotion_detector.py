"""
Detect user emotional state from message text + conversation history.
No external API — pure LLM classification.
"""

from __future__ import annotations
from dataclasses import dataclass
from truenorth.llm.router import LLMRouter

EMOTION_STATES = ["engaged", "neutral", "confused", "frustrated", "anxious", "rushed", "distressed"]

SYSTEM = """You are an emotion classifier for a conversational AI system.
Analyze the user's message and conversation history to determine their emotional state.
Be conservative — only classify as 'distressed' if there are clear signs of crisis."""

PROMPT_TEMPLATE = """
Conversation history (last 4 turns):
{history}

Latest user message: "{message}"

Classify the user's current emotional state. Choose exactly one from:
{states}

Also rate your confidence (0.0-1.0) and suggest how the agent should adapt.

Respond in JSON:
{{
  "state": "<state>",
  "confidence": 0.0,
  "reasoning": "<brief>",
  "agent_adaptation": {{
    "tone": "<suggestion>",
    "skip_optional": false,
    "add_acknowledgment": false,
    "slow_down": false,
    "trigger_escalation": false
  }}
}}
"""


@dataclass
class EmotionResult:
    state: str
    confidence: float
    reasoning: str
    skip_optional: bool = False
    add_acknowledgment: bool = False
    slow_down: bool = False
    trigger_escalation: bool = False


class EmotionDetector:
    def __init__(self, router: LLMRouter):
        self.router = router

    async def detect(self, message: str, history: list) -> EmotionResult:
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}"
            for t in history[-4:]
        )

        prompt = PROMPT_TEMPLATE.format(
            history=history_text or "(no history yet)",
            message=message,
            states=", ".join(EMOTION_STATES)
        )

        try:
            data, _ = await self.router.complete_json(
                task="emotion_detection",
                prompt=prompt,
                system=SYSTEM,
                temperature=0.1,
                max_tokens=300,
            )
            adaptation = data.get("agent_adaptation", {})
            return EmotionResult(
                state=data.get("state", "neutral"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                skip_optional=adaptation.get("skip_optional", False),
                add_acknowledgment=adaptation.get("add_acknowledgment", False),
                slow_down=adaptation.get("slow_down", False),
                trigger_escalation=adaptation.get("trigger_escalation", False),
            )
        except Exception:
            return EmotionResult(state="neutral", confidence=0.0, reasoning="detection failed")
