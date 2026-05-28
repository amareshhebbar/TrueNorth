"""
truenorth/intelligence/emotion_detector.py

Classifies the emotional state of the user's message.
Uses a two-stage approach:
  1. Fast heuristic rules (regex + word lists) — < 1ms, runs always
  2. LLM classification — only when heuristics are inconclusive (> 0.3 uncertainty)

The LLM stage uses the cheapest model (Gemini Flash) with a structured prompt
that returns a JSON object.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion categories
# ---------------------------------------------------------------------------

class Emotion:
    NEUTRAL     = "neutral"
    HAPPY       = "happy"
    FRUSTRATED  = "frustrated"
    CONFUSED    = "confused"
    DISTRESSED  = "distressed"
    ANGRY       = "angry"
    EXCITED     = "excited"
    ANXIOUS     = "anxious"
    DISENGAGED  = "disengaged"


@dataclass
class EmotionResult:
    label:      str    # Emotion.*
    score:      float  # 0.0–1.0 confidence in this label
    is_negative: bool  # shortcut for engine routing
    raw_signals: list  # which heuristic signals fired

    def to_dict(self) -> dict:
        return {
            "label":       self.label,
            "score":       round(self.score, 3),
            "is_negative": self.is_negative,
        }


# ---------------------------------------------------------------------------
# Heuristic word banks
# ---------------------------------------------------------------------------

_PATTERNS = {
    Emotion.FRUSTRATED: re.compile(
        r"\b(ugh|argh|annoying|frustrated|stop|enough|already|this is taking|waste|pointless|"
        r"why (do|are|keep|is)|forget it|never mind|whatever|done with this)\b",
        re.IGNORECASE,
    ),
    Emotion.ANGRY: re.compile(
        r"\b(angry|furious|outraged|ridiculous|unacceptable|stupid|idiot|moron|"
        r"this is (a )?joke|absolutely not|demand|lawyer)\b",
        re.IGNORECASE,
    ),
    Emotion.DISTRESSED: re.compile(
        r"\b(help|please|urgent|emergency|scared|afraid|worried|can'?t (breathe|cope|handle)|"
        r"overwhelmed|panic|crisis|hurt|pain|depressed|suicid|harm)\b",
        re.IGNORECASE,
    ),
    Emotion.CONFUSED: re.compile(
        r"\b(what\?|huh\?|don'?t understand|not sure what|confused|unclear|"
        r"what do you mean|explain|clarify|lost)\b",
        re.IGNORECASE,
    ),
    Emotion.HAPPY: re.compile(
        r"\b(great|awesome|love|perfect|amazing|thanks|thank you|wonderful|"
        r"fantastic|excellent|happy|glad|appreciate)\b",
        re.IGNORECASE,
    ),
    Emotion.EXCITED: re.compile(
        r"\b(excited|can'?t wait|so ready|pumped|stoked|yay|yes!|finally|looking forward)\b",
        re.IGNORECASE,
    ),
    Emotion.ANXIOUS: re.compile(
        r"\b(nervous|anxious|anxiety|worried|what if|hope this|not sure if|scared to|"
        r"afraid that|concern|second.?guess)\b",
        re.IGNORECASE,
    ),
}

_DISENGAGEMENT_SIGNS = re.compile(
    r"^(ok|okay|sure|fine|whatever|idk|yes|no|mhm|yep|nope|k|hmm+|uh+)\s*\.?$",
    re.IGNORECASE,
)

_NEGATIVE_EMOTIONS = {Emotion.FRUSTRATED, Emotion.ANGRY, Emotion.DISTRESSED, Emotion.ANXIOUS}


# ---------------------------------------------------------------------------
# EmotionDetector
# ---------------------------------------------------------------------------

class EmotionDetector:
    """
    Detect user emotion from message text.

    Usage:
        detector = EmotionDetector(router=llm_router)
        result = await detector.detect("This is taking forever, I give up")
        # EmotionResult(label='frustrated', score=0.85, is_negative=True, ...)
    """

    UNCERTAINTY_THRESHOLD: float = 0.40
    MIN_LLM_LENGTH: int = 15

    def __init__(self, router: Optional["LLMRouter"] = None):
        self._router = router

    async def detect(self, text: str, use_llm: bool = True) -> EmotionResult:
        """
        Detect emotion from user text.

        Args:
            text:    User message
            use_llm: Fall back to LLM when heuristics are inconclusive (default True)
        """
        if not text or not text.strip():
            return EmotionResult(
                label="neutral", score=0.5, is_negative=False, raw_signals=[]
            )

        text = text.strip()

        # Stage 1: Heuristic
        heuristic = self._heuristic(text)

        # Stage 2: LLM fallback when uncertain
        if (
            use_llm
            and self._router is not None
            and heuristic.score < self.UNCERTAINTY_THRESHOLD
            and len(text) >= self.MIN_LLM_LENGTH
        ):
            llm_result = await self._llm_classify(text)
            if llm_result and llm_result.score > heuristic.score:
                return llm_result

        return heuristic

    def detect_sync(self, text: str) -> EmotionResult:
        """Synchronous heuristic-only detection (for dry-run / testing)."""
        return self._heuristic(text or "")

    # ------------------------------------------------------------------
    # Stage 1: Heuristics
    # ------------------------------------------------------------------

    def _heuristic(self, text: str) -> EmotionResult:
        signals: list[str] = []
        scores: dict[str, float] = {}

        # Check disengagement first (short filler answers)
        if _DISENGAGEMENT_SIGNS.match(text.strip()):
            return EmotionResult(
                label=Emotion.DISENGAGED, score=0.65,
                is_negative=False, raw_signals=["short_filler"],
            )

        # Check each emotion pattern
        for emotion, pattern in _PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                count  = len(matches)
                score  = min(0.50 + count * 0.20, 0.92)
                scores[emotion] = score
                words = [m.group(0) for m in matches[:2]]
                signals.append(f"{emotion}:{','.join(words)}")

        if not scores:
            return EmotionResult(
                label=Emotion.NEUTRAL, score=0.50,
                is_negative=False, raw_signals=[],
            )

        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]
        return EmotionResult(
            label        = top_label,
            score        = top_score,
            is_negative  = top_label in _NEGATIVE_EMOTIONS,
            raw_signals  = signals,
        )

    # ------------------------------------------------------------------
    # Stage 2: LLM classification
    # ------------------------------------------------------------------

    async def _llm_classify(self, text: str) -> Optional[EmotionResult]:
        """Ask Gemini Flash to classify emotion. Returns None on any error."""
        from truenorth.llm.base import Message
        from truenorth.llm.router import TASK_CLASSIFY

        prompt = (
            f"Classify the emotional tone of this user message in ONE word from this list: "
            f"neutral, happy, frustrated, confused, distressed, angry, excited, anxious, disengaged.\n\n"
            f"User message: {text!r}\n\n"
            f"Reply ONLY with valid JSON: {{\"label\": \"<emotion>\", \"score\": <0.0-1.0>}}"
        )
        try:
            resp = await self._router.generate(
                task=TASK_CLASSIFY,
                messages=[Message(role="user", content=prompt)],
                max_tokens=40,
                temperature=0.0,
            )
            data = json.loads(resp.content.strip())
            label = data.get("label", Emotion.NEUTRAL)
            score = float(data.get("score", 0.5))
            return EmotionResult(
                label=label, score=score,
                is_negative=label in _NEGATIVE_EMOTIONS,
                raw_signals=["llm"],
            )
        except Exception as e:
            logger.debug("emotion LLM fallback failed: %s", e)
            return None