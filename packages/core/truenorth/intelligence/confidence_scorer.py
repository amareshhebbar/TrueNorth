"""Score how confident we are in each extracted field value."""

from __future__ import annotations
import re


UNCERTAINTY_PHRASES = [
    r"\bI think\b", r"\bmaybe\b", r"\bprobably\b", r"\baround\b",
    r"\babout\b", r"\bapproximately\b", r"\bsomewhere\b", r"\bnot sure\b",
    r"\bI guess\b", r"\bI believe\b", r"\bI'm not certain\b", r"\broughly\b",
    r"\bsomething like\b", r"\bI don't know exactly\b",
]

CERTAINTY_PHRASES = [
    r"\bexactly\b", r"\bprecisely\b", r"\bmy\b.*\bis\b",
    r"\bI am\b", r"\bI'm\b \d", r"\bI weigh\b", r"\bI was born\b",
]


def score_confidence(raw_text: str, extracted_value: any) -> float:
    """
    Rule-based confidence scoring from raw user text.
    Returns 0.0 to 1.0.
    """
    if not raw_text:
        return 0.5

    text_lower = raw_text.lower()
    score = 0.7  # default baseline

    # Penalize uncertainty language
    for pattern in UNCERTAINTY_PHRASES:
        if re.search(pattern, raw_text, re.IGNORECASE):
            score -= 0.15
            break

    # Boost certainty language
    for pattern in CERTAINTY_PHRASES:
        if re.search(pattern, raw_text, re.IGNORECASE):
            score += 0.15
            break

    # Exact numeric values with units → high confidence
    if isinstance(extracted_value, (int, float)):
        if re.search(r"\b\d+(\.\d+)?\s*(kg|lbs|cm|m|ft|in|years|y\.?o\.?)\b",
                     raw_text, re.IGNORECASE):
            score += 0.1

    # Range given → low confidence ("70-75 kg")
    if re.search(r"\b\d+\s*[-–]\s*\d+\b", raw_text):
        score -= 0.2

    return max(0.1, min(1.0, score))
