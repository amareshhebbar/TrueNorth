"""Detect user language. Supports Indian languages."""
from __future__ import annotations

INDIAN_LANGUAGE_CODES = {
    "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada",
    "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi",
    "ml": "Malayalam", "or": "Odia"
}

# Unicode ranges for Indian scripts
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),   # Devanagari
    "te": (0x0C00, 0x0C7F),   # Telugu
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "bn": (0x0980, 0x09FF),   # Bengali
    "ml": (0x0D00, 0x0D7F),   # Malayalam
}


def detect_language(text: str) -> str:
    """Fast script-based language detection for Indian languages."""
    if not text:
        return "en"
    for lang, (start, end) in SCRIPT_RANGES.items():
        if any(start <= ord(c) <= end for c in text):
            return lang
    return "en"
