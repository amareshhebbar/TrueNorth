"""
Fast language detection for TrueNorth conversations. Handles Indian languages
as first-class targets (Hindi, Tamil, Telugu, Kannada, Bengali, Marathi, Gujarati)
plus common global languages.

Detection strategy (in order of cost):
  1. Unicode block detection — instant, works for Devanagari / Tamil / Telugu / Kannada
  2. Script-level heuristics — detects CJK, Arabic, Cyrillic, etc.
  3. Vocabulary fingerprint — short Latin-script messages (English / Hinglish / Romanized Hindi)
  4. Fallback: "en" (safe default)

No external API calls. No ML models. < 1ms per call.

Returned language codes follow BCP-47:
  hi  = Hindi
  ta  = Tamil
  te  = Telugu
  kn  = Kannada
  bn  = Bengali
  mr  = Marathi
  gu  = Gujarati
  pa  = Punjabi
  en  = English
  es  = Spanish
  fr  = French
  de  = German
  pt  = Portuguese
  ar  = Arabic
  zh  = Chinese (Mandarin, simplified/traditional)
  ja  = Japanese
  ko  = Korean
  ru  = Russian
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class LanguageDetectionResult:
    language_code: str          # BCP-47 code e.g. "hi", "ta", "en"
    language_name: str          
    confidence:    float        # 0.0–1.0
    script:        str          # Unicode script family e.g. "Devanagari", "Latin"
    is_indian:     bool         # True for the 7 Indian languages
    is_romanized:  bool         # True when Indian lang written in Latin script (Hinglish)

    def to_dict(self) -> dict:
        return {
            "language_code": self.language_code,
            "language_name": self.language_name,
            "confidence":    round(self.confidence, 3),
            "script":        self.script,
            "is_indian":     self.is_indian,
            "is_romanized":  self.is_romanized,
        }


# ---------------------------------------------------------------------------
# Unicode block ranges for South / Southeast Asian scripts
# ---------------------------------------------------------------------------

_UNICODE_SCRIPT_RANGES = {
    # Indian scripts — exact Unicode block boundaries
    "Devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Sanskrit
    "Bengali":    (0x0980, 0x09FF),   # Bengali, Assamese
    "Gujarati":   (0x0A80, 0x0AFF),
    "Gurmukhi":   (0x0A00, 0x0A7F),   # Punjabi
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
    "Sinhala":    (0x0D80, 0x0DFF),
    # Other world scripts
    "Arabic":     (0x0600, 0x06FF),
    "Hebrew":     (0x0590, 0x05FF),
    "Cyrillic":   (0x0400, 0x04FF),
    "Greek":      (0x0370, 0x03FF),
    "CJK":        (0x4E00, 0x9FFF),   # covers most CJK unified ideographs
    "Hangul":     (0xAC00, 0xD7AF),   # Korean
    "Hiragana":   (0x3040, 0x309F),
    "Katakana":   (0x30A0, 0x30FF),
    "Thai":       (0x0E00, 0x0E7F),
}

_SCRIPT_TO_LANGUAGE: dict[str, tuple[str, str]] = {
    "Devanagari": ("hi",  "Hindi"),
    "Bengali":    ("bn",  "Bengali"),
    "Gujarati":   ("gu",  "Gujarati"),
    "Gurmukhi":   ("pa",  "Punjabi"),
    "Tamil":      ("ta",  "Tamil"),
    "Telugu":     ("te",  "Telugu"),
    "Kannada":    ("kn",  "Kannada"),
    "Malayalam":  ("ml",  "Malayalam"),
    "Arabic":     ("ar",  "Arabic"),
    "Hebrew":     ("he",  "Hebrew"),
    "Cyrillic":   ("ru",  "Russian"),
    "Greek":      ("el",  "Greek"),
    "CJK":        ("zh",  "Chinese"),
    "Hangul":     ("ko",  "Korean"),
    "Hiragana":   ("ja",  "Japanese"),
    "Katakana":   ("ja",  "Japanese"),
    "Thai":       ("th",  "Thai"),
}

_INDIAN_CODES = {"hi", "ta", "te", "kn", "bn", "mr", "gu", "pa", "ml"}

# ---------------------------------------------------------------------------
# Vocabulary fingerprints for Latin-script languages
# ---------------------------------------------------------------------------

_LATIN_VOCAB_FINGERPRINTS: dict[str, list[str]] = {
    "hi": ["hai", "hain", "kya", "nahi", "mera", "meri", "aap", "tum", "yeh",
           "woh", "kar", "karo", "tha", "thi", "hoga", "bhi", "sirf", "lekin",
           "aur", "ya", "ki", "ke", "ko", "se", "mein", "par", "ek", "do"],
    "ta": ["enna", "illai", "vandha", "irukku", "sollu", "yenna", "naanga",
           "neenga", "intha", "antha", "romba", "konjam", "vaanga", "ponga"],
    "te": ["enti", "ledu", "cheppandi", "cheyandi", "mee", "nenu", "meeru",
           "okka", "rendu", "ikkade", "akkade", "chala", "ante", "ayite"],
    "kn": ["hege", "enu", "illa", "banni", "hogri", "namge", "nimge", "adhu",
           "idhu", "alli", "illi", "thumba", "summane", "haagu", "mattu"],
    "en": ["the", "is", "are", "was", "were", "have", "has", "will", "would",
           "could", "should", "can", "may", "might", "do", "does", "did",
           "and", "but", "or", "so", "that", "this", "it", "he", "she", "they"],
    "es": ["el", "la", "los", "las", "es", "son", "está", "están", "tengo",
           "tiene", "quiero", "necesito", "gracias", "hola", "por", "que"],
    "fr": ["le", "la", "les", "est", "sont", "je", "tu", "il", "nous", "vous",
           "ils", "avec", "pour", "dans", "qui", "que", "très", "mais", "bonjour"],
    "de": ["ich", "du", "er", "wir", "sie", "ist", "sind", "haben", "werden",
           "und", "oder", "aber", "nicht", "ein", "eine", "der", "die", "das"],
    "pt": ["eu", "você", "ele", "nós", "são", "está", "tem", "tenho", "quero",
           "obrigado", "olá", "para", "com", "que", "não", "sim", "mais"],
}


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------

class LanguageDetector:
    """
    Detect the language of a user message. Fast, zero-API, works offline.

    Usage:
        detector = LanguageDetector()
        result = detector.detect("main thodi der mein aaunga")
        # LanguageDetectionResult(language_code='hi', ...)
    """

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or {}

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of a text string.

        Steps:
          1. Strip and normalise
          2. Unicode block scan → identifies non-Latin scripts immediately
          3. Latin vocab fingerprint → discriminates Latin-script languages
          4. Default to English

        Args:
            text: User message (any length; works on as little as 1-2 words)

        Returns:
            LanguageDetectionResult
        """
        if not text or not text.strip():
            return self._result("en", "English", 0.50, "Latin", False, False)

        text_clean = text.strip()

        script_result = self._detect_by_unicode_block(text_clean)
        if script_result:
            return script_result

        vocab_result = self._detect_by_vocab(text_clean)
        if vocab_result:
            return vocab_result

        logger.debug("language_detector: fell through to English default for text=%r", text_clean[:40])
        return self._result("en", "English", 0.50, "Latin", False, False)

    def detect_from_history(self, messages: List[str], window: int = 5) -> LanguageDetectionResult:
        """
        Detect language from a sliding window of recent messages.
        More robust than single-message detection for very short messages.
        """
        recent = [m for m in messages if m.strip()][-window:]
        if not recent:
            return self._result("en", "English", 0.50, "Latin", False, False)

        combined = " ".join(recent)
        return self.detect(combined)

    def is_indian_language(self, language_code: str) -> bool:
        return language_code in _INDIAN_CODES

    def is_romanized_indian(self, text: str) -> bool:
        """Quick check: is this Romanized Hindi / Tamil etc. (Hinglish)?"""
        result = self._detect_by_vocab(text)
        if result and result.is_indian and result.script == "Latin":
            return True
        return False

    # ------------------------------------------------------------------
    # Internal: Unicode block detection
    # ------------------------------------------------------------------

    def _detect_by_unicode_block(self, text: str) -> Optional[LanguageDetectionResult]:
        """
        Count characters from each Unicode block. If any non-Latin script dominates
        (> 30% of alphabetic chars), return that language.
        """
        counts: dict[str, int] = {script: 0 for script in _UNICODE_SCRIPT_RANGES}
        alpha_total = 0

        for char in text:
            cp = ord(char)
            if not unicodedata.category(char).startswith("L"):
                continue
            alpha_total += 1
            for script, (lo, hi) in _UNICODE_SCRIPT_RANGES.items():
                if lo <= cp <= hi:
                    counts[script] += 1
                    break

        if alpha_total == 0:
            return None

        dominant_script = max(counts, key=counts.get)
        dominant_count  = counts[dominant_script]
        ratio           = dominant_count / alpha_total

        if ratio < 0.25:
            return None

        if dominant_script not in _SCRIPT_TO_LANGUAGE:
            return None

        lang_code, lang_name = _SCRIPT_TO_LANGUAGE[dominant_script]

        if lang_code == "hi" and self._looks_marathi(text):
            lang_code, lang_name = "mr", "Marathi"

        confidence = min(0.50 + ratio * 0.50, 0.97)
        is_indian  = lang_code in _INDIAN_CODES

        logger.debug(
            "language_detector: unicode block=%s ratio=%.2f → %s",
            dominant_script, ratio, lang_code,
        )
        return self._result(lang_code, lang_name, confidence, dominant_script, is_indian, False)

    def _looks_marathi(self, text: str) -> bool:
        marathi_markers = re.compile(r"\b(आहे|नाही|माझा|माझी|तुम्ही|आम्ही|काय|कुठे|केव्हा)\b")
        return bool(marathi_markers.search(text))

    # ------------------------------------------------------------------
    # Internal: Vocabulary fingerprint
    # ------------------------------------------------------------------

    def _detect_by_vocab(self, text: str) -> Optional[LanguageDetectionResult]:
        """
        Match words against per-language vocabulary lists.
        Returns the best match above a minimum score threshold.
        """
        words = re.findall(r"\b[a-zA-ZÀ-ÿ\u00C0-\u024F]+\b", text.lower())
        if not words:
            return None

        word_set = set(words)
        total    = len(words)
        scores: dict[str, float] = {}

        for lang_code, vocab in _LATIN_VOCAB_FINGERPRINTS.items():
            hits = sum(1 for w in vocab if w in word_set)
            scores[lang_code] = hits / max(total, 1)

        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        if best_score < 0.04:
            return None  

        lang_names = {
            "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
            "en": "English", "es": "Spanish", "fr": "French",
            "de": "German", "pt": "Portuguese",
        }
        lang_name  = lang_names.get(best_lang, best_lang)
        is_indian  = best_lang in _INDIAN_CODES
        is_roman   = is_indian  # Romanized because we're in the Latin branch

        confidence = min(0.35 + best_score * 3.0, 0.85)  # cap at 0.85 — vocab is imprecise

        logger.debug(
            "language_detector: vocab match=%s score=%.3f → %s",
            best_lang, best_score, lang_name,
        )
        return self._result(best_lang, lang_name, confidence, "Latin", is_indian, is_roman)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _result(
        code: str, name: str, confidence: float,
        script: str, is_indian: bool, is_romanized: bool,
    ) -> LanguageDetectionResult:
        return LanguageDetectionResult(
            language_code = code,
            language_name = name,
            confidence    = confidence,
            script        = script,
            is_indian     = is_indian,
            is_romanized  = is_romanized,
        )


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_default_detector = LanguageDetector()

def detect_language(text: str) -> LanguageDetectionResult:
    """Module-level convenience function. Uses the default detector instance."""
    return _default_detector.detect(text)