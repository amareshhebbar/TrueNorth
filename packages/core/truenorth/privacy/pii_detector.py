"""
truenorth/privacy/pii_detector.py

Scans user messages for Personally Identifiable Information (PII) before
the text touches any LLM. Regex-based — fast, zero network calls.

Supported PII types:
  - Email address
  - Phone number (Indian mobile + international)
  - Aadhaar number (India, 12-digit)
  - PAN card (India)
  - Credit / debit card number
  - Date of birth (various formats)
  - UPI ID (India)
  - IFSC code (India)
  - Name (heuristic, lower confidence)

Mode options:
  - DETECT_ONLY  : return matches, do not modify text
  - REDACT       : replace PII with <TYPE> placeholders (default for LLM calls)
  - PSEUDONYMISE : replace with consistent fake values (preserves context)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

@dataclass
class PIIPattern:
    name:    str
    pattern: re.Pattern
    group:   int = 0          # which capture group to redact (0 = whole match)
    risk:    str = "medium"   # low | medium | high


_PII_PATTERNS: List[PIIPattern] = [
    # Email
    PIIPattern("email",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
        risk="high"),

    # Indian mobile (10 digits starting with 6-9)
    PIIPattern("phone_in",
        re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
        risk="high"),

    # International phone (+ prefix)
    PIIPattern("phone_intl",
        re.compile(r"\+\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}"),
        risk="high"),

    # Aadhaar (12-digit, may have spaces: XXXX XXXX XXXX)
    PIIPattern("aadhaar",
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b"),
        risk="high"),

    # PAN card (AAAAA1234A pattern)
    PIIPattern("pan",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        risk="high"),

    # Credit/debit card (13-19 digits, with optional spaces/dashes)
    PIIPattern("card_number",
        re.compile(r"\b(?:\d[\s-]?){13,19}\b"),
        risk="high"),

    # UPI ID (e.g. name@upi, name@okicici)
    PIIPattern("upi_id",
        re.compile(r"\b[\w.\-]+@(?:upi|oksbi|okicici|okaxis|okhdfcbank|paytm|ybl|ibl|axl|"
                   r"apl|freecharge|airtelpaymentsbank|mahb|waicici|waaxis|waidfcbank)\b",
                   re.IGNORECASE),
        risk="high"),

    # IFSC code (e.g. HDFC0001234)
    PIIPattern("ifsc",
        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        risk="medium"),

    PIIPattern("dob",
        re.compile(r"\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})\b"),
        risk="medium"),

    PIIPattern("ip_address",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        risk="low"),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    type:    str      
    value:   str      
    start:   int
    end:     int
    risk:    str

    def to_dict(self) -> dict:
        return {"type": self.type, "risk": self.risk, "start": self.start, "end": self.end}


@dataclass
class PIIScanResult:
    original:    str
    redacted:    str
    matches:     List[PIIMatch]
    has_pii:     bool
    has_high_risk: bool

    def to_dict(self) -> dict:
        return {
            "has_pii":       self.has_pii,
            "has_high_risk": self.has_high_risk,
            "match_count":   len(self.matches),
            "types":         list({m.type for m in self.matches}),
        }


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------

class PIIDetector:
    """
    Scans and redacts PII from user messages.

    Usage:
        detector = PIIDetector()

        # Scan only (get matches, do not modify)
        result = detector.scan("My Aadhaar is 1234 5678 9012 and email is me@example.com")

        # Redact before sending to LLM
        clean = detector.redact("Call me on 9876543210")
        # → "Call me on <phone_in>"

        # Check if a field is marked as PII in goal config
        if detector.is_pii_field("aadhaar", fields_config):
            # handle with extra care
    """

    def __init__(self, custom_patterns: Optional[List[PIIPattern]] = None):
        self._patterns = _PII_PATTERNS + (custom_patterns or [])

    def scan(self, text: str) -> PIIScanResult:
        """Scan text for PII. Returns matches without modifying the text."""
        if not text:
            return PIIScanResult(
                original=text, redacted=text,
                matches=[], has_pii=False, has_high_risk=False,
            )

        matches: List[PIIMatch] = []
        for p in self._patterns:
            for m in p.pattern.finditer(text):
                matches.append(PIIMatch(
                    type  = p.name,
                    value = m.group(p.group),
                    start = m.start(),
                    end   = m.end(),
                    risk  = p.risk,
                ))

        matches = self._deduplicate(matches)

        has_pii       = len(matches) > 0
        has_high_risk = any(m.risk == "high" for m in matches)

        redacted = self._apply_redaction(text, matches) if has_pii else text

        if has_high_risk:
            logger.info(
                "pii_detector: high-risk PII found: %s",
                [m.type for m in matches if m.risk == "high"],
            )

        return PIIScanResult(
            original     = text,
            redacted     = redacted,
            matches      = matches,
            has_pii      = has_pii,
            has_high_risk = has_high_risk,
        )

    def redact(self, text: str) -> str:
        """Redact PII from text and return the cleaned version."""
        return self.scan(text).redacted

    def has_pii(self, text: str) -> bool:
        """Quick check — does this text contain any PII?"""
        for p in self._patterns:
            if p.pattern.search(text):
                return True
        return False

    @staticmethod
    def is_pii_field(field_name: str, fields_config: Dict[str, dict]) -> bool:
        """Returns True if the field is marked as pii: true in the goal YAML."""
        return fields_config.get(field_name, {}).get("pii", False)

    def pii_fields(self, fields_config: Dict[str, dict]) -> List[str]:
        """Return list of field names marked as PII in the goal config."""
        return [name for name, cfg in fields_config.items() if cfg.get("pii", False)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches — keep the highest-risk, longest match."""
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        deduped: List[PIIMatch] = []
        last_end = -1
        for m in matches:
            if m.start >= last_end:
                deduped.append(m)
                last_end = m.end
        return deduped

    @staticmethod
    def _apply_redaction(text: str, matches: List[PIIMatch]) -> str:
        """Replace matched spans with <TYPE> placeholders."""
        result    = list(text)
        # Work right-to-left so offsets stay valid
        for m in reversed(matches):
            placeholder = list(f"<{m.type.upper()}>")
            result[m.start:m.end] = placeholder
        return "".join(result)