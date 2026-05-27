"""PII detection and redaction for free-text fields."""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "indian_mobile": (r"\b[6-9]\d{9}\b", "phone number"),
    "email":         (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
    "aadhaar":       (r"\b\d{4}\s\d{4}\s\d{4}\b", "Aadhaar number"),
    "pan":           (r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", "PAN number"),
    "credit_card":   (r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b", "credit card"),
    "passport_in":   (r"\b[A-Z]{1}[0-9]{7}\b", "passport number"),
}


@dataclass
class PIIScanResult:
    original: str
    redacted: str
    found: list[str]   # list of PII type names found

    @property
    def has_pii(self) -> bool:
        return len(self.found) > 0


class PIIDetector:
    def scan(self, text: str, field_name: str = "") -> PIIScanResult:
        found = []
        redacted = text

        for pii_type, (pattern, label) in PII_PATTERNS.items():
            if re.search(pattern, text):
                found.append(label)
                redacted = re.sub(pattern, f"[{label.upper()} REDACTED]", redacted)

        if found:
            logger.warning(
                "PII detected in field '%s': %s — redacted from logs",
                field_name, ", ".join(found)
            )

        return PIIScanResult(original=text, redacted=redacted, found=found)

    def redact_for_logging(self, text: str) -> str:
        result = self.scan(text)
        return result.redacted
