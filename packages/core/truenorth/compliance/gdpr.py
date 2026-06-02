"""
GDPR (EU General Data Protection Regulation) compliance.

Mirrors the DPDP manager but with EU-specific terminology and rules.
Differences from DPDP:
  - Legal bases: consent, legitimate interest, contract, vital interest, legal obligation
  - DPO (Data Protection Officer) designation
  - Data breach notification (72-hour rule)
  - Supervisory authority reporting
  - Stricter cross-border transfer rules (SCCs, adequacy decisions)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GDPRLegalBasis(str, Enum):
    CONSENT              = "consent"
    CONTRACT             = "contract"
    LEGAL_OBLIGATION     = "legal_obligation"
    VITAL_INTEREST       = "vital_interest"
    PUBLIC_TASK          = "public_task"
    LEGITIMATE_INTEREST  = "legitimate_interest"


class DataSubjectRight(str, Enum):
    ACCESS              = "access"
    RECTIFICATION       = "rectification"
    ERASURE             = "erasure"           
    RESTRICT            = "restrict_processing"
    PORTABILITY         = "data_portability"
    OBJECT              = "object"
    AUTOMATED_DECISION  = "automated_decision_opt_out"


@dataclass
class GDPRConsentRecord:
    """GDPR consent record."""
    record_id:    str
    user_id:      str
    session_id:   str
    legal_basis:  GDPRLegalBasis
    purpose:      str
    controller:   str           # data controller name
    processor:    Optional[str] = None   # data processor if different
    categories:   List[str]     = field(default_factory=list)
    consent_text: str           = ""
    ip_address:   Optional[str] = None
    timestamp:    float         = field(default_factory=time.time)
    withdrawn_at: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.withdrawn_at is None

    def to_dict(self) -> dict:
        return {
            "record_id":   self.record_id,
            "user_id":     self.user_id,
            "session_id":  self.session_id,
            "legal_basis": self.legal_basis.value,
            "purpose":     self.purpose,
            "controller":  self.controller,
            "is_active":   self.is_active,
            "timestamp":   self.timestamp,
        }


class GDPRManager:
    """
    GDPR compliance manager for TrueNorth.
    """

    def __init__(
        self,
        controller:   str,
        dpo_email:    Optional[str]   = None,
        legal_basis:  GDPRLegalBasis  = GDPRLegalBasis.CONSENT,
        retention_days: int           = 365,
    ):
        self._controller  = controller
        self._dpo         = dpo_email
        self._legal_basis = legal_basis
        self._retention   = retention_days
        self._consents:   Dict[str, List[GDPRConsentRecord]] = {}
        self._audit_log:  List[dict] = []
        self._rights_log: List[dict] = []

    def privacy_notice(self, purpose: str = "") -> str:
        """Generate a GDPR Article 13/14 compliant privacy notice."""
        return (
            f"Privacy Notice (GDPR Art. 13)\n\n"
            f"Controller: {self._controller}\n"
            f"DPO: {self._dpo or 'N/A'}\n"
            f"Purpose: {purpose or 'Service provision'}\n"
            f"Legal basis: {self._legal_basis.value}\n"
            f"Retention: {self._retention} days\n\n"
            f"You have the right to: access, rectify, erase, restrict processing, "
            f"data portability, and to object. Contact {self._dpo or self._controller}."
        )

    def grant_consent(
        self,
        user_id:      str,
        session_id:   str,
        purpose:      str           = "",
        categories:   Optional[List[str]] = None,
        ip_address:   Optional[str] = None,
        consent_text: str           = "",
    ) -> GDPRConsentRecord:
        record = GDPRConsentRecord(
            record_id   = str(uuid.uuid4())[:12],
            user_id     = user_id,
            session_id  = session_id,
            legal_basis = self._legal_basis,
            purpose     = purpose,
            controller  = self._controller,
            categories  = categories or [],
            consent_text= consent_text,
            ip_address  = ip_address,
        )
        self._consents.setdefault(user_id, []).append(record)
        self._log("consent_granted", user_id, session_id)
        return record

    def withdraw_consent(self, user_id: str) -> bool:
        records = self._consents.get(user_id, [])
        changed = False
        for r in records:
            if r.is_active:
                r.withdrawn_at = time.time()
                changed = True
        if changed:
            self._log("consent_withdrawn", user_id, "")
        return changed

    def has_valid_consent(self, user_id: str) -> bool:
        return any(r.is_active for r in self._consents.get(user_id, []))

    def request_right(self, user_id: str, right: DataSubjectRight, details: str = "") -> str:
        """Register a data subject rights request. Returns request ID."""
        req_id = str(uuid.uuid4())[:12]
        self._rights_log.append({
            "request_id": req_id,
            "user_id":    user_id,
            "right":      right.value,
            "details":    details,
            "status":     "pending",
            "submitted":  time.time(),
        })
        self._log(f"right_requested:{right.value}", user_id, "")
        return req_id

    def audit_log(self, user_id: Optional[str] = None) -> List[dict]:
        if user_id:
            return [e for e in self._audit_log if e.get("user_id") == user_id]
        return list(self._audit_log)

    def _log(self, action: str, user_id: str, session_id: str) -> None:
        self._audit_log.append({
            "action":     action,
            "user_id":    user_id,
            "session_id": session_id,
            "timestamp":  time.time(),
        })