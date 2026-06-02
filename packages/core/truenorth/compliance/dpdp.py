"""
India Digital Personal Data Protection Act 2023 (DPDP) compliance.

This is the India moat — two-year head start over any Western framework.

DPDP requires:
  1. Explicit consent before collecting personal data
  2. Purpose limitation — data used only for declared purpose
  3. Data minimisation — collect only what's needed
  4. Data principal rights:
       - Right to access their data
       - Right to correction
       - Right to erasure ("right to be forgotten")
       - Right to grievance redressal
  5. Data fiduciary obligations (the app operator)
  6. Significant Data Fiduciary rules for sensitive data
  7. Cross-border transfer restrictions
  8. Audit log for all consent and data actions

Applicable to: any TrueNorth deployment processing personal data
of Indian residents — healthcare, HR, finance, legal, fitness.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConsentStatus(str, Enum):
    PENDING   = "pending"
    GRANTED   = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED   = "expired"


class DataPrincipalRight(str, Enum):
    ACCESS    = "access"      # right to see their data
    CORRECT   = "correct"     # right to correct inaccurate data
    ERASE     = "erase"       # right to erasure
    GRIEVANCE = "grievance"   # right to raise a complaint
    NOMINATE  = "nominate"    # right to nominate another person


@dataclass
class ConsentRecord:
    """One consent event — creation, grant, or withdrawal."""
    record_id:        str
    user_id:          str
    session_id:       str
    purpose:          str
    data_fiduciary:   str
    status:           ConsentStatus
    data_categories:  List[str] = field(default_factory=list)
    consent_text:     str       = ""
    ip_address:       Optional[str] = None
    timestamp:        float      = field(default_factory=time.time)
    expiry_timestamp: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expiry_timestamp and time.time() > self.expiry_timestamp:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "record_id":       self.record_id,
            "user_id":         self.user_id,
            "session_id":      self.session_id,
            "purpose":         self.purpose,
            "data_fiduciary":  self.data_fiduciary,
            "status":          self.status.value,
            "data_categories": self.data_categories,
            "is_valid":        self.is_valid,
            "timestamp":       self.timestamp,
        }


@dataclass
class RightRequest:
    """A data principal rights request."""
    request_id:  str
    user_id:     str
    right:       DataPrincipalRight
    status:      str = "pending"    # pending / fulfilled / rejected
    details:     str = ""
    submitted_at: float = field(default_factory=time.time)
    fulfilled_at: Optional[float] = None


class DPDPManager:
    """
    DPDP Act 2023 compliance manager for TrueNorth.

    Every session involving Indian users should register consent
    before collecting personal data.
    """

    def __init__(
        self,
        data_fiduciary:  str,
        purpose:         str,
        retention_days:  int            = 90,
        sensitive_categories: List[str] = None,
    ):
        self._fiduciary   = data_fiduciary
        self._purpose     = purpose
        self._retention   = retention_days
        self._sensitive   = sensitive_categories or []
        self._consents:   Dict[str, List[ConsentRecord]] = {}  # user_id → records
        self._audit_log:  List[dict]  = []
        self._rights_log: List[RightRequest] = []

    # ------------------------------------------------------------------
    # Consent management
    # ------------------------------------------------------------------

    def consent_notice(self, categories: Optional[List[str]] = None) -> str:
        """
        Generate a DPDP-compliant consent notice in plain language.
        Show this to the user before collecting any personal data.
        """
        cats = categories or ["personal information"]
        cats_str = ", ".join(cats)
        expiry = f"{self._retention} days"
        return (
            f"Notice under the Digital Personal Data Protection Act, 2023:\n\n"
            f"Data Fiduciary: {self._fiduciary}\n"
            f"Purpose: {self._purpose}\n"
            f"Data collected: {cats_str}\n"
            f"Retention period: {expiry}\n\n"
            f"You have the right to access, correct, or erase your data at any time. "
            f"By proceeding, you grant consent for the above purpose only. "
            f"You may withdraw consent at any time."
        )

    def grant_consent(
        self,
        user_id:      str,
        session_id:   str,
        consent_text: str       = "",
        categories:   Optional[List[str]] = None,
        ip_address:   Optional[str] = None,
    ) -> ConsentRecord:
        """Record explicit consent grant."""
        expiry = time.time() + (self._retention * 86400)
        record = ConsentRecord(
            record_id       = str(uuid.uuid4())[:12],
            user_id         = user_id,
            session_id      = session_id,
            purpose         = self._purpose,
            data_fiduciary  = self._fiduciary,
            status          = ConsentStatus.GRANTED,
            data_categories = categories or [],
            consent_text    = consent_text,
            ip_address      = ip_address,
            expiry_timestamp = expiry,
        )
        self._consents.setdefault(user_id, []).append(record)
        self._audit("consent_granted", user_id, session_id, {"record_id": record.record_id})
        logger.info("dpdp: consent granted user=%s session=%s", user_id, session_id)
        return record

    def withdraw_consent(self, user_id: str, session_id: str = "") -> bool:
        """Record consent withdrawal."""
        records = self._consents.get(user_id, [])
        changed = False
        for r in records:
            if r.status == ConsentStatus.GRANTED:
                r.status = ConsentStatus.WITHDRAWN
                changed  = True
        if changed:
            self._audit("consent_withdrawn", user_id, session_id, {})
            logger.info("dpdp: consent withdrawn user=%s", user_id)
        return changed

    def has_valid_consent(self, user_id: str) -> bool:
        """Check if the user has a current valid consent."""
        return any(r.is_valid for r in self._consents.get(user_id, []))

    def get_latest_consent(self, user_id: str) -> Optional[ConsentRecord]:
        records = self._consents.get(user_id, [])
        return records[-1] if records else None

    # ------------------------------------------------------------------
    # Data principal rights
    # ------------------------------------------------------------------

    def request_access(self, user_id: str) -> RightRequest:
        """Data principal requests to see their data."""
        req = self._make_right_request(user_id, DataPrincipalRight.ACCESS)
        logger.info("dpdp: access request user=%s req=%s", user_id, req.request_id)
        return req

    def request_correction(self, user_id: str, details: str = "") -> RightRequest:
        """Data principal requests correction of inaccurate data."""
        req = self._make_right_request(user_id, DataPrincipalRight.CORRECT, details)
        return req

    def request_erasure(self, user_id: str) -> RightRequest:
        """Data principal requests erasure ("right to be forgotten")."""
        req = self._make_right_request(user_id, DataPrincipalRight.ERASE)
        self._audit("erasure_requested", user_id, "", {"request_id": req.request_id})
        logger.info("dpdp: erasure requested user=%s req=%s", user_id, req.request_id)
        return req

    def fulfill_erasure(self, request_id: str) -> bool:
        """Mark an erasure request as fulfilled."""
        for req in self._rights_log:
            if req.request_id == request_id:
                req.status      = "fulfilled"
                req.fulfilled_at = time.time()
                return True
        return False

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def audit_log(self, user_id: Optional[str] = None) -> List[dict]:
        """Return the full audit log, optionally filtered by user."""
        if user_id:
            return [e for e in self._audit_log if e.get("user_id") == user_id]
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_right_request(
        self, user_id: str, right: DataPrincipalRight, details: str = ""
    ) -> RightRequest:
        req = RightRequest(
            request_id = str(uuid.uuid4())[:12],
            user_id    = user_id,
            right      = right,
            details    = details,
        )
        self._rights_log.append(req)
        return req

    def _audit(self, action: str, user_id: str, session_id: str, extra: dict) -> None:
        self._audit_log.append({
            "action":     action,
            "user_id":    user_id,
            "session_id": session_id,
            "timestamp":  time.time(),
            **extra,
        })