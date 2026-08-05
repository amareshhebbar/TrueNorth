"""DPDP (India) and GDPR compliance."""
from truenorth.compliance.dpdp import DPDPManager, ConsentRecord, DataPrincipalRight
from truenorth.compliance.gdpr import (
    GDPRManager, GDPRConsentRecord, GDPRLegalBasis, DataSubjectRight,
)

__all__ = [
    "DPDPManager", "ConsentRecord", "DataPrincipalRight",
    "GDPRManager", "GDPRConsentRecord", "GDPRLegalBasis", "DataSubjectRight",
]
