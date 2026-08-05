"""
truenorth/safety/__init__.py
Safety layer — hallucination firewall, output verification, claim tracing.
"""
from truenorth.safety.hallucination_firewall import (
    HallucinationFirewall,
    FirewallResult,
    FirewallVerdict,
    VerifiedClaim,
    ClaimVerdict,
)

__all__ = [
    "HallucinationFirewall",
    "FirewallResult",
    "FirewallVerdict",
    "VerifiedClaim",
    "ClaimVerdict",
]
