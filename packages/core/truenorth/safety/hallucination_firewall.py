"""

HallucinationFirewall — the single biggest differentiator from LangChain/CrewAI.

Every LLM output passes through this firewall before the user sees it.
Each factual claim in the output is verified against the collected session fields.
Any claim that cannot be traced to a real collected value is flagged or blocked.

Architecture — three stages:
  Stage 1: ClaimExtractor   — parse output into discrete verifiable claims
  Stage 2: ClaimVerifier    — verify each claim against collected fields
                               (rule-based first, LLM supervisor for ambiguous)
  Stage 3: OutputSanitiser  — reconstruct clean output with hallucinations removed

Claim types:
  FIELD_REFERENCE  — directly names a collected field value ("You weigh 65 kg")
  DERIVED          — mathematically calculated from fields (BMI = weight/height²)
  GENERIC_ADVICE   — general guidance with no specific field value ("Stay hydrated")
  HALLUCINATION    — contradicts or invents a field value ("You weigh 80 kg" when 65)

Verdicts per claim:
  VERIFIED        — matches collected field exactly
  DERIVED_PASS    — correct derivation from collected fields
  GENERIC_PASS    — generic advice, no field reference needed
  LOW_CONFIDENCE  — probable hallucination, flagged but allowed through with warning
  BLOCKED         — certain hallucination, removed from output

Session-level verdicts:
  CLEAN           — all claims verified, output safe to send
  FLAGGED         — some low-confidence claims, output sent with audit note
  BLOCKED         — critical hallucinations found, output regenerated or rejected

Usage:
    firewall = HallucinationFirewall(router=llm_router)
    result = await firewall.check(
        output="Your fitness plan for Alex (age 28, weight 65 kg)...",
        collected_fields={"name": "Alex", "age": 28, "weight_kg": 65},
        fields_config=goal_config["fields"],
        session_id="sess-123",
    )
    if result.is_safe:
        send_to_user(result.safe_output)
    else:
        regenerate_output()
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.llm.router import LLMRouter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class ClaimType(str, Enum):
    FIELD_REFERENCE = "field_reference"   # directly cites a field value
    DERIVED         = "derived"           # computed from fields (BMI, etc.)
    GENERIC_ADVICE  = "generic_advice"    # no specific value, general guidance
    UNKNOWN         = "unknown"           # cannot classify


class ClaimVerdict(str, Enum):
    VERIFIED        = "verified"          # exact match to collected field
    DERIVED_PASS    = "derived_pass"      # mathematically correct derivation
    GENERIC_PASS    = "generic_pass"      # generic — no field to verify against
    LOW_CONFIDENCE  = "low_confidence"    # probable hallucination (flag, allow)
    BLOCKED         = "blocked"           # certain hallucination (remove)


class FirewallVerdict(str, Enum):
    CLEAN   = "clean"    # all claims pass, safe to send as-is
    FLAGGED = "flagged"  # some low-confidence claims, send with audit note
    BLOCKED = "blocked"  # critical hallucination, do not send


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawClaim:
    """One sentence/phrase from the LLM output, before verification."""
    text:        str
    claim_type:  ClaimType
    field_refs:  List[str]       # field names this claim appears to reference
    values_seen: List[str]       # numeric/text values extracted from the claim
    start_char:  int             # position in original output (for replacement)
    end_char:    int


@dataclass
class VerifiedClaim:
    """A RawClaim with its verification result attached."""
    raw:           RawClaim
    verdict:       ClaimVerdict
    traced_field:  Optional[str]    
    expected_val:  Optional[Any]    
    found_val:     Optional[str]    # what the claim says
    confidence:    float            # 0.0–1.0 confidence in this verdict
    reason:        str              # human-readable explanation

    def to_dict(self) -> dict:
        return {
            "text":         self.raw.text,
            "verdict":      self.verdict.value,
            "traced_field": self.traced_field,
            "expected":     self.expected_val,
            "found":        self.found_val,
            "confidence":   round(self.confidence, 3),
            "reason":       self.reason,
        }


@dataclass
class FirewallResult:
    """Complete firewall result for one LLM output."""
    session_id:     str
    verdict:        FirewallVerdict
    safe_output:    str                    # cleaned output (hallucinations removed)
    original_output: str
    claims:         List[VerifiedClaim]
    blocked_count:  int
    flagged_count:  int
    verified_count: int
    latency_ms:     int
    supervisor_used: bool                  # did we call the LLM supervisor?
    audit_log:      List[dict]             # structured log for compliance

    @property
    def is_safe(self) -> bool:
        return self.verdict in (FirewallVerdict.CLEAN, FirewallVerdict.FLAGGED)

    @property
    def hallucination_rate(self) -> float:
        total = len(self.claims)
        if total == 0:
            return 0.0
        bad = self.blocked_count + self.flagged_count
        return round(bad / total, 3)

    def summary(self) -> str:
        lines = [
            f"FirewallResult [{self.verdict.value.upper()}]",
            f"  Claims:   {len(self.claims)} total  "
            f"({self.verified_count} verified, "
            f"{self.flagged_count} flagged, "
            f"{self.blocked_count} blocked)",
            f"  Latency:  {self.latency_ms}ms",
            f"  Supervisor used: {self.supervisor_used}",
        ]
        if self.blocked_count:
            lines.append("  ⚠ BLOCKED claims:")
            for c in self.claims:
                if c.verdict == ClaimVerdict.BLOCKED:
                    lines.append(f"    - {c.raw.text[:80]!r}")
                    lines.append(f"      traced={c.traced_field}  "
                                 f"expected={c.expected_val!r}  found={c.found_val!r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "verdict":          self.verdict.value,
            "is_safe":          self.is_safe,
            "blocked_count":    self.blocked_count,
            "flagged_count":    self.flagged_count,
            "verified_count":   self.verified_count,
            "hallucination_rate": self.hallucination_rate,
            "latency_ms":       self.latency_ms,
            "supervisor_used":  self.supervisor_used,
            "claims":           [c.to_dict() for c in self.claims],
            "audit_log":        self.audit_log,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1: ClaimExtractor
# ─────────────────────────────────────────────────────────────────────────────

class ClaimExtractor:
    """
    Splits LLM output into discrete verifiable claims.

    Approach:
      - Split on sentence boundaries
      - Identify which sentences contain field-like values (numbers, names)
      - Classify each sentence as field_reference / derived / generic_advice
      - Extract the specific values mentioned

    This is purely rule-based — fast, zero LLM calls.
    """
    _FIELD_CLAIM_SIGNALS = re.compile(
        r"\b(your|you are|you weigh|you have|you're|based on your|"
        r"at \d+|aged \d+|age of \d+|weight of \d+|height of \d+|"
        r"you said|you mentioned|per your|according to your)\b",
        re.IGNORECASE,
    )
    _GENERIC_SIGNALS = re.compile(
        r"\b(should|recommend|suggest|consider|try|aim for|make sure|"
        r"important to|remember to|keep in mind|generally|typically|"
        r"most people|studies show|research suggests|it is important)\b",
        re.IGNORECASE,
    )
    _DERIVED_SIGNALS = re.compile(
        r"\b(bmi|body mass index|tdee|bmr|basal metabolic|"
        r"calculated|based on these|therefore your|"
        r"this gives|which means your|equates to)\b",
        re.IGNORECASE,
    )
    _NUMBER_RE = re.compile(
        r"(\d+(?:\.\d+)?)\s*"
        r"(kg|lbs?|lb|cm|m|ft|inches?|in|years?|days?|weeks?|months?|"
        r"minutes?|hours?|kcal|calories?|%|km|miles?|reps?|sets?)?",
        re.IGNORECASE,
    )

    def extract(self, text: str, fields_config: Dict[str, dict]) -> List[RawClaim]:
        """
        Extract verifiable claims from output text.

        Args:
            text:          LLM output to analyse
            fields_config: Field specs from goal YAML

        Returns:
            List of RawClaim objects, one per sentence/clause
        """
        sentences = self._split_sentences(text)
        claims    = []
        pos       = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            start = text.find(sentence, pos)
            end   = start + len(sentence)
            pos   = end

            claim_type = self._classify(sentence)
            field_refs = self._find_field_references(sentence, fields_config)
            values     = self._extract_values(sentence)

            claims.append(RawClaim(
                text        = sentence,
                claim_type  = claim_type,
                field_refs  = field_refs,
                values_seen = values,
                start_char  = max(start, 0),
                end_char    = end,
            ))

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split on sentence boundaries, preserving meaningful units."""
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\-*#])", text)
        result = []
        for part in parts:
            sub = re.split(r"\n[-*•]\s+|\n\d+\.\s+", part)
            result.extend(s.strip() for s in sub if s.strip())
        return result if result else [text]

    def _classify(self, sentence: str) -> ClaimType:
        """Classify a sentence into one of the four claim types."""
        if self._DERIVED_SIGNALS.search(sentence):
            return ClaimType.DERIVED
        if self._FIELD_CLAIM_SIGNALS.search(sentence):
            return ClaimType.FIELD_REFERENCE
        if self._GENERIC_SIGNALS.search(sentence):
            return ClaimType.GENERIC_ADVICE
        if self._NUMBER_RE.search(sentence):
            return ClaimType.FIELD_REFERENCE
        return ClaimType.GENERIC_ADVICE

    def _find_field_references(
        self,
        sentence:      str,
        fields_config: Dict[str, dict],
    ) -> List[str]:
        """
        Find which field names / labels appear explicitly in this sentence.
        Requires the actual field name or label keyword — NOT just generic
        pronouns like 'your' or 'you'. This prevents false positives where
        a sentence about workout duration gets matched to the 'age' field.
        """
        sentence_lower = sentence.lower()
        refs = []
        for field_name, cfg in fields_config.items():
            label  = cfg.get("label", field_name.replace("_", " ")).lower()
            name_tokens  = [t for t in field_name.replace("_", " ").lower().split() if len(t) > 3]
            label_tokens = [t for t in label.split() if len(t) > 3]
            all_tokens   = list(set(name_tokens + label_tokens))
            if field_name.lower() in sentence_lower:
                refs.append(field_name)
                continue
            if label and label in sentence_lower:
                refs.append(field_name)
                continue

            # ALL meaningful tokens of the field name/label appear in the sentence
            if all_tokens and all(t in sentence_lower for t in all_tokens):
                refs.append(field_name)
                continue

        return refs

    def _extract_values(self, sentence: str) -> List[str]:
        """Extract numeric values and quoted strings from a sentence."""
        values = []
        for m in self._NUMBER_RE.finditer(sentence):
            val  = m.group(1)
            unit = m.group(2) or ""
            values.append(f"{val}{unit}")
        # Also extract quoted strings
        for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', sentence):
            values.append(m.group(1) or m.group(2))
        return values


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2: ClaimVerifier
# ─────────────────────────────────────────────────────────────────────────────

class ClaimVerifier:
    """
    Verifies each RawClaim against collected session fields.

    Two-stage verification:
      Stage A (rule-based, always runs): numeric/string matching against fields
      Stage B (LLM supervisor, only for ambiguous): semantic verification

    Stage B is only invoked when:
      - Stage A finds a potential field reference but can't confirm the value
      - The claim is a DERIVED type (need to check the math)
      - The claim confidence from Stage A is below SUPERVISOR_THRESHOLD
    """

    SUPERVISOR_THRESHOLD: float = 0.55
    _FIELD_CLAIM_SIGNALS = re.compile(
        r"\b(your|you are|you weigh|you have|you\'re|based on your|"
        r"you can|you plan|you work|you train|you will|"
        r"at \d+|aged \d+|age of \d+|weight of \d+|height of \d+|"
        r"you said|you mentioned|per your|according to your)\b",
        re.IGNORECASE,
    )

    NUMERIC_TOLERANCE: float = 0.02

    HALLUCINATION_TOLERANCE: float = 0.08   # 8% deviation = hallucination

    def __init__(self, router: Optional["LLMRouter"] = None):
        self._router = router
        self._supervisor_calls = 0

    async def verify_all(
        self,
        claims:           List[RawClaim],
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        session_id:       str,
    ) -> Tuple[List[VerifiedClaim], bool]:
        """
        Verify all claims. Returns (verified_claims, supervisor_was_used).
        """
        verified       = []
        supervisor_used = False

        for claim in claims:
            result = self._rule_verify(claim, collected_fields, fields_config)

            # Call LLM supervisor for ambiguous claims
            if (
                self._router is not None
                and result.confidence < self.SUPERVISOR_THRESHOLD
                and claim.claim_type != ClaimType.GENERIC_ADVICE
            ):
                llm_result = await self._supervisor_verify(
                    claim, collected_fields, fields_config, session_id
                )
                if llm_result is not None:
                    result = llm_result
                    supervisor_used = True
                    self._supervisor_calls += 1

            verified.append(result)
            logger.debug(
                "firewall: session=%s claim=%r verdict=%s confidence=%.2f traced=%s",
                session_id,
                claim.text[:60],
                result.verdict.value,
                result.confidence,
                result.traced_field,
            )

        return verified, supervisor_used

    # ------------------------------------------------------------------
    # Stage A: Rule-based verification
    # ------------------------------------------------------------------

    def _rule_verify(
        self,
        claim:            RawClaim,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
    ) -> VerifiedClaim:
        """
        Verify a claim using deterministic rules. Fast, no API calls.

        Algorithm for FIELD_REFERENCE claims with numeric values:
          1. Extract all numbers from the claim sentence
          2. For each number, find the BEST matching field across ALL collected fields
             (prefer explicitly mentioned fields, then all numeric fields)
          3. If a number matches a field exactly (within 2%) → VERIFIED
          4. If a number is close to a field (within 8%) but not exact → LOW_CONFIDENCE
          5. If a number is way off from ALL field values → BLOCKED
          6. Text fields: check if expected value string appears in claim
        """

        if claim.claim_type == ClaimType.GENERIC_ADVICE:
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.GENERIC_PASS,
                traced_field=None, expected_val=None, found_val=None,
                confidence=0.95, reason="Generic advice — no field value to verify",
            )

        if claim.claim_type == ClaimType.DERIVED:
            return self._verify_derived(claim, collected_fields, fields_config)

        if not claim.field_refs and not claim.values_seen:
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.GENERIC_PASS,
                traced_field=None, expected_val=None, found_val=None,
                confidence=0.75, reason="No specific field value referenced",
            )

        numeric_fields: Dict[str, float] = {}
        for fn, fv in collected_fields.items():
            ftype = fields_config.get(fn, {}).get("type", "text")
            if ftype in ("integer", "number", "float"):
                try:
                    numeric_fields[fn] = float(str(fv).replace(",", ""))
                except (ValueError, TypeError):
                    pass

        # ── Step 1: Check numeric values against all fields ──────────────────
        if claim.values_seen:
            best_num_result = self._best_numeric_match(
                claim, numeric_fields, fields_config,
                priority_fields=claim.field_refs,
            )
            if best_num_result is not None:
                return best_num_result

        # ── Step 2: Check text fields that were explicitly referenced ─────────
        for field_name in claim.field_refs:
            if field_name in collected_fields:
                ftype = fields_config.get(field_name, {}).get("type", "text")
                if ftype not in ("integer", "number", "float"):
                    result = self._match_text(claim, field_name, collected_fields[field_name])
                    if result.verdict != ClaimVerdict.LOW_CONFIDENCE or len(claim.field_refs) == 1:
                        return result

        # ── Step 3: Field ref detected but nothing matched ────────────────────
        if claim.field_refs:
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.LOW_CONFIDENCE,
                traced_field=claim.field_refs[0] if claim.field_refs else None,
                expected_val=None, found_val=None,
                confidence=0.45,
                reason="Field referenced but expected value not confirmed in claim",
            )

        return VerifiedClaim(
            raw=claim, verdict=ClaimVerdict.GENERIC_PASS,
            traced_field=None, expected_val=None, found_val=None,
            confidence=0.65, reason="No verifiable field values found in claim",
        )

    def _best_numeric_match(
        self,
        claim:           RawClaim,
        numeric_fields:  Dict[str, float],  
        fields_config:   Dict[str, dict],
        priority_fields: List[str],           # fields explicitly named in claim
    ) -> Optional["VerifiedClaim"]:
        """
        For each numeric value in the claim, find the best matching field and
        return the verdict for the MOST SUSPICIOUS number found.

        Returns None if no numeric values are present.
        """
        if not claim.values_seen or not numeric_fields:
            return None

        worst_result: Optional[VerifiedClaim] = None   

        for val_str in claim.values_seen:
            # Parse the number (strip units)
            num_str = re.sub(r"[a-zA-Z%,]", "", val_str).strip()
            try:
                found_num = float(num_str)
            except (ValueError, TypeError):
                continue

            candidate_fields = list(priority_fields) + [
                fn for fn in numeric_fields if fn not in priority_fields
            ]

            # Check explicitly referenced fields first
            # Note: do NOT early-return even on verified match — other values may be hallucinated
            for fn in priority_fields:
                if fn not in numeric_fields:
                    continue
                expected = numeric_fields[fn]
                diff = abs(found_num - expected) / max(abs(expected), 1e-9)
                if diff <= self.NUMERIC_TOLERANCE:
                    r = VerifiedClaim(
                        raw=claim, verdict=ClaimVerdict.VERIFIED,
                        traced_field=fn,
                        expected_val=expected, found_val=val_str,
                        confidence=0.97,
                        reason=f"{found_num} matches field '{fn}'={expected}",
                    )
                    if worst_result is None or _verdict_severity(r.verdict) > _verdict_severity(worst_result.verdict):
                        worst_result = r
                    break  

            # Check all numeric fields — find closest match
            best_diff  = float("inf")
            best_field = None
            for fn, expected in numeric_fields.items():
                diff = abs(found_num - expected) / max(abs(expected), 1e-9)
                if diff < best_diff:
                    best_diff  = diff
                    best_field = fn

            if best_field is None:
                continue

            best_expected = numeric_fields[best_field]
            r: Optional[VerifiedClaim] = None

            if best_diff <= self.NUMERIC_TOLERANCE:
                # Exact match to some field — verified
                r = VerifiedClaim(
                    raw=claim, verdict=ClaimVerdict.VERIFIED,
                    traced_field=best_field,
                    expected_val=best_expected, found_val=val_str,
                    confidence=0.93,
                    reason=f"{found_num} matches field '{best_field}'={best_expected}",
                )
            elif best_diff <= self.HALLUCINATION_TOLERANCE:
                # Close but not exact — suspicious
                r = VerifiedClaim(
                    raw=claim, verdict=ClaimVerdict.LOW_CONFIDENCE,
                    traced_field=best_field,
                    expected_val=best_expected, found_val=val_str,
                    confidence=0.40,
                    reason=f"{found_num} differs from '{best_field}'={best_expected} by {best_diff:.1%}",
                )
            else:
                # This number doesn't match any collected field closely.
                # Only BLOCK if the claim clearly references a specific field value
                if claim.field_refs or self._FIELD_CLAIM_SIGNALS.search(claim.text):
                    r = VerifiedClaim(
                        raw=claim, verdict=ClaimVerdict.BLOCKED,
                        traced_field=best_field,
                        expected_val=best_expected, found_val=val_str,
                        confidence=0.88,
                        reason=(
                            f"HALLUCINATION: {found_num} does not match any collected field "
                            f"(closest: '{best_field}'={best_expected}, diff={best_diff:.1%})"
                        ),
                    )

            if r is not None:
                if worst_result is None:
                    worst_result = r
                elif _verdict_severity(r.verdict) > _verdict_severity(worst_result.verdict):
                    worst_result = r

        return worst_result

    def _match_field(
        self,
        claim:            RawClaim,
        field_name:       str,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
    ) -> VerifiedClaim:
        """Try to verify one specific field reference in the claim."""
        collected_val = collected_fields[field_name]
        field_cfg     = fields_config.get(field_name, {})
        ftype         = field_cfg.get("type", "text")

        if ftype in ("integer", "number", "float"):
            return self._match_numeric(claim, field_name, collected_val)

        return self._match_text(claim, field_name, collected_val)

    def _match_numeric(
        self,
        claim:         RawClaim,
        field_name:    str,
        expected:      Any,
    ) -> VerifiedClaim:
        """Verify a numeric field claim."""
        try:
            expected_num = float(str(expected).replace(",", ""))
        except (ValueError, TypeError):
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.GENERIC_PASS,
                traced_field=field_name, expected_val=expected, found_val=None,
                confidence=0.60, reason="Cannot parse expected value as number",
            )

        for val_str in claim.values_seen:
            try:
                num_str = re.sub(r"[a-zA-Z%]", "", val_str).strip()
                found_num = float(num_str)
            except (ValueError, TypeError):
                continue

            diff = abs(found_num - expected_num) / max(abs(expected_num), 1e-9)

            if diff <= self.NUMERIC_TOLERANCE:
                return VerifiedClaim(
                    raw=claim, verdict=ClaimVerdict.VERIFIED,
                    traced_field=field_name,
                    expected_val=expected_num, found_val=val_str,
                    confidence=0.97,
                    reason=f"Numeric value {found_num} matches field '{field_name}'={expected_num}",
                )
            elif diff <= self.HALLUCINATION_TOLERANCE:
                return VerifiedClaim(
                    raw=claim, verdict=ClaimVerdict.LOW_CONFIDENCE,
                    traced_field=field_name,
                    expected_val=expected_num, found_val=val_str,
                    confidence=0.40,
                    reason=(
                        f"Value {found_num} differs from collected "
                        f"'{field_name}'={expected_num} by {diff:.1%}"
                    ),
                )
            else:
                return VerifiedClaim(
                    raw=claim, verdict=ClaimVerdict.BLOCKED,
                    traced_field=field_name,
                    expected_val=expected_num, found_val=val_str,
                    confidence=0.92,
                    reason=(
                        f"HALLUCINATION: claim says {found_num} but "
                        f"'{field_name}' is {expected_num} "
                        f"(difference: {diff:.1%})"
                    ),
                )

        return VerifiedClaim(
            raw=claim, verdict=ClaimVerdict.LOW_CONFIDENCE,
            traced_field=field_name,
            expected_val=expected, found_val=None,
            confidence=0.50,
            reason=f"Field '{field_name}' referenced but no numeric value to verify",
        )

    def _match_text(
        self,
        claim:      RawClaim,
        field_name: str,
        expected:   Any,
    ) -> VerifiedClaim:
        """Verify a text/categorical field claim."""
        expected_str  = str(expected).strip().lower()
        claim_lower   = claim.text.lower()

        if not expected_str:
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.GENERIC_PASS,
                traced_field=field_name, expected_val=expected, found_val=None,
                confidence=0.65, reason="Empty expected value",
            )

        if expected_str in claim_lower:
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.VERIFIED,
                traced_field=field_name,
                expected_val=expected, found_val=expected_str,
                confidence=0.93,
                reason=f"Value {expected!r} found verbatim for field '{field_name}'",
            )

        words     = set(expected_str.split())
        in_claim  = sum(1 for w in words if w in claim_lower and len(w) > 3)
        if words and in_claim / len(words) > 0.5:
            return VerifiedClaim(
                raw=claim, verdict=ClaimVerdict.VERIFIED,
                traced_field=field_name,
                expected_val=expected, found_val=f"partial:{in_claim}/{len(words)}",
                confidence=0.78,
                reason=f"Partial match for field '{field_name}' ({in_claim}/{len(words)} words)",
            )

        return VerifiedClaim(
            raw=claim, verdict=ClaimVerdict.LOW_CONFIDENCE,
            traced_field=field_name,
            expected_val=expected, found_val=None,
            confidence=0.42,
            reason=f"Field '{field_name}' referenced but expected value not found in claim",
        )

    def _verify_derived(
        self,
        claim:            RawClaim,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
    ) -> VerifiedClaim:
        """
        Verify derived/calculated values (BMI, TDEE, calorie targets, etc.).
        Checks known formulas against collected fields.
        """
        claim_lower = claim.text.lower()

        if "bmi" in claim_lower or "body mass index" in claim_lower:
            weight = collected_fields.get("weight_kg")
            height = collected_fields.get("height_cm")
            if weight is not None and height is not None:
                bmi_expected = float(weight) / ((float(height) / 100) ** 2)
                for val_str in claim.values_seen:
                    try:
                        num_str  = re.sub(r"[a-zA-Z%]", "", val_str).strip()
                        bmi_found = float(num_str)
                        diff      = abs(bmi_found - bmi_expected) / max(bmi_expected, 1)
                        if diff <= 0.05:   # 5% tolerance for BMI
                            return VerifiedClaim(
                                raw=claim, verdict=ClaimVerdict.DERIVED_PASS,
                                traced_field="weight_kg+height_cm",
                                expected_val=round(bmi_expected, 1),
                                found_val=val_str,
                                confidence=0.95,
                                reason=f"BMI {bmi_found:.1f} matches calculated {bmi_expected:.1f}",
                            )
                        else:
                            return VerifiedClaim(
                                raw=claim, verdict=ClaimVerdict.BLOCKED,
                                traced_field="weight_kg+height_cm",
                                expected_val=round(bmi_expected, 1),
                                found_val=val_str,
                                confidence=0.90,
                                reason=(
                                    f"HALLUCINATION: BMI stated as {bmi_found:.1f} "
                                    f"but calculated as {bmi_expected:.1f}"
                                ),
                            )
                    except (ValueError, TypeError):
                        continue

        return VerifiedClaim(
            raw=claim, verdict=ClaimVerdict.DERIVED_PASS,
            traced_field=None, expected_val=None, found_val=None,
            confidence=0.65,
            reason="Derived value — formula not verified (cannot auto-check)",
        )

    # ------------------------------------------------------------------
    # Stage B: LLM Supervisor verification
    # ------------------------------------------------------------------

    _SUPERVISOR_SYSTEM = """\
You are a factual accuracy auditor for an AI health and fitness coach.
Your job is to verify whether a statement in the AI's output is accurate
given the user's collected profile data.

You must respond ONLY with valid JSON. No explanation, no preamble.
"""

    async def _supervisor_verify(
        self,
        claim:            RawClaim,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        session_id:       str,
    ) -> Optional[VerifiedClaim]:
        """
        Call the LLM supervisor to verify ambiguous claims.
        Uses claude-sonnet (highest quality) for accuracy.
        """
        from truenorth.llm.base import Message
        from truenorth.llm.router import TASK_OUTPUT  # use highest-quality model

        profile_str = json.dumps(
            {k: v for k, v in collected_fields.items()},
            default=str,
        )

        prompt = f"""User profile (GROUND TRUTH — do not question these values):
{profile_str}

AI output sentence to verify:
"{claim.text}"

Task: Determine if this sentence accurately reflects the user's profile.
Check:
1. Are all specific numbers/values in the sentence correct based on the profile?
2. Are all specific facts (name, age, weight, etc.) correct?
3. Does the sentence contradict anything in the profile?

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "verdict": "VERIFIED" | "LOW_CONFIDENCE" | "BLOCKED",
  "confidence": 0.0-1.0,
  "issue": "brief explanation if LOW_CONFIDENCE or BLOCKED, else null",
  "traced_field": "field_name or null",
  "expected_value": "what the profile says or null",
  "found_value": "what the claim says or null"
}}"""

        try:
            resp = await self._router.generate(
                task       = TASK_OUTPUT,
                messages   = [Message(role="user", content=prompt)],
                system     = self._SUPERVISOR_SYSTEM,
                max_tokens  = 200,
                temperature = 0.0,   # deterministic — this is safety-critical
                model       = "claude-sonnet-4-20250514",  # always use best model
            )

            raw = resp.content.strip()
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(raw)

            verdict_map = {
                "VERIFIED":        ClaimVerdict.VERIFIED,
                "LOW_CONFIDENCE":  ClaimVerdict.LOW_CONFIDENCE,
                "BLOCKED":         ClaimVerdict.BLOCKED,
            }
            verdict = verdict_map.get(
                data.get("verdict", "LOW_CONFIDENCE"),
                ClaimVerdict.LOW_CONFIDENCE,
            )

            return VerifiedClaim(
                raw           = claim,
                verdict       = verdict,
                traced_field  = data.get("traced_field"),
                expected_val  = data.get("expected_value"),
                found_val     = data.get("found_value"),
                confidence    = float(data.get("confidence", 0.7)),
                reason        = f"[supervisor] {data.get('issue') or 'Verified by LLM supervisor'}",
            )

        except Exception as e:
            logger.warning(
                "firewall: supervisor call failed for session=%s: %s",
                session_id, e,
            )
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_severity(verdict: ClaimVerdict) -> int:
    """Numeric severity for comparing verdicts. Higher = more severe."""
    return {
        ClaimVerdict.GENERIC_PASS:   0,
        ClaimVerdict.DERIVED_PASS:   0,
        ClaimVerdict.VERIFIED:       0,
        ClaimVerdict.LOW_CONFIDENCE: 1,
        ClaimVerdict.BLOCKED:        2,
    }.get(verdict, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3: OutputSanitiser
# ─────────────────────────────────────────────────────────────────────────────

class OutputSanitiser:
    """
    Reconstructs the output with hallucinated claims removed.

    BLOCKED claims: replaced with a safe placeholder based on the real field value.
    LOW_CONFIDENCE claims: kept but marked in the audit log (not shown to user).
    VERIFIED / GENERIC_PASS: kept as-is.
    """

    def sanitise(
        self,
        original_output: str,
        verified_claims: List[VerifiedClaim],
        collected_fields: Dict[str, Any],
    ) -> str:
        """Remove BLOCKED claims and fix obvious hallucinations."""
        output = original_output

        blocked = [
            c for c in verified_claims
            if c.verdict == ClaimVerdict.BLOCKED
        ]
        # Sort by position descending so we edit end-to-start
        blocked.sort(key=lambda c: c.raw.start_char, reverse=True)

        for claim in blocked:
            replacement = self._build_replacement(claim, collected_fields)
            # Find the claim text in output and replace it
            start = claim.raw.start_char
            end   = claim.raw.end_char
            if start < len(output) and output[start:end].strip():
                output = output[:start] + replacement + output[end:]
            else:
                # Fallback: simple string replace (handles position drift)
                if claim.raw.text in output:
                    output = output.replace(claim.raw.text, replacement, 1)

        return output.strip()

    @staticmethod
    def _build_replacement(
        claim:            VerifiedClaim,
        collected_fields: Dict[str, Any],
    ) -> str:
        """Build a corrected replacement for a blocked hallucination."""
        if claim.traced_field and claim.traced_field in collected_fields:
            field_label = claim.traced_field.replace("_", " ")
            real_val    = collected_fields[claim.traced_field]
            return f"[your {field_label}: {real_val}]"
        # Can't correct it — remove entirely
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Main: HallucinationFirewall
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationFirewall:
    """
    Production-grade hallucination firewall for TrueNorth.

    Every LLM output passes through here before being shown to the user.
    Integrates with the engine via output/generator.py.

    Usage:
        firewall = HallucinationFirewall(router=llm_router)

        # After OutputGenerator.generate():
        result = await firewall.check(
            output          = generated_text,
            collected_fields = state.collected_fields,
            fields_config   = state.fields_config,
            session_id      = state.session_id,
        )

        if result.is_safe:
            return result.safe_output
        else:
            # Critical hallucination — regenerate or reject
            return fallback_output(collected_fields)
    """

    BLOCK_THRESHOLD: int = 1

    def __init__(
        self,
        router:          Optional["LLMRouter"] = None,
        block_threshold: int   = 1,
        config:          Optional[dict] = None,
    ):
        self._router          = router
        self._block_threshold = block_threshold
        self._config          = config or {}
        self._extractor       = ClaimExtractor()
        self._verifier        = ClaimVerifier(router=router)
        self._sanitiser       = OutputSanitiser()

        # Metrics (process-lifetime)
        self._total_checks:   int = 0
        self._total_blocked:  int = 0
        self._total_flagged:  int = 0

    async def check(
        self,
        output:           str,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        session_id:       str = "unknown",
    ) -> FirewallResult:
        """
        Main firewall entry point. Run this on every LLM output before delivery.

        Args:
            output:           Raw LLM output text
            collected_fields: {field_name: value} from session state
            fields_config:    Field specifications from goal YAML
            session_id:       Session ID for logging

        Returns:
            FirewallResult — contains safe_output, verdict, and full audit log
        """
        t0 = time.perf_counter()
        self._total_checks += 1

        logger.info(
            "firewall: checking output session=%s chars=%d fields=%d",
            session_id, len(output), len(collected_fields),
        )

        if not output or not output.strip():
            return self._pass_result(output, [], session_id, 0, False)

        # Stage 1: Extract claims
        claims = self._extractor.extract(output, fields_config)
        logger.debug(
            "firewall: extracted %d claims from output", len(claims)
        )

        if not claims:
            return self._pass_result(output, [], session_id, 0, False)

        verified, supervisor_used = await self._verifier.verify_all(
            claims, collected_fields, fields_config, session_id
        )

        blocked_claims  = [c for c in verified if c.verdict == ClaimVerdict.BLOCKED]
        flagged_claims  = [c for c in verified if c.verdict == ClaimVerdict.LOW_CONFIDENCE]
        verified_claims = [
            c for c in verified
            if c.verdict in (
                ClaimVerdict.VERIFIED,
                ClaimVerdict.DERIVED_PASS,
                ClaimVerdict.GENERIC_PASS,
            )
        ]

        self._total_blocked += len(blocked_claims)
        self._total_flagged += len(flagged_claims)

        if len(blocked_claims) >= self._block_threshold:
            safe_output = self._sanitiser.sanitise(output, verified, collected_fields)
            verdict     = FirewallVerdict.BLOCKED
            logger.warning(
                "firewall: BLOCKED session=%s blocked=%d output_preview=%r",
                session_id, len(blocked_claims), output[:100],
            )
        elif flagged_claims:
            safe_output = output   # allow through but log
            verdict     = FirewallVerdict.FLAGGED
            logger.info(
                "firewall: FLAGGED session=%s flagged=%d",
                session_id, len(flagged_claims),
            )
        else:
            safe_output = output
            verdict     = FirewallVerdict.CLEAN

        latency_ms = int((time.perf_counter() - t0) * 1000)

        audit_log = self._build_audit_log(
            session_id, verdict, verified, collected_fields, latency_ms
        )

        result = FirewallResult(
            session_id      = session_id,
            verdict         = verdict,
            safe_output     = safe_output,
            original_output = output,
            claims          = verified,
            blocked_count   = len(blocked_claims),
            flagged_count   = len(flagged_claims),
            verified_count  = len(verified_claims),
            latency_ms      = latency_ms,
            supervisor_used = supervisor_used,
            audit_log       = audit_log,
        )

        logger.info(
            "firewall: done session=%s verdict=%s blocked=%d flagged=%d "
            "verified=%d latency=%dms supervisor=%s",
            session_id, verdict.value,
            len(blocked_claims), len(flagged_claims),
            len(verified_claims), latency_ms, supervisor_used,
        )

        return result

    async def check_conversation_turn(
        self,
        agent_response:   str,
        collected_fields: Dict[str, Any],
        fields_config:    Dict[str, dict],
        session_id:       str,
    ) -> str:
        """
        Lightweight check for mid-conversation agent responses (not final output).
        Less strict than full check — only blocks definite hallucinations.
        Returns the safe response text.
        """
        result = await self.check(
            output           = agent_response,
            collected_fields = collected_fields,
            fields_config    = fields_config,
            session_id       = session_id,
        )

        if result.verdict == FirewallVerdict.BLOCKED:
            # Don't block the conversation — just return a safe fallback
            logger.warning(
                "firewall: mid-conversation hallucination blocked, using fallback. "
                "session=%s", session_id
            )
            return result.safe_output or "Let me continue with the next question."

        return result.safe_output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pass_result(
        output:          str,
        claims:          list,
        session_id:      str,
        latency_ms:      int,
        supervisor_used: bool,
    ) -> FirewallResult:
        return FirewallResult(
            session_id       = session_id,
            verdict          = FirewallVerdict.CLEAN,
            safe_output      = output,
            original_output  = output,
            claims           = claims,
            blocked_count    = 0,
            flagged_count    = 0,
            verified_count   = 0,
            latency_ms       = latency_ms,
            supervisor_used  = supervisor_used,
            audit_log        = [],
        )

    @staticmethod
    def _build_audit_log(
        session_id:       str,
        verdict:          FirewallVerdict,
        verified_claims:  List[VerifiedClaim],
        collected_fields: Dict[str, Any],
        latency_ms:       int,
    ) -> List[dict]:
        log = []
        for c in verified_claims:
            if c.verdict in (ClaimVerdict.BLOCKED, ClaimVerdict.LOW_CONFIDENCE):
                log.append({
                    "session_id":    session_id,
                    "timestamp":     time.time(),
                    "claim":         c.raw.text[:200],
                    "verdict":       c.verdict.value,
                    "traced_field":  c.traced_field,
                    "expected":      str(c.expected_val)[:100] if c.expected_val else None,
                    "found":         str(c.found_val)[:100] if c.found_val else None,
                    "confidence":    c.confidence,
                    "reason":        c.reason[:200],
                    "latency_ms":    latency_ms,
                })
        return log

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        """Return lifetime metrics for this firewall instance."""
        return {
            "total_checks":  self._total_checks,
            "total_blocked": self._total_blocked,
            "total_flagged": self._total_flagged,
            "supervisor_calls": self._verifier._supervisor_calls,
            "block_rate": (
                round(self._total_blocked / self._total_checks, 4)
                if self._total_checks else 0.0
            ),
        }