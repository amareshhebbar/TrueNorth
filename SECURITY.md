# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x (current) | ✅ |
| < 0.1 | ❌ |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **security@truenorth.ai** with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (optional)

We will acknowledge within 48 hours and aim to ship a fix within 7 days for critical issues.

## Scope

### In scope
- Authentication bypass (API key validation, JWT verification)
- PII leakage through the hallucination firewall
- Prompt injection that bypasses the safety pipeline
- DPDP/GDPR compliance violations in the compliance managers
- Rate limiter bypass allowing DoS
- Budget guard bypass allowing unlimited spend

### Out of scope
- Issues in LLM providers themselves (report to Anthropic, Google, OpenAI)
- Social engineering
- Physical attacks
- Issues requiring physical access to the server

## Security design

### What TrueNorth does to protect user data

**PII detection before LLM calls.** Stage 3 of the engine pipeline detects and masks PII (Aadhaar numbers, phone numbers, email addresses, financial data) before any text is sent to a cloud LLM. When mobile/on-device routing is configured, extraction tasks run entirely on-device — PII never leaves the user's phone.

**API keys never stored raw.** All API keys are stored as `sha256(raw_key)`. The raw key is shown once at creation. Even if the database is compromised, raw keys cannot be recovered.

**JWT verification.** JWTs use HMAC-SHA256. Tokens include expiry. Signature verification uses constant-time comparison to prevent timing attacks.

**Budget hard stops.** The budget guard enforces cost caps at the API layer, before the engine processes any request. A compromised API key cannot run up unlimited LLM costs.

**Audit log.** Every consent grant, withdrawal, rights request, and PII detection is logged to the compliance audit trail. Logs are append-only.

### Responsible disclosure

We follow responsible disclosure. If you report a valid vulnerability, we will:
- Credit you in the release notes (unless you prefer anonymity)
- Not pursue legal action for good-faith security research
- Keep you informed of our progress fixing the issue