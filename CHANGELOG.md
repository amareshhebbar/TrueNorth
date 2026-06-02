# Changelog

All notable changes to TrueNorth are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Postgres session store (currently in-memory)
- APScheduler integration for production reminder delivery
- Studio UI (Next.js YAML editor + analytics dashboard)
- iOS Swift SDK (native Foundation Models)
- Android Kotlin SDK (native Gemini Nano)
- Voice input (Whisper + ElevenLabs)
- Goal marketplace public launch

---

## [0.1.0] — 2025-06

### Added — Phase 0-4: Foundation + Intelligence + LLM + MCP + Multi-Agent

**Core engine (13-stage pipeline)**
- `TrueNorthEngine` — orchestrates the full conversation lifecycle
- YAML loader with schema validation
- `FieldTree` — conditional field branching (`if_true`, `if_value_is`, `if_numeric_gt`)
- Session manager — persist and resume conversations
- Conversation planner — decides what to ask next

**Intelligence layer**
- Hallucination firewall (1,243 lines) — 3-stage: ClaimExtractor → ClaimVerifier → OutputSanitiser
- Confidence scorer — 8-factor scoring per extracted field
- Conflict detector — 7 conflict types, semantic aliases, unit normalisation
- Source tracer — sentence → field → turn attribution
- Emotion detector — valence + arousal + shift detection
- Language detector — 50+ languages, auto-detect and respond in kind

**LLM infrastructure**
- Router with fallback chain and circuit breaker
- `MobileLLMClient` — iOS (Apple Intelligence) and Android (Gemini Nano) on-device bridge
- `LocalLLMClient` — Ollama and llama.cpp support
- `CostTracker` — 3-tier budget enforcement, per-turn cost rollup, projection engine
- Pricing table — 53 models across 8 providers

**MCP (Model Context Protocol)**
- JSON-RPC 2.0 client over stdio and SSE
- Tool registry and executor (Stage 13 of engine pipeline)
- Built-in tools: calculator, web search, datetime

**Multi-agent layer**
- `AgentOrchestrator` — capability routing, parallel + sequential workflows
- `AgentSupervisor` — OFF/LIGHT/STANDARD/STRICT quality control
- Specialist agents: extraction, validation, research, writer
- A2A protocol bridge — Google Agent-to-Agent standard
- LangGraph bridge — bidirectional integration
- State transfer + GoalChain — cross-goal memory

### Added — Phase 5-7: Reminder AI + Memory + Compliance

**Reminder & Async AI**
- `ReminderEngine` — evaluates follow_up rules, schedules reminders
- `MultiChannelDelivery` — WhatsApp, Email (SMTP), SMS (Twilio), Console
- `FollowUpPlanner` — LLM composes personalised reminder messages from session context

**Memory**
- `LongTermMemory` — persists user facts across sessions with confidence threshold
- `SessionResume` — load prior state and generate re-engagement message
- `VectorStore` — bag-of-words cosine similarity for semantic session search

**Compliance**
- `DPDPManager` — India DPDP Act 2023 (consent, rights, audit log)
- `GDPRManager` — EU GDPR (6 legal bases, 7 data subject rights)
- `WhatsAppChannel` — native inbound WhatsApp conversation channel

### Added — Phase 8: Observability

- 7 typed log streams: CONVERSATION, EXTRACTION, EMOTION, CONFLICT, COST, HALLUCINATION, COMPLIANCE
- `TrueNorthTracer` — per-turn structured event tracer with pluggable sinks
- `MemorySink`, `StdoutSink`, `CallbackSink`, `HTTPSink`
- `HealthMonitor` — completion rate, field skip rates, abandonment heatmap, p95 latency, alerts
- `ABEngine` — deterministic hash assignment, two-proportion z-test significance
- `ABRegistry` — manage multiple concurrent A/B tests
- `CostDashboard` — goal/session/trend/model analytics, FastAPI router

### Added — Phase 9: Production + Cloud

- `RateLimiter` — Redis sliding-window, 3 dimensions (api_key / goal / user), 4 plans
- `AuthMiddleware` — API key (SHA256 hash) + JWT (HS256) dual-scheme
- `APIKeyManager` — create, validate, revoke keys (raw key never stored)
- `BudgetGuard` — session + goal + tenant budget enforcement at API layer
- `SelfHostConfig` — generates docker-compose.yml, .env.template, nginx.conf, README.md
- `GoalRegistry` — publish, install, search goal packages; 6 official goals seeded
- CLI: `truenorth cost`, `truenorth pricing`, `truenorth estimate`, `truenorth self-host init`

### Added — Phase 10: REST API + SDKs

- FastAPI REST API: `/sessions`, `/goals`, `/analytics`
- Python SDK — sync (`TrueNorth`) + async (`AsyncTrueNorth`), stdlib only
- TypeScript SDK — works in Node.js, Bun, Deno, Next.js, React
- Go SDK — idiomatic Go, context-aware, full test coverage
- React Native / Expo SDK — `useTrueNorthSession` hook, `useRunSession`, bare client

### Stats
- 25,506 production lines
- 1,258 tests passing
- 0 test failures
- 105 Python modules