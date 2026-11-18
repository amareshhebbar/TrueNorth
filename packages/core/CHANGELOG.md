# Changelog

## [0.2.0](https://github.com/amareshhebbar/TrueNorth/compare/truenorth-framework-v0.1.1...truenorth-framework-v0.2.0) (2026-11-18)


### Features

* **agents:** multi-agent layer — Orchestrator + Supervisor + 4 specialist agents (extract/validate/research/write), parallel+sequential workflows, 5-sector tests — 657 tests pass ([6a60eff](https://github.com/amareshhebbar/TrueNorth/commit/6a60effd6f53431a3ae138774b25a7d6f64681ea))
* **agents:** Phase 4 complete — A2A protocol bridge, LangGraph bidirectional integration, cross-goal state transfer + GoalChain — 891 tests pass ([7b25e96](https://github.com/amareshhebbar/TrueNorth/commit/7b25e96075ce747e1d985c601af2aff80e123f9f))
* complete core modules, CLI, and integration tests ([d55c192](https://github.com/amareshhebbar/TrueNorth/commit/d55c192dc70b50e8bd6788e50a386b94e5e53924))
* **core:** add memory persistence, GDPR compliance, SDK client, scheduler, and specialist agents ([0c12b51](https://github.com/amareshhebbar/TrueNorth/commit/0c12b518062723f2a26c263555e6e2de2d871acc))
* **intelligence:** harden conflict_detector — 7 types, severity, auto-resolve, aliases, unit normalisation, cross-field rules, ConflictStore — 275 tests pass ([6946eb2](https://github.com/amareshhebbar/TrueNorth/commit/6946eb2a1e5cd3a66559e3c5c564ed025d5edfaf))
* **llm,cli:** harden pricing.py (53 models, 8 providers) + CLI cost/pricing/estimate commands — 823 tests pass ([3f0f040](https://github.com/amareshhebbar/TrueNorth/commit/3f0f040150d6f550dad39d623524681ef03003d0))
* **llm:** harden cost_tracker — 3-tier budget enforcement, TurnCost/GoalCost/BudgetStatus types, projection engine, CSV/JSON export, alert callback, rate limit, 750 tests pass ([2e4cc83](https://github.com/amareshhebbar/TrueNorth/commit/2e4cc838f6a387a18d047593537cef2758384172))
* **llm:** harden router — fallback chain, retry+backoff, circuit breaker, budget guard, 6 new providers, stats — 441 tests pass ([d475d1f](https://github.com/amareshhebbar/TrueNorth/commit/d475d1f998815ff5343b5fb63eaafb078ddb6d03))
* **llm:** mobile_llm.py — iOS/Android on-device bridge, battery throttling, privacy routing (PII stays on device), 493 tests pass ([9692b86](https://github.com/amareshhebbar/TrueNorth/commit/9692b8606d615882f51aca6b9e7cc248e7e2552a))
* **mcp:** complete MCP stack — JSON-RPC 2.0 client, registry, tool executor, 3 built-in tools (calculator/datetime/web_search), Stage 13 engine wired — 571 tests pass ([e41ec5b](https://github.com/amareshhebbar/TrueNorth/commit/e41ec5b1acd6474fd9519b0163716b131cf6519b))
* Node.js/TS SDK, Go SDK, React Native/Expo SDK + hooks (useTrueNorthSession), Python SDK ([8e5b371](https://github.com/amareshhebbar/TrueNorth/commit/8e5b371c38e8c6a8faedb5689a108a3bc763c77e))
* **observability:** typed log streams, per-turn tracer, health monitor, A/B engine, cost dashboard REST API ([d8e60dd](https://github.com/amareshhebbar/TrueNorth/commit/d8e60ddffa4cfee407a28bc574d58406b278cf26))
* Redis rate limiter (3 dims), API key + JWT auth, budget guard (3 scopes), self-host docker-compose generator, goal marketplace (npm for AI agents) ([ca57744](https://github.com/amareshhebbar/TrueNorth/commit/ca5774421b1e4e14362780183cabd1e52cf2043d))
* Reminder AI (scheduler/delivery/planner), Memory (long_term/resume/vector_store), India DPDP + GDPR compliance, WhatsApp channel — 991 tests pass, 19,732 production lines ([2f7221b](https://github.com/amareshhebbar/TrueNorth/commit/2f7221baa8da5db65764c6396aa74bffd18a669d))
* **safety:** hallucination firewall — supervisor agent verifies every output ([c80c15a](https://github.com/amareshhebbar/TrueNorth/commit/c80c15a4e9624162f126086a79fa1158d9caec23))


### Bug Fixes

* **backend:** resolve pipeline, routing, and test failures ([bfb938b](https://github.com/amareshhebbar/TrueNorth/commit/bfb938b8a4b198b772054631d02e347e088d6118))
* **cli:** repair CLI command registry and pipeline integration test ([4334ffd](https://github.com/amareshhebbar/TrueNorth/commit/4334ffd9355c9ba56839764daa48fbcafac7369f))
* **cloud:** update self-host config generator for new env var schema ([fff4976](https://github.com/amareshhebbar/TrueNorth/commit/fff4976dc939f1381c8273f0562ecc0d378d8d03))
* **core:** add FieldValue TypedDict to graph_state — fixes API startup ImportError ([382fa08](https://github.com/amareshhebbar/TrueNorth/commit/382fa08cbab7c749018ae51ca1754f6c94467ca0))
* **core:** bump version to 0.1.8 — restore top-level exports ([75b4881](https://github.com/amareshhebbar/TrueNorth/commit/75b4881d7e361cbc6b545688c8cef3d72227a36f))
* **core:** restore top-level TrueNorthEngine and YAMLLoader exports ([4486963](https://github.com/amareshhebbar/TrueNorth/commit/448696341c36cd75a4b8c38763cb7396319084f9))
* **cost:** repair cost tracker session aggregation and per-turn logging ([31072c8](https://github.com/amareshhebbar/TrueNorth/commit/31072c868b5ea43faf9a4f3b66fa1b47f32abeb4))
* **llm:** correct gemini routing, update pricing table, fix router fallback chain ([73ec2e9](https://github.com/amareshhebbar/TrueNorth/commit/73ec2e9a598bb38ccd38685e6ba7bc4ff3c88a03))
* **testing:** repair dry runner session mock and pipeline integration harness ([4917ff6](https://github.com/amareshhebbar/TrueNorth/commit/4917ff6965312f719c5fe2a18c2a54acdac7e652))
* update test imports after package restructuring ([e4107c8](https://github.com/amareshhebbar/TrueNorth/commit/e4107c837a14c53a92ef1eaa807706ddee80976d))


### Documentation

* **core:** update .env.example with new vars and fix core README ([3b4c21a](https://github.com/amareshhebbar/TrueNorth/commit/3b4c21a02db1c8280e2ed1ae7965fddd0c170d41))
