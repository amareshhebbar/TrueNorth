# TrueNorth

> Build AI agents that actually work for real humans.

TrueNorth is a conversation-first agent framework. While other frameworks solve the *technical* problem of calling LLMs, TrueNorth solves the *product* problem: collecting data reliably, handling edge cases gracefully, protecting user privacy, and getting better over time.

## What's inside

```
packages/
├── core/       Python engine + FastAPI server  (start here)
├── studio/     React/Next.js admin UI
├── sdk-node/   TypeScript / Node.js SDK
├── sdk-go/     Go SDK
└── specs/      Shared YAML schemas + OpenAPI contract
```

## Quick start

```bash
# 1. Start the server
cd packages/core
cp .env.example .env          # add your LLM API keys
docker-compose up -d          # Postgres + Redis
poetry install
poetry run alembic upgrade head
poetry run uvicorn truenorth.api.main:app --reload

# 2. Try the dry-run CLI (zero API calls)
poetry run truenorth dry-run examples/goals/fitness_plan.yaml \
  --scenario tests/fixtures/scenarios/fitness_happy_path.json

# 3. Node.js
cd packages/sdk-node && npm install && npm run build
```

## The core idea

Define your agent entirely in YAML:

```yaml
goal_id: fitness_plan
persona: "You are a friendly fitness coach."

required_fields:
  - name: age
    type: integer
  - name: weight
    type: float
  - name: fitness_goal
    type: enum
    values: [weight_loss, muscle_gain, endurance]
  - name: injuries
    type: text
    optional: true

output:
  format: structured_report
```

That YAML is your entire agent. TrueNorth handles the conversation, extraction, validation, and output generation.

## Docs

See `docs/` for architecture, YAML reference, and deployment guides.
# TrueNorth
