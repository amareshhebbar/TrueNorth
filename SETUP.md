# Setup Guide

Everything you need to run TrueNorth locally, in Docker, or in production.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | 3.12 recommended |
| Node.js | 18+ | For Studio and Node SDK |
| Go | 1.21+ | For Go SDK only |
| Docker + Compose | Latest | For full-stack deployment |
| PostgreSQL | 15+ | Session storage (optional for local dev) |
| Redis | 7+ | Rate limiting + cost cache (optional for local dev) |

---

## Option 1 — Local dev (simplest)

No database or Redis needed. Everything runs in-memory.

```bash
# Clone the repo
git clone https://github.com/truenorth-ai/truenorth.git
cd truenorth

# Install the Python package
cd packages/core
pip install -e ".[dev]"

# Set at least one LLM API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export GEMINI_API_KEY=AIza...
# or
export OPENAI_API_KEY=sk-...

# Run the test suite to verify everything works
python -m pytest tests/ --asyncio-mode=auto -q
# Expected: 1258 passed

# Try the CLI
truenorth pricing --provider anthropic
truenorth estimate --model claude-haiku-4-5-20251001 --tokens 1000

# Run a goal interactively
python -c "
import asyncio
from truenorth import TrueNorthEngine, YAMLLoader

async def run():
    engine = TrueNorthEngine(goal_config=YAMLLoader.load('../../fitness_plan.yaml'))
    await engine.start()
    while True:
        resp = await engine.process_message(input('> '))
        print(resp.text)
        if resp.is_complete:
            print(resp.output)
            break

asyncio.run(run())
"
```

---

## Option 2 — Docker Compose (recommended for development)

Full stack: API + Postgres + Redis.

```bash
# From repo root
cp packages/core/.env.example packages/core/.env
# Edit .env — add your LLM API keys at minimum

docker compose -f docker-compose.dev.yml up -d

# Verify
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}

# Check logs
docker compose logs -f truenorth-api
```

### What's in docker-compose.dev.yml

```yaml
services:
  truenorth-api:   # FastAPI server on :8000
  postgres:        # Session + memory storage
  redis:           # Rate limiting + cost cache
```

---

## Option 3 — Self-host init (production)

Generates a production-ready docker-compose.yml with nginx, TLS, and a Celery worker for reminders.

```bash
pip install truenorth

# Generate deployment files
truenorth self-host init \
  --dir ./my-truenorth \
  --profile standard \
  --domain api.mycompany.com

cd my-truenorth
ls
# docker-compose.yml  .env.template  nginx.conf  README.md

# Configure
cp .env.template .env
nano .env  # fill in all CHANGE_ME values

# Deploy
docker compose up -d

# Verify
curl https://api.mycompany.com/health
```

### Profiles

| Profile | Services | Use case |
|---------|----------|----------|
| `minimal` | API + Postgres + Redis | Single server, internal use |
| `standard` | + nginx + Celery worker | Production, reminders enabled |
| `enterprise` | + Prometheus + Grafana | High-scale, full observability |

---

## Environment variables

### Required (at least one LLM key)

```bash
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...
```

### Optional but recommended

```bash
# Database (required for session persistence beyond in-memory)
DATABASE_URL=postgresql://truenorth:password@localhost:5432/truenorth

# Redis (required for rate limiting + distributed cost cache)
REDIS_URL=redis://:password@localhost:6379/0

# Security
TRUENORTH_JWT_SECRET=your-64-char-random-string
TRUENORTH_API_KEY=tn_live_your-initial-admin-key

# LLM routing overrides
TRUENORTH_MODEL_EXTRACT=gemini-1.5-flash
TRUENORTH_MODEL_CONVERSE=claude-haiku-4-5-20251001
TRUENORTH_MODEL_OUTPUT=claude-sonnet-4-20250514

# WhatsApp (for reminder delivery via WhatsApp)
WA_VERIFY_TOKEN=your-webhook-verify-token
WA_ACCESS_TOKEN=your-meta-access-token
WA_PHONE_NUMBER_ID=your-phone-number-id

# Email reminders (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASSWORD=your-app-password

# Compliance
COMPLIANCE_MODE=dpdp   # dpdp | gdpr | none
DATA_FIDUCIARY=Your Company Name
```

---

## Database setup

TrueNorth uses Alembic for migrations.

```bash
cd packages/core

# Run migrations
alembic upgrade head

# Create a new migration (after model changes)
alembic revision --autogenerate -m "add session metadata"
alembic upgrade head
```

### Schema overview

```sql
-- Sessions
CREATE TABLE sessions (
    session_id       TEXT PRIMARY KEY,
    goal_id          TEXT NOT NULL,
    user_id          TEXT,
    collected_fields JSONB,
    state            JSONB,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Long-term memory
CREATE TABLE user_facts (
    user_id    TEXT,
    fact_key   TEXT,
    value      JSONB,
    confidence FLOAT,
    goal_id    TEXT,
    updated_at TIMESTAMPTZ,
    PRIMARY KEY (user_id, fact_key)
);

-- Reminders
CREATE TABLE reminders (
    reminder_id TEXT PRIMARY KEY,
    session_id  TEXT,
    rule_id     TEXT,
    channel     TEXT,
    fire_at     TIMESTAMPTZ,
    status      TEXT,
    message     TEXT
);
```

---

## SDK setup

### Python SDK

```bash
pip install truenorth-sdk
```

```python
from truenorth_sdk import TrueNorth
tn = TrueNorth(api_key="tn_live_...", base_url="http://localhost:8000")
```

### Node.js / TypeScript SDK

```bash
npm install truenorth
# or
yarn add truenorth
# or
pnpm add truenorth
```

```typescript
import { TrueNorth } from 'truenorth'
const tn = new TrueNorth({ apiKey: process.env.TRUENORTH_API_KEY })
```

### Go SDK

```bash
go get github.com/truenorth-ai/truenorth-go
```

```go
import "github.com/truenorth-ai/truenorth-go"
client := truenorth.New(truenorth.Options{APIKey: os.Getenv("TRUENORTH_API_KEY")})
```

### React Native / Expo

```bash
npx expo install truenorth-rn
# or
npm install truenorth-rn
```

```tsx
import { useTrueNorthSession } from 'truenorth-rn'
```

---

## API authentication

All API requests require an API key in the `X-TrueNorth-Key` header.

### Create an API key

```bash
# Via CLI (after server is running)
truenorth keys create --label "My app" --plan pro

# Response:
# Key: tn_live_abc123...  (shown once — save it)
# ID:  key_8f4a2b...
```

### Use the key

```bash
curl http://localhost:8000/sessions \
  -H "X-TrueNorth-Key: tn_live_abc123..." \
  -H "Content-Type: application/json" \
  -d '{"goal_id": "fitness-coach"}'
```

### Studio dashboard (JWT)

The Studio dashboard uses JWT Bearer tokens. Issue one:

```bash
truenorth auth issue-token --user admin --tenant my-org
# Bearer eyJhbGc...
```

---

## Running tests

```bash
cd packages/core

# Full suite
python -m pytest tests/ --asyncio-mode=auto -q

# One phase
python -m pytest tests/unit/test_observability.py --asyncio-mode=auto -v

# Integration tests (requires running server + DB)
python -m pytest tests/integration/ --asyncio-mode=auto -v

# With coverage
python -m pytest tests/ --asyncio-mode=auto --cov=truenorth --cov-report=html
open htmlcov/index.html
```

---

## Troubleshooting

### "No LLM client registered for model"

You haven't set an API key for the model TrueNorth is trying to use.

```bash
# Check which model is being used
export TRUENORTH_LOG_LEVEL=debug

# Set the key for the model
export ANTHROPIC_API_KEY=sk-ant-...
```

### "Budget exceeded" on first call

The default budget in the YAML is too low for your setup.

```yaml
llm:
  budget_usd: 1.00   # increase this
```

### WhatsApp webhook not receiving messages

1. Verify your verify token matches `WA_VERIFY_TOKEN`
2. Confirm your webhook URL is publicly reachable (ngrok for dev)
3. Check Meta's webhook subscription is active in the developer console

```bash
# Test webhook locally with ngrok
ngrok http 8000
# Use the ngrok URL in Meta's webhook settings
```

### Reminders not firing

The Celery worker must be running:

```bash
celery -A truenorth.worker worker --loglevel=info -Q reminders
```

Or in Docker:
```bash
docker compose up truenorth-worker -d
```