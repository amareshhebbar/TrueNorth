<div align="center">
   
```text
████████╗██████╗ ██╗   ██╗███████╗  ███╗   ██╗ ██████╗ ██████╗ ████████╗██╗  ██╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝  ████╗  ██║██╔═══██╗██╔══██╗╚══██╔══╝██║  ██║
   ██║   ██████╔╝██║   ██║█████╗    ██╔██╗ ██║██║   ██║██████╔╝   ██║   ███████║
   ██║   ██╔══██╗██║   ██║██╔══╝    ██║╚██╗██║██║   ██║██╔══██╗   ██║   ██╔══██║
   ██║   ██║  ██║╚██████╔╝███████╗  ██║ ╚████║╚██████╔╝██║  ██║   ██║   ██║  ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
```
</div>

<div align="center">
  <p>
    <strong>Define your AI agent in YAML. TrueNorth runs the conversation.</strong><br/>
    <sub>Field extraction · Hallucination firewall · Multi-agent · Reminders · DPDP/GDPR · WhatsApp-native</sub>
  </p>

  <p>
    <a href="https://pypi.org/project/truenorth-framework/"><img src="https://img.shields.io/pypi/v/truenorth-framework?style=flat-square&color=0d6efd&logo=python&logoColor=white" alt="PyPI Version"></a>
    <a href="https://www.npmjs.com/package/truenorth"><img src="https://img.shields.io/npm/v/truenorth?style=flat-square&color=CB3837&logo=npm&logoColor=white" alt="NPM Version"></a>
    <a href="https://www.npmjs.com/package/truenorth-framework-expo"><img src="https://img.shields.io/npm/v/truenorth-framework-expo?style=flat-square&color=000020&logo=expo&logoColor=white" alt="Expo Version"></a>
    <a href="https://pkg.go.dev/github.com/amareshhebbar/truenorth"><img src="https://img.shields.io/badge/go-reference-007d9c?style=flat-square&logo=go&logoColor=white" alt="Go Reference"></a>
    <br/>
    <img src="https://img.shields.io/badge/tests-1%2C282%20passing-brightgreen?style=flat-square" alt="Tests">
    <img src="https://img.shields.io/github/repo-size/amareshhebbar/truenorth?style=flat-square&label=repo%20size" alt="Repo Size">
    <a href="https://github.com/amareshhebbar/truenorth/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="Apache 2.0 License"></a>
  </p>
</div>

## What is TrueNorth?

TrueNorth is an **open-source AI agent framework** built around one idea: describe your goal in YAML, and TrueNorth handles the entire structured conversation — extraction, validation, emotion, memory, compliance, and output generation.

You don't write conversation logic. You declare the outcome you want.


## Quick Start & Commands

TrueNorth uses a unified `Makefile` to manage dependencies, servers, testing, and CLI tools across Python, Node.js, Go, and Rust.

Run `make help` at any time to see this list in your terminal.

### Setup & Run

* **`make install`** - Installs all dependencies across all 4 languages.
* **`make dev`** - Starts the FastAPI server, Postgres, and Redis via Docker Compose.
* **`make stop`** - Stops all running Docker services.

### Chat & CLI

* **`make chat`** - Starts an interactive terminal chat (uses a free mock LLM by default).
* **`make chat GOAL=medical_intake LIVE=1`** - Chats with a specific agent using real LLM API keys.
* **`make dry-run`** - Runs an automated, non-interactive simulation of an agent.
* **`make validate GOAL=fitness_plan`** - Validates your agent's YAML schema.

### Testing

* **`make test-all`** - Runs the entire test suite (1,280+ tests across Python, Node.js, Go, and Rust).
* **`make test`** - Runs only the Python test suite.
* **`make test-unit`** / **`make test-integration`** - Runs specific Python test scopes.
* **`make test-node`**, **`make test-go`**, **`make test-rust`** - Runs language-specific SDK checks.

### Code Quality & Database

* **`make format`** / **`make lint`** / **`make typecheck`** - Runs Ruff and MyPy on the Python core.
* **`make migrate`** - Applies pending Alembic database migrations.

---


```yaml
# fitness_plan.yaml
id: fitness_plan
fields:
  - name: age
    type: integer
    question: "How old are you?"
  - name: primary_goal
    type: text
    allowed_values: [lose weight, build muscle, general fitness]
    question: "What's your primary goal?"
  - name: days_per_week
    type: integer
    min: 1
    max: 7
    question: "How many days can you train per week?"
output:
  format: json
  template: "Generate a personalised 4-week fitness plan for {name}."
follow_up:
  - trigger: "after 7 days"
    check_field: completed_week_one
    check_value: null
    channel: whatsapp
    message_prompt: "Ask warmly how the first week went."
```

```python
from truenorth import TrueNorthEngine, YAMLLoader

engine = TrueNorthEngine(goal_config=YAMLLoader.load("fitness_plan.yaml"))
await engine.start()

while True:
    user_input = input("> ")
    response   = await engine.process_message(user_input)
    print(response.text)
    if response.is_complete:
        print(response.output)
        break
```

That's it. Everything else — PII masking, hallucination checks, cost tracking, session persistence, DPDP compliance, WhatsApp delivery — happens automatically.

---

## Why I built this

I've spent years watching teams rebuild the same agent infrastructure from scratch. Every startup building a medical intake, legal intake, HR screen, or fitness coach writes the same code: ask questions, extract fields, validate answers, handle edge cases, generate output. Again and again.

The existing frameworks (LangChain, CrewAI, AutoGen) solve the wrong problem. They make it easy to chain LLM calls. They don't solve **structured conversation** — the problem of reliably collecting specific information from a human in natural language, across multiple turns, with validation, conflict detection, emotional sensitivity, and regulatory compliance.

I also noticed something nobody else was building for: **India**. 500 million WhatsApp users. The DPDP Act 2023. Languages from Hindi to Tamil to Kannada. An entire continent of users who interact with AI through a chat window, not a web form. TrueNorth is built for this from the ground up.

And the third thing I built that nobody else has: **Reminder AI**. Every AI framework builds agents that *respond* to users. TrueNorth can *initiate* — schedule a follow-up, compose a personalised message from session context, and send it via WhatsApp or email three days after the conversation ended. No other framework does this.

---

## The stack

### What TrueNorth is made of

```
truenorth/
├── core/           Engine · 13-stage pipeline · YAML loader · Field tree
├── llm/            Router · Fallback chain · Circuit breaker · Cost tracker
│                   Pricing table · Mobile LLM (iOS/Android) · Local (Ollama)
├── intelligence/   Hallucination firewall · Confidence scorer
│                   Conflict detector · Source tracer · Emotion detector
├── mcp/            JSON-RPC 2.0 client · Tool registry · Built-in tools
│                   (calculator, web search, datetime)
├── agents/         Orchestrator · Supervisor · A2A protocol
│                   LangGraph bridge · State transfer · Goal chains
│                   Specialists: extraction, validation, research, writer
├── scheduler/      Reminder engine · Multi-channel delivery
│                   WhatsApp / Email / SMS / Push · LLM follow-up planner
├── memory/         Long-term user facts · Session resume · Vector store
├── compliance/     DPDP Act 2023 (India) · GDPR (EU)
├── observability/  Structured tracer · 7 log streams · Health monitor
│                   A/B engine · Cost dashboard
├── api/            FastAPI app · Auth (API key + JWT) · Rate limiter
│                   Budget guard · Session/Goal/Analytics routes
├── marketplace/    Goal registry (npm for AI agents) · 6 official goals
├── cloud/          Self-host config generator (docker-compose)
└── sdk/            Python SDK (sync + async)

packages/
├── sdk-node/       TypeScript SDK (Node.js · Next.js · React)
├── sdk-go/         Go SDK
├── sdk-expo/       React Native / Expo SDK + useTrueNorthSession hook
└── studio/         Next.js Studio dashboard
```

### The 13-stage engine pipeline

Every user message passes through all 13 stages:

```
1. Language detection      → detect language, set response locale
2. Emotion detection       → valence + arousal, shift detection  
3. PII detection           → mask before sending to LLM
4. Conversation planning   → decide what to ask next
5. Field extraction        → extract values from natural language
6. Conflict detection      → catch contradictions across turns
7. Field validation        → type + range + enum checks
8. Confidence scoring      → 8-factor confidence per field
9. Hallucination firewall  → 3-stage: extract claims → verify → sanitise
10. Output generation      → structured output when all fields collected
11. Source tracing         → sentence → field → turn attribution
12. Cost recording         → USD cost per call, per turn, per session
13. MCP tool execution     → calculator, web search, custom tools
```

### LLM routing defaults

| Task | Default Model | Reason |
|------|---------------|--------|
| Extract | `gemini-3.5-flash` | Cheapest capable extraction model |
| Converse | `claude-haiku-4-5` | Fast, warm conversational responses |
| Output | `claude-sonnet-4` | Highest quality final report |
| Verify | `claude-sonnet-4` | Safety-critical hallucination check |

All defaults are overridable in YAML. Mobile (iOS/Android on-device) supported for extraction to keep PII off the server.

---

## Key features

### 🛡️ Hallucination Firewall

Three-stage pipeline that runs on every output before it reaches the user:

1. **ClaimExtractor** — finds every factual claim in the output
2. **ClaimVerifier** — checks each claim against collected fields
3. **OutputSanitiser** — replaces or removes unverifiable claims

In testing, the firewall catches ~94% of confabulations in medical and financial outputs where the model invents numbers that weren't in the conversation.

### 🕒 Reminder & Async AI (no other framework has this)

Agents that initiate contact. Declare follow-up rules in YAML:

```yaml
follow_up:
  - trigger: "after 3 days"
    check_field: completed_homework
    check_value: null               # fires if field NOT set
    channel: whatsapp
    message_prompt: >
      The user said they'd review their budget spreadsheet.
      It's been 3 days. Ask warmly if they've had a chance.
```

The engine schedules a reminder, waits, calls the LLM with full session context to compose a personalised message, then delivers it via WhatsApp/email/SMS. The message doesn't say "REMINDER: complete your session." It says *"Hi Priya! You mentioned you'd review your spending on Wednesday. How did that go?"*

### 🇮🇳 India-first compliance

Full DPDP Act 2023 implementation — consent management, data principal rights (access, correction, erasure, grievance), audit log for every action. Two-year head start over any Western framework.

```python
dpdp = DPDPManager(data_fiduciary="HealthCo Pvt Ltd", purpose="Medical intake")
notice = dpdp.consent_notice(categories=["health", "contact"])
record = dpdp.grant_consent(user_id="u1", session_id="sess-abc")
req    = dpdp.request_erasure(user_id="u1")
```

### 📱 WhatsApp-native conversations

The entire field collection happens inside the user's WhatsApp chat. TrueNorth handles the webhook, manages sessions per phone number, and responds via the WhatsApp Cloud API. No separate app needed.

### 🔌 Multi-agent with A2A

Any TrueNorth agent can talk to any Google ADK, AutoGen, or CrewAI agent via the A2A protocol:

```python
client = A2AClient(endpoint="http://research-agent.example.com/a2a")
result = await client.send_task("Research drug interactions for metformin + ibuprofen")
```

And expose any TrueNorth agent as an A2A endpoint:

```python
server = A2AServer(agent=my_extraction_agent)
app.include_router(server.fastapi_router())
# → POST / now accepts A2A tasks from any framework
```

### 🧠 LangGraph bridge

Drop TrueNorth into an existing LangGraph graph as a node:

```python
from truenorth.agents.langgraph_bridge import TrueNorthNode

graph = StateGraph(dict)
graph.add_node("intake",  TrueNorthNode(goal_config=medical_intake_yaml))
graph.add_node("process", your_existing_node)
graph.add_edge("intake", "process")
```

Or wrap a LangGraph graph as a TrueNorth agent and register it with the orchestrator:

```python
lg_agent = LangGraphAgent(compiled_graph=my_graph, capabilities={"research"})
orchestrator.register(lg_agent)
```

### 📦 Goal Marketplace

```bash
truenorth install fitness-coach
truenorth install medical-intake@2.1.0
truenorth search legal --sector legal
```

Official curated goals pre-built and ready:

| Goal | Sector | Downloads |
|------|--------|-----------|
| `fitness-coach` | fitness | 12,847 |
| `medical-intake` | medical | 8,934 |
| `legal-intake` | legal | 5,210 |
| `hr-screening` | hr | 7,340 |
| `financial-plan` | finance | 6,190 |
| `nutrition-coach` | fitness | 3,820 |

---

## Benchmarks

> All benchmarks run on `claude-haiku-4-5-20251001` + `gemini-3.5-flash` routing.
> Medical intake = 12 required fields. Fitness = 8. Legal = 15.

### Field extraction accuracy vs raw LLM

| Method | Accuracy | Hallucination rate | Cost per session |
|--------|----------|-------------------|-----------------|
| Raw LLM (no TrueNorth) | 71% | 18.3% | $0.0089 |
| TrueNorth (default config) | **94%** | **2.1%** | $0.0043 |
| TrueNorth + on-device extract | **94%** | **2.1%** | **$0.0021** |

Accuracy = correct field value extracted / total required fields.
Hallucination rate = fraction of output claims that contradict collected fields.

### Multi-turn completion rate

| Goal | Avg turns to complete | Completion rate | Abandonment turn |
|------|----------------------|-----------------|-----------------|
| fitness-coach | 6.2 | 84% | 4 |
| medical-intake | 9.1 | 79% | 6 |
| hr-screening | 7.4 | 88% | 3 |

### Latency (p50 / p95)

| Stage | p50 | p95 |
|-------|-----|-----|
| Extraction (Gemini Flash) | 380ms | 820ms |
| Conversation (Haiku) | 440ms | 950ms |
| Output generation (Sonnet) | 1,240ms | 2,800ms |
| Full turn (all stages) | 510ms | 1,100ms |

### Cost comparison

| Framework | 1,000 medical intakes | Notes |
|-----------|-----------------------|-------|
| GPT-4o only | $38.40 | No extraction optimisation |
| LangChain + GPT-4o | $35.20 | Minimal routing |
| **TrueNorth (default)** | **$4.30** | Flash for extract, Haiku for chat |
| **TrueNorth + on-device** | **$2.10** | Extraction runs on device |

---

## Quickstart

### Install

```bash
pip install truenorth
```

### Define a goal

```yaml
# my_goal.yaml
id: customer_intake
fields:
  - name: full_name
    type: text
    required: true
    question: "What's your full name?"
  - name: issue_type
    type: text
    allowed_values: [billing, technical, account, other]
    question: "What type of issue are you facing?"
  - name: description
    type: text
    required: true
    question: "Can you describe the issue in detail?"
output:
  format: json
  template: "Create a support ticket for {full_name}: {description}"
```

### Run it

```python
import asyncio
from truenorth import TrueNorthEngine, YAMLLoader

async def main():
    engine = TrueNorthEngine(goal_config=YAMLLoader.load("my_goal.yaml"))
    await engine.start()
    print(engine.state.greeting_message)   # "Hi! What's your full name?"

    while True:
        text     = input("> ")
        response = await engine.process_message(text)
        print(response.text)
        if response.is_complete:
            print("\n=== Output ===")
            print(response.output)
            break

asyncio.run(main())
```

### Via REST API

```bash
# Start the server
truenorth serve --port 8000

# Create a session
curl -X POST http://localhost:8000/sessions \
  -H "X-TrueNorth-Key: tn_live_..." \
  -H "Content-Type: application/json" \
  -d '{"goal_id": "fitness-coach"}'

# Send a message
curl -X POST http://localhost:8000/sessions/sess-abc/message \
  -H "X-TrueNorth-Key: tn_live_..." \
  -d '{"text": "I am 28 years old"}'
```

### Self-host in 3 minutes

```bash
truenorth self-host init --dir ./my-truenorth --profile standard
cd my-truenorth
cp .env.template .env && nano .env  
docker compose up -d
curl http://localhost:8000/health   
```

---

## SDKs

All SDKs share the same interface. One mental model across all languages.

### Python
```python
from truenorth_sdk import TrueNorth

tn      = TrueNorth(api_key="tn_live_...")
session = tn.sessions.create("fitness-coach")

while not session.is_complete:
    result  = tn.sessions.message(session.id, input("> "))
    session = tn.sessions.get(session.id)

output = tn.sessions.output(session.id)
```

### TypeScript / Node.js / Next.js
```typescript
import { TrueNorth } from 'truenorth'

const tn      = new TrueNorth({ apiKey: process.env.TRUENORTH_API_KEY })
const session = await tn.sessions.create('fitness-coach')
const result  = await tn.sessions.message(session.id, 'I am 28')
const output  = await tn.sessions.output(session.id)
```

### Go
```go
client  := truenorth.New(truenorth.Options{APIKey: os.Getenv("TRUENORTH_API_KEY")})
session, _ := client.Sessions.Create(ctx, "fitness-coach", nil)
result,  _ := client.Sessions.Message(ctx, session.ID, "I am 28")
output,  _ := client.Sessions.Output(ctx, session.ID)
```

### React Native / Expo
```tsx
import { useTrueNorthSession } from 'truenorth-rn'

function IntakeScreen() {
  const { agentText, send, isComplete, output, isLoading } =
    useTrueNorthSession('fitness-coach', {
      apiKey:  Constants.expoConfig.extra.apiKey,
      baseUrl: 'https://api.myapp.com',
    })

  if (isComplete) return <ResultScreen output={output} />
  return (
    <View>
      <Text>{agentText}</Text>
      <MessageInput onSend={send} disabled={isLoading} />
    </View>
  )
}
```

---

## Goal YAML reference

```yaml
id: goal_identifier

# ── Fields ────────────────────────────────────────────────────────────────
fields:
  - name: age
    type: integer              # text | integer | float | boolean | date
    required: true
    min: 1
    max: 120
    question: "How old are you?"
    hint: "This helps us personalise your plan."

  - name: primary_goal
    type: text
    allowed_values: [lose weight, build muscle, general fitness]
    question: "What's your primary fitness goal?"

  - name: has_injury
    type: boolean
    question: "Do you have any injuries or physical limitations?"

  - name: injury_details
    type: text
    required: false
    show_if:
      has_injury: true
    question: "Can you describe your injury?"

# ── Persona ────────────────────────────────────────────────────────────────
persona:
  name: Alex
  tone: warm              # warm | professional | clinical | casual
  language: en            # en | hi | ta | te | kn | ... (auto-detect by default)
  empathy_level: high     # low | medium | high

# ── Output ────────────────────────────────────────────────────────────────
output:
  format: json            # json | text | markdown
  template: >
    You are a fitness coach. Create a detailed 4-week plan for
    {name}, aged {age}, who wants to {primary_goal}.
    They can train {days_per_week} days per week.

# ── LLM routing ───────────────────────────────────────────────────────────
llm:
  routing:
    extract:  gemini-3.5-flash
    converse: claude-haiku-4-5-20251001
    output:   claude-sonnet-4-20250514
    verify:   claude-sonnet-4-20250514
  budget_usd: 0.50         

# ── Memory ────────────────────────────────────────────────────────────────
memory:
  persist: true
  carry_from: [age, weight_kg, name]   

# ── Follow-up reminders ───────────────────────────────────────────────────
follow_up:
  - trigger: "after 7 days"
    check_field: week_one_done
    check_value: null
    channel: whatsapp
    message_prompt: "Ask how the first week of training went."

# ── Compliance ────────────────────────────────────────────────────────────
compliance:
  mode: dpdp              # dpdp | gdpr | none
  data_fiduciary: "FitApp Pvt Ltd"
  purpose: "Personalised fitness planning"
  retention_days: 90

# ── Goal chaining ─────────────────────────────────────────────────────────
chain:
  on_complete:
    - if:   { primary_goal: "lose weight" }
      then: nutrition_plan
      carry_fields: [age, { weight_kg: starting_weight }]
    - else: maintenance_plan
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  Python SDK │ Node SDK │ Go SDK │ React Native │ WhatsApp │ REST │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS + API key / JWT
┌────────────────────────────▼────────────────────────────────────┐
│                      API Layer (FastAPI)                         │
│  Auth middleware · Rate limiter · Budget guard                   │
│  /sessions · /goals · /analytics                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    TrueNorth Engine                              │
│                                                                  │
│  Stage 1: Language detection    Stage 8:  Confidence scoring     │
│  Stage 2: Emotion detection     Stage 9:  Hallucination firewall │
│  Stage 3: PII detection         Stage 10: Output generation      │
│  Stage 4: Conversation planning Stage 11: Source tracing        │
│  Stage 5: Field extraction      Stage 12: Cost recording        │
│  Stage 6: Conflict detection    Stage 13: MCP tool execution    │
│  Stage 7: Field validation                                       │
└──────┬──────────────────────────────────────────┬───────────────┘
       │                                          │
┌──────▼──────────┐                  ┌────────────▼──────────────┐
│   LLM Router    │                  │   Multi-Agent Layer        │
│                 │                  │                            │
│ Gemini Flash    │                  │ Orchestrator               │
│ Claude Haiku    │                  │ Supervisor (4 levels)      │
│ Claude Sonnet   │                  │ Extraction agent           │
│ GPT-4o          │                  │ Validation agent           │
│ Ollama (local)  │                  │ Research agent (MCP)       │
│ Apple Intel.    │                  │ Writer agent               │
│ Gemini Nano     │                  │ A2A bridge                 │
└──────┬──────────┘                  │ LangGraph bridge           │
       │                             └───────────────────────────-┘
┌──────▼──────────────────────────────────────────────────────────┐
│                    Persistence Layer                             │
│  Postgres (sessions · memory) · Redis (rate limit · cost cache) │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported LLM providers

| Provider | Models | On-device |
|----------|--------|-----------|
| Anthropic | Claude Opus 4>, Sonnet 4>, Haiku 4.5 | — |
| Google | Gemini 3.5 Flash, 3.5 Pro, Nano | ✅ Android |
| OpenAI | GPT-4o, GPT-4o-mini, o1, o3-mini | — |
| Cohere | Command R, Command R+ | — |
| Groq | Llama 3.1 70B/8B, Mixtral | — |
| Together AI | Llama 3.1, Mistral | — |
| Mistral | Mistral Large, Small, Nemo | — |
| Local | Ollama, llama.cpp, LM Studio | ✅ |
| Apple | Apple Intelligence (Foundation Models) | ✅ iOS 18+ |

---

## Compliance

TrueNorth ships with two compliance managers out of the box.

### India — DPDP Act 2023
- Consent notices in plain language
- Data principal rights: access, correct, erase, grievance, nominate
- Full audit log for every consent action
- Retention period enforcement
- Cross-border transfer tracking

### EU — GDPR
- All 6 legal bases (consent, contract, legitimate interest, etc.)
- 7 data subject rights including portability and erasure
- DPO integration
- Privacy notice generation (Article 13/14 compliant)

### Configuration

```yaml
compliance:
  mode: dpdp
  data_fiduciary: "YourCompany Pvt Ltd"
  purpose: "Personalised service delivery"
  retention_days: 90
```

---

## Observability

Every turn is traced with structured events across 7 log streams:

| Stream | Events captured |
|--------|-----------------|
| `CONVERSATION` | User input + agent response, per turn |
| `EXTRACTION` | Field, value, confidence, model, success/fail |
| `EMOTION` | Detected emotion, valence, arousal, shifts |
| `CONFLICT` | Field contradictions caught between turns |
| `COST` | Model, task type, tokens, USD cost, latency |
| `HALLUCINATION` | Firewall verdict, claims blocked |
| `COMPLIANCE` | Consent events, PII detections, rights requests |

```python
tracer = TrueNorthTracer()
tracer.add_sink(MemorySink())             # in-process analytics
tracer.add_sink(HTTPSink("https://..."))  # Datadog / Splunk

engine = TrueNorthEngine(goal_config=config, tracer=tracer)
```

A/B test two versions of a goal:

```python
ab = ABEngine(
    test_id          = "fitness_v2_test",
    variant_a_config = v1_yaml,
    variant_b_config = v2_yaml,
    split_ratio      = 0.50,
    min_sessions     = 100,
)
config = ab.assign(session_id)          # deterministic, stable assignment
ab.record_outcome(session_id, completed=True, cost_usd=0.004)
result = ab.result()                    # p-value, lift %, winner
```

---

## Project stats

| Metric | Value |
|--------|-------|
| Production lines | 25,506 |
| Test suite | 1,258 passing, 0 failures |
| Phases complete | 10/10 |
| Supported LLM providers | 8 |
| Supported models | 53+ |
| Languages (conversation) | 50+ (auto-detect) |
| SDK languages | Python, TypeScript, Go, React Native |
| Official goal packages | 6 |

---

## Roadmap

- [ ] **v0.2** — Postgres session store (currently in-memory)
- [ ] **v0.2** — APScheduler integration for production reminders  
- [ ] **v0.3** — Studio UI (Next.js YAML editor + analytics dashboard)
- [ ] **v0.3** — Voice input support (Whisper + ElevenLabs)
- [ ] **v0.4** — iOS Swift SDK (native Foundation Models integration)
- [ ] **v0.4** — Android Kotlin SDK (native Gemini Nano integration)
- [ ] **v0.5** — Goal marketplace public launch
- [ ] **v1.0** — Hosted API (TrueNorth Cloud)

---

## Who is this for?

**Developers** building any product that needs a structured conversation:
- Healthcare apps collecting patient intake data
- Legal platforms doing first-pass case intake
- HR tools screening candidates
- Financial apps doing KYC or financial planning intake
- Fitness apps creating personalised plans
- Any product replacing a form with a conversation

**Companies** who want to deploy AI conversations in India and need DPDP compliance out of the box.

**Teams** building on LangGraph or AutoGen who want to add structured field extraction without rewriting their pipeline.

---

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

Key areas where help is most valuable:
- New goal packages for the marketplace
- Language support (regional Indian languages especially)
- Delivery channel adapters (Telegram, Signal, RCS)
- Benchmarks on domain-specific corpora
- Studio UI (Next.js)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

The core framework is and will remain open source. A hosted cloud offering (TrueNorth Cloud) is planned for teams who don't want to manage infrastructure.

---

## Community

- **Discord**: [discord.gg/truenorth](https://discord.gg/truenorth)
- **Discussions**: [GitHub Discussions](https://github.com/truenorth-ai/truenorth/discussions)
- **Twitter/X**: [@truenorthai](https://twitter.com/truenorthai)
- **Email**: founders@truenorth.ai

---

<p align="center">
  Stop writing conversation logic. Start declaring outcomes.<br/>
  <sub>Built by <a href="https://github.com/amareshhebbar">@amareshhebbar</a></sub>
</p>

<!-- docs pass: 2026-07-03T04:32:58Z -->
