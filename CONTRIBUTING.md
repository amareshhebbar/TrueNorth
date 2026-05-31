# Contributing to TrueNorth

Thank you for taking the time to contribute. TrueNorth is built in public and every contribution matters — whether it's fixing a typo, adding a goal package, or building a new delivery channel.

---

## Before you start

Read the [README](README.md) and [SETUP.md](SETUP.md) so you understand what TrueNorth is and how to run it locally.

Join the [Discord](https://discord.gg/truenorth) — it's the fastest way to ask questions and get feedback before you spend time on something.

---

## Ways to contribute

### 1. Goal packages (highest impact)

The goal marketplace is the heart of TrueNorth's community. If you've built a good YAML for a specific domain — mental health intake, legal aid intake, insurance claim, school admission, anything — publish it.

```bash
# Write your goal YAML
# Test it locally
python -c "
from truenorth import TrueNorthEngine, YAMLLoader
import asyncio

async def test():
    e = TrueNorthEngine(goal_config=YAMLLoader.load('my_goal.yaml'))
    await e.start()
    # run a few test conversations
    
asyncio.run(test())
"

# Publish to the registry
truenorth publish my_goal.yaml
```

Goal packages live in `packages/goals/`. PRs welcome.

Good goals have:
- Clear required fields with sensible types and ranges
- A question for every field that reads naturally
- At least one conditional field (makes conversations feel smart)
- A follow_up block for long-horizon use cases
- Tested against at least 20 example conversations

### 2. Delivery channel adapters

`truenorth/scheduler/delivery.py` has adapters for WhatsApp, Email, and SMS. Missing channels:
- Telegram
- Signal
- Slack (for HR/internal tools)
- RCS (rich messages on Android)
- FCM push notifications
- LINE (Southeast Asia)

Each adapter is ~50 lines. Add it to `delivery.py` and write tests in `tests/unit/test_phases_5_6_7.py`.

### 3. Language support

TrueNorth auto-detects conversation language and can respond in any language the underlying LLM supports. But some things are hardcoded in English:
- Default question templates
- System prompts to the LLM
- Error messages returned to the user

If you're fluent in Hindi, Tamil, Kannada, Telugu, Bengali, Marathi, or any other regional language: translations of the base prompts would make TrueNorth dramatically better for those users.

### 4. Benchmarks

The benchmarks in the README are honest but limited. We need:
- Domain-specific evaluation datasets (medical intake, legal intake, HR)
- Multilingual extraction accuracy tests
- Comparison against LangChain + GPT-4o on real tasks
- Latency benchmarks across cloud providers

If you have labelled data or can run evaluations, open a discussion first.

### 5. Bug fixes and improvements

Look for issues labelled `good first issue` or `help wanted` on GitHub. If you find a bug not reported yet, open an issue before writing code.

---

## Development workflow

### Setup

```bash
git clone https://github.com/truenorth-ai/truenorth.git
cd truenorth/packages/core
pip install -e ".[dev]"
```

### Make your change

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### Write tests

Every new feature needs tests. Every bug fix needs a test that would have caught it.

Tests live in `packages/core/tests/unit/` and follow the naming pattern `test_<module>.py`.

```bash
# Run tests while you work
python -m pytest tests/unit/test_your_module.py --asyncio-mode=auto -v -x
```

We use:
- `pytest` for all tests
- `pytest-asyncio` for async tests
- `unittest.mock` for mocking
- No real network calls in unit tests

### Check your work

```bash
# Full test suite must pass
python -m pytest tests/ --asyncio-mode=auto -q
# Expected: all passing

# Syntax check
python -m py_compile truenorth/your_module.py
```

### Commit style

We use conventional commits:

```
feat(scheduler): add Telegram delivery adapter
fix(engine): handle empty user message gracefully
docs(readme): update benchmark table with new numbers
test(observability): add sector parity tests for health monitor
refactor(llm): extract pricing table to separate module
```

Prefix options: `feat` `fix` `docs` `test` `refactor` `perf` `ci`

### Pull request

- One PR per change
- Link the issue it closes (`Closes #123`)
- Include a short description of what changed and why
- Update `CHANGELOG.md` if it's a user-visible change
- Keep PRs focused — don't bundle unrelated changes

---

## Code style

- Python: `ruff` for linting, `black` for formatting (88 char line length)
- TypeScript: `prettier` + `eslint`
- Go: `gofmt`

```bash
# Format Python
black truenorth/ tests/
ruff check truenorth/ --fix

# Format TypeScript  
cd packages/sdk-node && npx prettier --write src/

# Format Go
cd packages/sdk-go && gofmt -w .
```

---

## Adding a new LLM provider

1. Create `truenorth/llm/your_provider.py` implementing `LLMBase`
2. Add models to `truenorth/llm/pricing.py`
3. Register the provider in `truenorth/llm/router.py`
4. Add tests in `tests/unit/test_llm.py`
5. Add the provider to the README table

The `LLMBase` interface requires:
- `generate(messages, system, max_tokens, temperature) → LLMResponse`
- `generate_stream(messages, ...) → AsyncIterator[StreamChunk]`
- `health_check() → bool`

---

## What we won't accept

- Breaking changes to the YAML schema without a migration path
- Tests that make real network calls (mock them)
- New dependencies without discussion first (we keep deps minimal)
- Code that removes or weakens the hallucination firewall
- Compliance features that reduce privacy protections
- Goal packages that collect data without clear user benefit

---

## Questions?

Open a GitHub Discussion or ask in Discord. We're happy to help.