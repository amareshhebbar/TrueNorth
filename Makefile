# ─────────────────────────────────────────────────────────────────────────────
#  TrueNorth Makefile
#  Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

SHELL       := /bin/bash
CORE_DIR    := packages/core
GOAL        ?= fitness_plan
SESSION_ID  ?=
SCENARIO    ?=

.DEFAULT_GOAL := help

# ── Colors ────────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m
DIM   := \033[2m

.PHONY: help install dev stop chat dry-run validate cost \
        test test-unit test-integration \
        migrate migrate-create format lint typecheck \
        clean docker-build

# ─────────────────────────────────────────────────────────────────────────────
#  Help
# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@printf "$(BOLD)TrueNorth — AI Agent Framework$(RESET)\n"
	@echo ""
	@printf "$(CYAN)CONVERSATION$(RESET)\n"
	@printf "  make chat                      interactive chat (mock LLM, free)\n"
	@printf "  make chat GOAL=medical_intake  chat with different goal\n"
	@printf "  make chat LIVE=1               chat with real LLM (needs API keys)\n"
	@printf "  make dry-run                   automated test run (no human input)\n"
	@printf "  make dry-run GOAL=medical_intake\n"
	@printf "  make dry-run SCENARIO=tests/fixtures/scenarios/fitness_happy_path.json\n"
	@printf "  make validate GOAL=fitness_plan validate YAML schema\n"
	@echo ""
	@printf "$(CYAN)SERVER$(RESET)\n"
	@printf "  make dev                       start API + Postgres + Redis (hot-reload)\n"
	@printf "  make stop                      stop all docker services\n"
	@printf "  make logs                      tail all service logs\n"
	@echo ""
	@printf "$(CYAN)TESTING$(RESET)\n"
	@printf "  make test                      run all tests\n"
	@printf "  make test-unit                 run unit tests only\n"
	@printf "  make test-integration          run integration tests only\n"
	@echo ""
	@printf "$(CYAN)DATABASE$(RESET)\n"
	@printf "  make migrate                   run pending Alembic migrations\n"
	@printf "  make migrate-create MSG='...' create a new migration\n"
	@echo ""
	@printf "$(CYAN)CODE QUALITY$(RESET)\n"
	@printf "  make format                    auto-format with ruff\n"
	@printf "  make lint                      lint check with ruff\n"
	@printf "  make typecheck                 type-check with mypy\n"
	@echo ""
	@printf "$(CYAN)SETUP$(RESET)\n"
	@printf "  make install                   install all dependencies\n"
	@printf "  make clean                     remove build artifacts\n"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────────────────────────────────────
install:
	@printf "$(BOLD)Installing dependencies...$(RESET)\n"
	@cd $(CORE_DIR) && pip install poetry --quiet && poetry install
	@cd packages/sdk-node && npm install --silent 2>/dev/null || true
	@printf "$(GREEN)Done.$(RESET)\n"

# ─────────────────────────────────────────────────────────────────────────────
#  CONVERSATION COMMANDS — the ones that matter most
# ─────────────────────────────────────────────────────────────────────────────

## Interactive chat — mock LLM by default (free, no API key needed)
## Use LIVE=1 for real LLM:  make chat LIVE=1
chat:
	@if [ -f .env ]; then export $$(grep -v '^#' .env | xargs) 2>/dev/null; fi; \
	MOCK_FLAG="--mock"; \
	if [ "$(LIVE)" = "1" ]; then MOCK_FLAG=""; fi; \
	cd $(CORE_DIR) && PYTHONPATH=. python cli/main.py chat \
		--goal $(GOAL) \
		$$MOCK_FLAG

## Automated dry-run — no human input, auto-answers all fields
## Runs MOCK by default (free). Use LIVE=1 for real LLM.
dry-run:
	@SCENARIO_FLAG=""; \
	if [ -n "$(SCENARIO)" ]; then SCENARIO_FLAG="--scenario $(SCENARIO)"; fi; \
	MOCK_FLAG="--mock"; \
	if [ "$(LIVE)" = "1" ]; then MOCK_FLAG="--no-mock"; fi; \
	cd $(CORE_DIR) && PYTHONPATH=. python cli/main.py dry-run \
		--goal $(GOAL) \
		$$MOCK_FLAG \
		$$SCENARIO_FLAG \
		--verbose

## Validate a goal YAML
validate:
	@cd $(CORE_DIR) && PYTHONPATH=. python cli/main.py validate --goal $(GOAL)

## Show cost for a session
cost:
	@cd $(CORE_DIR) && PYTHONPATH=. python cli/main.py cost --session-id $(SESSION_ID)

# ─────────────────────────────────────────────────────────────────────────────
#  Docker / Dev server
# ─────────────────────────────────────────────────────────────────────────────
dev:
	@printf "$(BOLD)Starting TrueNorth dev server...$(RESET)\n"
	@cp -n .env.example .env 2>/dev/null || true
	@docker compose -f docker-compose.dev.yml up --build

stop:
	@docker compose -f docker-compose.dev.yml down

logs:
	@docker compose -f docker-compose.dev.yml logs -f

# ─────────────────────────────────────────────────────────────────────────────
#  Testing
# ─────────────────────────────────────────────────────────────────────────────
test:
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/ -v --tb=short

test-unit:
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/unit/ -v --tb=short

test-integration:
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/integration/ -v --tb=short

# ─────────────────────────────────────────────────────────────────────────────
#  Database
# ─────────────────────────────────────────────────────────────────────────────
migrate:
	@cd $(CORE_DIR) && poetry run alembic upgrade head

migrate-create:
	@cd $(CORE_DIR) && poetry run alembic revision --autogenerate -m "$(MSG)"

# ─────────────────────────────────────────────────────────────────────────────
#  Code quality
# ─────────────────────────────────────────────────────────────────────────────
format:
	@cd $(CORE_DIR) && poetry run ruff format truenorth/ cli/ tests/

lint:
	@cd $(CORE_DIR) && poetry run ruff check truenorth/ cli/ tests/

typecheck:
	@cd $(CORE_DIR) && poetry run mypy truenorth/ --ignore-missing-imports

# ─────────────────────────────────────────────────────────────────────────────
#  Clean
# ─────────────────────────────────────────────────────────────────────────────
clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@printf "$(GREEN)Cleaned.$(RESET)\n"