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
        test test-unit test-integration test-all \
        test-node test-go test-rust test-expo \
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
	@printf "  make test-all                  run ALL tests (Python + Node + Go + Rust)\n"
	@printf "  make test                      run Python tests only\n"
	@printf "  make test-unit                 run Python unit tests only\n"
	@printf "  make test-integration          run Python integration tests only\n"
	@printf "  make test-node                 TypeScript type-check (sdk-node)\n"
	@printf "  make test-go                   Go build + vet (sdk-go)\n"
	@printf "  make test-rust                 Rust cargo check (sdk-rust)\n"
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
	@printf "  make install                   install ALL dependencies (Python + Node + Go + Rust)\n"
	@printf "  make clean                     remove build artifacts\n"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Setup — installs everything
# ─────────────────────────────────────────────────────────────────────────────
install:
	@printf "$(BOLD)Installing dependencies...$(RESET)\n"
	@echo ""

	@printf "$(CYAN)▶  Python$(RESET)\n"
	@cd $(CORE_DIR) && pip install poetry --quiet && poetry install
	@printf "   $(GREEN)✅  Python ready$(RESET)\n"

	@printf "$(CYAN)▶  Node SDK$(RESET)\n"
	@cd packages/sdk-node && npm install --silent
	@printf "   $(GREEN)✅  Node SDK ready$(RESET)\n"

	@printf "$(CYAN)▶  Go SDK$(RESET)\n"
	@cd packages/sdk-go && go mod tidy
	@printf "   $(GREEN)✅  Go SDK ready$(RESET)\n"

	@printf "$(CYAN)▶  Rust SDK$(RESET)\n"
	@cd packages/sdk-rust && cargo fetch --quiet
	@printf "   $(GREEN)✅  Rust SDK ready$(RESET)\n"

	@echo ""
	@printf "$(GREEN)$(BOLD)All dependencies installed.$(RESET)\n"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  CONVERSATION COMMANDS
# ─────────────────────────────────────────────────────────────────────────────
chat:
	@if [ -f .env ]; then export $$(grep -v '^#' .env | xargs) 2>/dev/null; fi; \
	MOCK_FLAG="--mock"; \
	if [ "$(LIVE)" = "1" ]; then MOCK_FLAG=""; fi; \
	cd $(CORE_DIR) && PYTHONPATH=. python cli/main.py chat \
		--goal $(GOAL) \
		$$MOCK_FLAG

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

validate:
	@cd $(CORE_DIR) && PYTHONPATH=. python cli/main.py validate --goal $(GOAL)

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

# Run every SDK in one command
test-all:
	@echo ""
	@printf "$(BOLD)╔══════════════════════════════════════════╗$(RESET)\n"
	@printf "$(BOLD)║     TrueNorth — Full Test Suite          ║$(RESET)\n"
	@printf "$(BOLD)╚══════════════════════════════════════════╝$(RESET)\n"
	@echo ""
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory test-node
	@$(MAKE) --no-print-directory test-go
	@$(MAKE) --no-print-directory test-rust
	@echo ""
	@printf "$(GREEN)$(BOLD)✅  All suites passed$(RESET)\n"
	@echo ""

# Python — full suite
test:
	@printf "$(CYAN)▶  Python — pytest$(RESET)\n"
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/ \
		--asyncio-mode=auto \
		-q \
		--tb=short \
		--no-header \
		-p no:warnings \
	&& printf "   $(GREEN)✅  Python tests passed$(RESET)\n" \
	|| (printf "   \033[31m❌  Python tests FAILED$(RESET)\n" && exit 1)

# Python — unit only (fast)
test-unit:
	@printf "$(CYAN)▶  Python unit tests$(RESET)\n"
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/unit/ \
		--asyncio-mode=auto \
		-q \
		--tb=short \
		--no-header \
		-p no:warnings

# Python — integration only
test-integration:
	@printf "$(CYAN)▶  Python integration tests$(RESET)\n"
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/integration/ \
		--asyncio-mode=auto \
		-v \
		--tb=short

# Node SDK — TypeScript type check
test-node:
	@printf "$(CYAN)▶  Node SDK — TypeScript$(RESET)\n"
	@cd packages/sdk-node && npx tsc --noEmit \
	&& printf "   $(GREEN)✅  TypeScript types valid$(RESET)\n" \
	|| (printf "   \033[31m❌  TypeScript errors found$(RESET)\n" && exit 1)

# Expo SDK — TypeScript type check
test-expo:
	@printf "$(CYAN)▶  Expo SDK — TypeScript$(RESET)\n"
	@cd packages/sdk-expo && npx tsc --noEmit \
	&& printf "   $(GREEN)✅  Expo types valid$(RESET)\n" \
	|| (printf "   \033[31m❌  Expo TypeScript errors$(RESET)\n" && exit 1)

# Go SDK — build + vet
test-go:
	@printf "$(CYAN)▶  Go SDK — build + vet$(RESET)\n"
	@cd packages/sdk-go && go build ./... && go vet ./... \
	&& printf "   $(GREEN)✅  Go SDK builds cleanly$(RESET)\n" \
	|| (printf "   \033[31m❌  Go SDK FAILED$(RESET)\n" && exit 1)

# Rust SDK — compile check (fast, no integration tests)
test-rust:
	@printf "$(CYAN)▶  Rust SDK — cargo check$(RESET)\n"
	@cd packages/sdk-rust && cargo check --quiet \
	&& printf "   $(GREEN)✅  Rust SDK compiles$(RESET)\n" \
	|| (printf "   \033[31m❌  Rust SDK FAILED$(RESET)\n" && exit 1)

# Python with coverage report
test-coverage:
	@cd $(CORE_DIR) && PYTHONPATH=. poetry run pytest tests/unit/ \
		--asyncio-mode=auto \
		--cov=truenorth \
		--cov-report=term-missing \
		--cov-report=html \
		-q \
	&& echo "Coverage report → packages/core/htmlcov/index.html"

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