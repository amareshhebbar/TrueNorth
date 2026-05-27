.PHONY: help setup dev test lint clean

help:
	@echo "TrueNorth Monorepo Commands"
	@echo "─────────────────────────────────────────"
	@echo "  make setup      Install all dependencies"
	@echo "  make dev        Start server + studio in dev mode"
	@echo "  make test       Run all tests"
	@echo "  make lint       Lint all packages"
	@echo "  make db         Start Postgres + Redis"
	@echo "  make migrate    Run DB migrations"
	@echo "  make chat       Start interactive CLI chat"
	@echo "  make dry-run    Run dry-run example"

setup:
	cd packages/core && pip install poetry && poetry install
	npm install

dev:
	@echo "Starting TrueNorth dev server..."
	cd packages/core && poetry run uvicorn truenorth.api.main:app --reload --port 8000

studio:
	npm run dev:studio

db:
	docker-compose up -d postgres redis
	@echo "Waiting for Postgres..." && sleep 3

migrate:
	cd packages/core && poetry run alembic upgrade head

test:
	cd packages/core && poetry run pytest tests/unit -v

test-all:
	cd packages/core && poetry run pytest -v

lint:
	cd packages/core && poetry run ruff check .
	cd packages/sdk-node && npm run build

chat:
	cd packages/core && poetry run truenorth chat examples/goals/fitness_plan.yaml

dry-run:
	cd packages/core && poetry run truenorth dry-run \
		examples/goals/fitness_plan.yaml \
		--scenario tests/fixtures/scenarios/fitness_happy_path.json

validate:
	cd packages/core && poetry run truenorth validate examples/goals/fitness_plan.yaml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf packages/sdk-node/dist packages/studio/.next
