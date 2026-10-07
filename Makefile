.PHONY: setup lint test eval secret-scan legacy-compile web-install web-test e2e dev check

setup:            ## install Python workspace (incl. dev tools) and web deps
	uv sync --all-packages
	cd apps/web && npm ci

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

eval:             ## run every evaluation suite; writes docs/EVAL_REPORT.md
	uv run python -m eval_harness run --suite all

secret-scan:
	python3 tools/secret_scan.py

legacy-compile:
	python3 -m compileall -q legacy

web-test:
	cd apps/web && npm run lint && npm run typecheck && npm test

e2e:
	cd apps/web && npx playwright test

dev:              ## one-command local stack
	docker compose up --build

check: secret-scan lint test legacy-compile
