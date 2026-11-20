# SupportRouter — every documented command is a make target (`make help`).

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test check requirements requirements-check data split train train-sample train-t1-sample import-t1-colab evaluate evaluate-t1-sample llm-smoke evaluate-llm evaluate-subset predictions compare errors robustness typos register threshold-curve serve api-latency compose-up compose-smoke compose-down quality-gate coverage secrets-scan simulate-traffic drift readme-results readme-check

# MLflow 3.17 prints an "agent hint" log line on import; this documented switch silences it.
export MLFLOW_DISABLE_AGENT_HINT := 1

# Docker Compose mounts mlruns/ at this absolute host path (recorded artifact URIs).
export HOST_PROJECT_DIR := $(CURDIR)

REQS_EXPORT = uv export --locked --no-header --no-dev --no-emit-project --format requirements-txt

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Create .venv and install the exact locked dependencies
	uv sync --locked

lint: ## Lint and check formatting (no changes)
	uv run ruff check .
	uv run ruff format --check .

format: ## Apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type check of src/ (mypy strict)
	uv run mypy

test: ## Run the test suite
	uv run pytest

coverage: ## Tests with coverage of src/; fails under 70%
	uv run pytest --cov --cov-report=term

check: lint typecheck coverage requirements-check ## Everything the CI check job runs

# Same pinned image as CI; scans the git history (not untracked files such as .env).
GITLEAKS_IMAGE = ghcr.io/gitleaks/gitleaks:v8.29.0@sha256:71d3ee5990f2176f763b438298453fc37e87b119122045e176ca9d44ff00b08b

secrets-scan: ## gitleaks over the whole git history (redacted output)
	docker run --rm --user "$$(id -u):$$(id -g)" -v "$(CURDIR):/repo:ro" $(GITLEAKS_IMAGE) \
		git --redact --no-banner /repo

requirements: ## Export pinned runtime deps from uv.lock to requirements.txt
	$(REQS_EXPORT) -o requirements.txt

requirements-check: ## Fail if requirements.txt is out of date with uv.lock
	$(REQS_EXPORT) | diff -q - requirements.txt

data: ## Download the pinned raw dataset to data/raw/ and print a first look
	uv run python scripts/download_data.py

# PYTHONWARNINGS hides tqdm's "IProgress not found" warning, which prints a local path.
split: ## Build the frozen train/val/test split + manifest (re-run verifies, never overwrites)
	uv run python scripts/make_splits.py

train: ## Train B0 and B1 on the full train split, choose C on val, log to MLflow
	uv run python -m supportrouter.train --model b0
	uv run python -m supportrouter.train --model b1

train-sample: ## Train B1 on a stratified 3000-row train sample (SAMPLE mode)
	uv run python -m supportrouter.train --model b1 --sample

quality-gate: ## Train B1 SAMPLE, fail if val macro-F1 < 0.85 -> reports/quality_gate.json
	uv run python -m supportrouter.quality_gate

train-t1-sample: ## Fine-tune T1 DistilBERT on CPU on the 3000-row train sample, best epoch on val
	uv run python -m supportrouter.train --model t1 --sample

import-t1-colab: ## Verify + log an unzipped T1 export (`export_t1_run` on a GPU machine) as the FULL run: DIR=<folder>
	uv run python -m supportrouter.train --model t1 --import-dir $(DIR)

evaluate: ## Score final B0/B1 (FULL + B1 SAMPLE) on the hash-checked test split -> reports/metrics/
	uv run python -m supportrouter.evaluate --model b0
	uv run python -m supportrouter.evaluate --model b1
	uv run python -m supportrouter.evaluate --model b1 --sample

evaluate-t1-sample: ## Score the final T1 SAMPLE model (hash-checked) on the test split -> reports/metrics/
	uv run python -m supportrouter.evaluate --model t1 --sample

llm-smoke: ## LLM pipeline check on 27 val messages with local Ollama ($0); writes no files
	uv run python -m supportrouter.evaluate --model l1 --smoke 27

evaluate-llm: ## L1 zero-shot + L2 few-shot on the fixed seeded 300 test subset (local Ollama, $0 cap)
	uv run python -m supportrouter.evaluate --model l1
	uv run python -m supportrouter.evaluate --model l2

evaluate-subset: ## B1 FULL and T1 FULL on the same seeded 300 test subset as L1/L2 (fair comparison rows)
	uv run python -m supportrouter.evaluate --model b1 --subset
	uv run python -m supportrouter.evaluate --model t1 --subset

predictions: ## Per-row test predictions of B1 FULL and T1 FULL (no text) -> reports/predictions/
	uv run python -m supportrouter.evaluate --model b1 --predictions
	uv run python -m supportrouter.evaluate --model t1 --predictions

compare: ## Regenerate reports/model_comparison.md from reports/metrics/*.json (no hand-typed numbers)
	uv run python scripts/compare_models.py

errors: ## Confusion figures + top confused pairs with examples -> reports/error_analysis.md (needs make predictions)
	uv run python scripts/compare_models.py --errors

robustness: ## Accuracy per flags letter and near-duplicate slice -> reports/robustness.md (needs make predictions)
	uv run python scripts/compare_models.py --robustness

typos: ## Typo test: B1 FULL and T1 FULL on seeded typo-perturbed test -> reports/robustness_typos.json
	uv run python -m supportrouter.evaluate --model t1 --typos

register: ## Register T1 FULL as @champion and B1 FULL as @fallback in the MLflow registry
	uv run python -m supportrouter.train --register

threshold-curve: ## Threshold evidence: coverage vs accuracy per confidence threshold on val (@champion, @fallback)
	uv run python -m supportrouter.evaluate --model t1 --threshold-curve

# `.env` (if present) is exported for the server only; its values are never printed.
serve: ## Run the API on http://127.0.0.1:8000 (Swagger at /docs); loads @champion once
	set -a; [ -f .env ] && . ./.env; set +a; \
	uv run uvicorn app.api:app --host 127.0.0.1 --port 8000

api-latency: ## Start the API, measure warm single-message POST /classify p50/p95 -> reports/api_latency.json
	@set -a; [ -f .env ] && . ./.env; set +a; \
	uv run uvicorn app.api:app --host 127.0.0.1 --port 8001 --log-level warning & pid=$$!; \
	for i in $$(seq 60); do curl -sf http://127.0.0.1:8001/health >/dev/null && break; sleep 1; done; \
	uv run python -m supportrouter.evaluate --model t1 --api-url http://127.0.0.1:8001; status=$$?; \
	kill $$pid; exit $$status

compose-up: ## Build the image, start API + MLflow server, wait until both are healthy
	docker compose up --build --detach --wait

compose-smoke: ## Call the running stack: /health, one /classify, MLflow /health
	curl -sf http://127.0.0.1:8000/health && echo
	curl -sf -X POST http://127.0.0.1:8000/classify -H 'Content-Type: application/json' \
		-d '{"texts": ["I want to cancel my order"]}' && echo
	curl -sf http://127.0.0.1:5000/health && echo

compose-down: ## Stop and remove the stack's containers (mounted files stay on the host)
	docker compose down

simulate-traffic: ## Send reference/normal/shifted val batches to a running API (start `make serve` first)
	uv run python scripts/simulate_traffic.py

drift: ## Evidently drift report: normal and shifted batches vs reference -> reports/drift/ (no telemetry)
	DO_NOT_TRACK=1 uv run python -m supportrouter.monitoring.drift_report

readme-results: ## Regenerate the README results block from reports/
	uv run python -m supportrouter.readme_results

readme-check: ## Fail if README numbers differ from reports/ (also run by pytest)
	uv run python -m supportrouter.readme_results --check
