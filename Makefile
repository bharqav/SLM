.PHONY: help test test-all test-fast bench verify clean lint format format-check typecheck security api docker-build docker-run install install-all harness harness-model export-onnx

harness: ## Run full harness (tests + verify + bench + security)
	python harness.py

harness-model: ## Run full harness including model-dependent steps
	python harness.py

help: ## Show this help message
	@echo "SLM Project - Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run pytest with coverage (excludes slow tests)
	python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=80

test-all: ## Run all tests including slow tests
	python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=80 -m "not slow" --override-ini="addopts="
	python -m pytest tests/ -v -m slow

test-fast: ## Run pytest without coverage
	python -m pytest tests/ -v

bench: ## Run full benchmark pipeline
	python run_all.py

verify: ## Run setup verification
	python verify_setup.py

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .venv -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info/

lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter
	ruff format .

format-check: ## Check formatting without modifying files
	ruff format --check .

typecheck: ## Run mypy type checker
	mypy .

security: ## Run pip-audit security scan
	python -m pip_audit

api: ## Start FastAPI server
	python -m uvicorn slm_api:app --host $(API_HOST) --port $(API_PORT) --reload

serve: ## Start production server locally
	./serving/start_server.sh

serve-prod: ## Start production server via Docker Compose
	docker compose -f docker-compose.prod.yml up --build

docker-build: ## Build Docker image
	docker build -t slm-project .

docker-run: ## Run benchmarks in Docker
	docker run --rm -v $$(pwd)/slm-comparison:/app/slm-comparison -v $$(pwd)/benchmarks:/app/benchmarks slm-project

install: ## Install project in editable mode with dev deps
	pip install -e ".[dev,server]"

install-all: ## Install project with all optional deps
	pip install -e ".[all]"

export-onnx: ## Export best_model.pt to ONNX with dynamic axes (requires onnx + torch)
	python -m slm-training.export_onnx --checkpoint slm-training/best_model.pt --output slm-training/best_model.onnx --max-seq-len 256
