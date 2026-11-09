.PHONY: help test test-all test-fast bench verify clean lint format format-check typecheck security api docker-build docker-run install install-all harness harness-model export-onnx ft-download-ed ft-data ft-steering ft-train ft-train2 ft-merge ft-all

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

security: ## Run pip-audit security scan (declared requirements)
	python -m pip_audit --requirement requirements.txt

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

export-onnx: ## Export best_clean_model_v2.pt to ONNX with dynamic axes (requires onnx + torch)
	python -m slm-training.export_onnx --checkpoint slm-training/best_clean_model_v2.pt --output slm-training/best_model.onnx --max-seq-len 256

# --- slm-ft: QLoRA fine-tune of Qwen2.5-0.5B-Instruct -------------------------
# These mirror slm-ft/README.md step for step. They need a CUDA GPU and the
# QLoRA extra (`pip install -e ".[qlora]"`); the base model is downloaded from
# the HuggingFace hub on first use.

ft-download-ed: ## Download the EmpatheticDialogues tarball (cc-by-nc-4.0)
	python scripts/download_empathetic_dialogues.py --output-dir slm-ft/data/ed_docraw

ft-data: ## Build the SFT corpus (ESConv + EmpatheticDialogues, hash-split)
	python slm-ft/prepare_data.py --ed-csv-dir slm-ft/data/ed_docraw

ft-steering: ## Generate the stage-2 steering + replay mix
	python slm-ft/steering/build_steering.py

ft-train: ## Stage 1: QLoRA on the empathetic corpus (committed config: r=8/alpha=8, q_proj,v_proj)
	python slm-ft/train_qlora.py --epochs 2 --out slm-ft/adapter --lora-r 8 --lora-alpha 8 --target-modules q_proj,v_proj

ft-train2: ## Stage 2: policy steering from the stage-1 adapter
	python slm-ft/train_qlora.py --data slm-ft/data/stage2_sft.jsonl --init-adapter slm-ft/adapter --epochs 1 --lr 5e-5 --accum 8 --out slm-ft/adapter_stage2

ft-merge: ## Merge the stage-2 adapter into a fp16 model folder
	python slm-ft/merge_adapter.py --adapter slm-ft/adapter_stage2 --out slm-ft/merged_fp16

ft-all: ## Full slm-ft pipeline: data -> steering -> stage 1 -> stage 2 -> merge (GPU, hours)
ft-all: ft-data ft-steering ft-train ft-train2 ft-merge
