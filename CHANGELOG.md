# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Comprehensive test suite with 15 passing pytest tests
- `verify_setup.py` - Automated artifact verification
- `evaluate_test_split.py` - Held-out test split perplexity evaluation
- `slm_api.py` - FastAPI inference server with Bearer auth
- `run_all.py` - Single entry-point for full pipeline
- `Makefile` - Build/test/bench/verify shortcuts
- `pyproject.toml` - Project metadata and tool configuration
- `requirements.txt` and `constraints.txt` - Dependency management
- `.pre-commit-config.yaml` - Pre-commit hooks for linting/formatting
- `.env.example` - Environment variable template
- `.gitignore` - Git ignore rules
- Dockerfile and docker-compose.yml for containerized deployment
- `notebooks/token_analysis.ipynb` - Token distribution EDA
- `MODEL_CARD.md` - HuggingFace Hub-ready model card
- Expanded prompt set from 9 to 20 prompts (10 empathetic + 10 code/technical)
- Seeded generation (`--seed 1337`) for reproducible outputs
- CI pipeline with ruff, pytest+coverage, verification, pip-audit, and artifact upload
- GitHub issue and PR templates
- CONTRIBUTING.md and SECURITY.md

### Changed
- Updated benchmark numbers to reflect 20-prompt evaluation
- Increased default model to 12L/12H/768D (~124M params)
- Restored original GPT-2 architecture from git history (10L/10H/640D)
- Fixed `tests/__init__.py` (was broken literal text)
- Reorganized README into 18 sections with consistent numbering
- Updated comparison pipeline documentation with honest placeholder disclosures
- Added `--data-dir`, `--train-bin`, `--val-bin` to `train.py`
- Added `preprocess_empathy.py` for empathetic dialogue tokenization

### Added
- Empathetic fine-tune pipeline: 23,427 dialogues, ~2.4M tokens
- `slm-training/empathy_train.bin` and `empathy_val.bin`
- Training on ESConv + EmpatheticDialogues data
- New default model config with 2.4x parameter increase

### Fixed
- Model loading failure due to architecture mismatch (restored `model.py` from git)
- Broken reference in `benchmarks/benchmark_report.md`

## [1.0.0] - 2026-10-20

### Added
- Initial release of SLM vs Qwen comparison
- From-scratch GPT training pipeline (81.6M params)
- Qwen2.5-0.5B QLoRA fine-tune with stage-2 steering
- Live same-prompt comparison generator
- Forward-pass and generation benchmarks
- Comprehensive README with architecture diagrams
