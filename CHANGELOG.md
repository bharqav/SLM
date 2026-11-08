# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Versioning is tag-driven: the distribution version is derived from the most
recent `vX.Y.Z` git tag via `setuptools-scm` (see `pyproject.toml`). This file
is maintained by hand and describes what changed; the tag is the source of
truth for the released artifact. No release tag has been cut yet — the only
tag in the repository is the local backup tag
`backup/pre-artifact-removal-20261107`, which does not match the `v*` release
trigger — so everything since the 1.0.0 hard-coded version is listed here.

### Added

- Fine-tuning core is now covered by tests: `tests/test_train_qlora.py` (11),
  `tests/test_build_steering.py` (10) and `tests/test_merge_adapter.py` (3).
  The suite now defines 320 test functions across 17 files, which expand to 329
  collected cases (327 run by default).
- `tests/test_prepare_data.py` (16 cases) covering the Empathetic-SFT corpus
  build: cleaning, trimming, leak-free splitting and EmpatheticDialogues
  parsing.
- Container build job in CI and a `.dockerignore`, so the image builds from a
  minimal context and never ships training data, adapters or checkpoints.
- `dependabot.yml` covering the `pip` ecosystem and GitHub Actions.
- Fine-tuning Makefile targets mirroring `slm-ft/README.md`:
  `ft-download-ed`, `ft-data`, `ft-steering`, `ft-train`, `ft-train2`,
  `ft-merge` and `ft-all`.
- `parse_args()` in `slm-ft/merge_adapter.py`, so the CLI surface can be
  tested without running a merge.
- Deterministic guardrails for inference: dialogue cleaning, degeneracy
  checks, toxicity filtering and prompt-injection handling.
- RAG demo (`rag_demo.py`) with TF-IDF indexing, chunking and augmented
  prompt assembly.
- Training harness (`harness.py`) orchestrating preprocessing, training and
  evaluation with monitoring hooks.
- Empathetic fine-tuning pipeline: corpus preparation, QLoRA stage 1, stage-2
  steering mix, behavioral gate and fp16 merge.
- Held-out test split evaluation (`evaluate_test_split.py`) and automated
  artifact verification (`verify_setup.py`).
- FastAPI inference server (`slm_api.py`) with Bearer auth, rate limiting,
  CORS and request timeouts.
- Seeded generation (`--seed 1337`) for reproducible outputs.

### Changed

- The model was scaled from the original 81.6M to the current 757M
  (24 layers / 24 heads / 1536 hidden), and the default training path targets
  that configuration.
- The distribution version is derived from git tags via `setuptools-scm`
  instead of the hard-coded `1.0.0` in `pyproject.toml`.
- `release.yml` validates that the tag is a clean `vX.Y.Z`, builds the sdist
  and then the wheel from that sdist, asserts the wheel's version equals the
  tag, and publishes both archives.
- `docs/production_grade_comparison.md` is now linked from the README project
  tree and Additional Resources; it was previously unreferenced.
- README test counts, the tests badge and the per-file coverage table are kept
  in lockstep with the real suite and enforced by
  `TestReadme::test_test_suite_counts_match`.
- Benchmark numbers now reflect the expanded prompt set (20 prompts: 10
  empathetic, 10 code/technical).

### Fixed

- `slm-ft/README.md` and the top-level README now agree with the shipped
  scripts: `prepare_data.py` constants and output layout, the
  `train_qlora.py` `--lora-r` / `--lora-alpha` / `--target-modules` flags, and
  the `merge_adapter.py` defaults.
- CI now runs mypy, and the type-checked surface is listed explicitly in
  `pyproject.toml`.
- The quality gates (ruff, mypy, pytest with coverage, pip-audit) are
  reproducible from a clean checkout and no longer depend on local artifacts.
- Comparison-artifact tests skip when the artifact has not been generated.
- The shipped verification gate passes outside a source checkout.

## [1.0.0] - 2026-10-20

### Added

- Initial release of the SLM vs Qwen comparison.
- From-scratch GPT training pipeline (81.6M parameters at this tag).
- Qwen2.5-0.5B QLoRA fine-tune with stage-2 steering.
- Live same-prompt comparison generator.
- Forward-pass and generation benchmarks.
- README with architecture diagrams and reproduction steps.
