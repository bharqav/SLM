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

- CI now enforces `ruff format --check .`, and the files that had drifted from
  the formatter were reformatted.
- Fine-tuning core is now covered by tests: `tests/test_train_qlora.py` (11),
  `tests/test_build_steering.py` (10) and `tests/test_merge_adapter.py` (3).
  The suite now defines 343 test functions across 17 files, which expand to 352
  collected cases (350 run by default).
- `tests/test_prepare_data.py` (16 cases) covering the Empathetic-SFT corpus
  build: cleaning, trimming, leak-free splitting and EmpatheticDialogues
  parsing.
- Container build job in CI and a `.dockerignore`, so the image builds from a
  minimal context and never ships training data, adapters or checkpoints.
  The ignore patterns are `**/`-prefixed because Docker treats `*` as "within
  one path segment": a bare `*.pt` drops only root-level checkpoints, while
  `slm-training/*.pt` is still sent. With the prefixes the context is 872 kB;
  without them it was 2.85 GB.
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
- `slm-training/preprocess_clean.py`, which rebuilds the training corpus as
  role-marked `User:/Assistant:` dialogues, merges consecutive same-role turns,
  mixes in Alpaca instruction/response pairs for general QA, and can append a
  slice of the code corpus for code completion. Emits `clean_train.bin`,
  `clean_val.bin` and `clean_stats.json`.
- `slm-training/distill_qwen.py`, which generates Qwen2.5-0.5B-Instruct answers
  for the Alpaca instructions in batches and writes them as JSONL.
  `preprocess_clean.py --distill <file>` mixes those records into the corpus in
  the same role format and holds `--distill-val` of them out of validation. The
  teacher's style, not just its structure, is what makes the answers read like
  Qwen's.
- `truncate_at_next_turn` in `slm-training/inference.py`, used by both the
  inference CLI and `slm_api.py`: the generated text is cut at the first `User:` or
  `Assistant:` marker after the answer, so the model cannot keep writing a second
  question or answer after it has finished the reply.
- `--init-from` in `slm-training/train.py` for weights-only continued training:
  it loads model weights (tolerating a `torch.compile` `_orig_mod.` prefix and
  non-matching buffers) while resetting the optimizer and step counter.
- Prompt-format routing for inference and the API. `slm-training/inference.py`
  gained `--mode {auto,chat,code}` and a `format_prompt()` helper that wraps
  dialogue prompts as `User: <text>\nAssistant:` and leaves code prompts
  verbatim; `/generate` and `/v1/completions` accept the same `mode` field, and
  the `/generate` response echoes the exact string sent as `prompt_sent`.
  `format_messages_to_prompt` in `slm_api.py` now appends the `\nAssistant:`
  cue when the last message is a user turn, so `/v1/chat/completions` answers
  instead of continuing its own reply.
- Tests for prompt-format routing: `TestFormatPrompt` (6 cases) in
  `tests/test_inference.py` and five endpoint cases in
  `tests/test_model_and_api.py` covering the `Assistant:` cue, `mode` handling
  and the `prompt_sent` echo.
- `GPT.generate` in `slm-training/model.py` accepts `stop_strings` and
  `decode_fn`: when both are given, a sequence ends as soon as its decoded
  continuation contains any stop string. Both default to off, so callers that
  do not need text-level stopping are unaffected. `slm_api.py` passes
  `inference.TURN_MARKERS` as the stop strings, so generation ends at the next
  turn boundary instead of running to `max_new_tokens`.

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
- README section 8.1 now points at `best_clean_model.pt` and demonstrates the
  role-marked prompt format for both dialogue and code completion.
- Inference and API defaults were lowered to `temperature 0.3` / `top_k 20`
  (from `1.0` / `50`) so answers stay on-topic, generation now stops on the EOS
  token the model was trained with, and the default checkpoint moved to
  `slm-training/best_clean_model_v2.pt` in `inference.py`, `slm_api.py` and
  `docker-compose.yml`.
- `constraints.txt` was refreshed to the versions the project actually runs on
  (torch 2.15.0, transformers 5.22.2, huggingface-hub 2.4.0, tiktoken 0.14.0,
  numpy 2.5.4, peft 0.21.2, datasets 5.1.0, pytest 9.2.0, ruff 0.17.4,
  mypy 2.4.1), and the GitHub Actions used by CI and the release workflow were
  bumped to their current majors.

### Fixed

- The Docker image now installs the CPU-only torch wheel. The default Linux wheel bundles the
  CUDA runtime, which filled the CI runner's disk (`No space left on device`) and made the
  `docker-build` job fail intermittently on pushes to `main`.
- Mermaid diagrams in the README (two Qwen architecture graphs) failed to render on GitHub
  because a node label with parentheses was unquoted; the label is now quoted.
- Stale claims removed: `SLM_ARCHITECTURE.md` described a 12-layer 768-wide model while stating
  757M parameters; `MODEL_CARD.md` and `docs/production_grade_comparison.md` claimed the 757M SLM
  was ~50x faster than Qwen2.5-0.5B and better at empathy. The current benchmarks show the 757M
  model is slower, and quality is worse. `benchmarks/benchmark_report.md` and its generator
  mislabelled Qwen as 24L/16H/1536D and carried the same speed claim.
- Answers opened with fragments like "ive" and drifted into loops and
  non-sequiturs instead of answering the question. The weights were not at
  fault: the corpus renders a turn as `User: <text>\nAssistant: <answer>`, and
  tiktoken merges the space after `Assistant:` into the answer's first token
  (`" Hi"` is one token, not `" "` plus `"Hi"`). `ASSISTANT_SUFFIX` in
  `slm-training/inference.py` and the cue in `slm_api.py` both ended with a
  trailing space, so the model was asked to continue from a standalone space
  token that never occurs in training. Measured on `best_clean_model_v3.pt`,
  that context flattened the first-token distribution (`" -"` 0.116,
  `" capital"` 0.094, `"ive"` 0.211); ending the cue at the colon makes it
  sharp (`"The"` 0.990, `"To"` 0.450, `"A"` 0.957). The cue is now
  `\nAssistant:` everywhere, and `truncate_at_next_turn` trims the leading space
  the first generated token carries.
- The 253M checkpoint answered every prompt with a non-sequitur, regardless of
  the question asked. Root cause: the legacy `empathy_train.bin` had no speaker
  markers and contained 2,511 runs of consecutive same-role turns, so the model
  learned to continue a message with an unrelated one; it also held no
  instruction data, so general questions were entirely out of distribution.
  The model is now continued on the clean role-marked corpus
  (`clean_train.bin`) built by `preprocess_clean.py`, and the best checkpoint is
  `best_clean_model.pt`. Inference must wrap non-code prompts as
  `User: <text>\nAssistant:` to match the training format (the trailing-space
  correction is recorded under Fixed below).
- Generation ran past the end of the answer: the model kept writing after it had
  finished the reply, emitting a second question or a follow-up answer that was
  never asked for, and it never stopped on its own because no EOS token was set.
  Generation now stops at the EOS token the corpus ends each record with, and
  `truncate_at_next_turn` cuts the text at the next `User:`/`Assistant:` marker as
  a second line of defence.
- The model loops on open-ended prompts, repeating a phrase or clause instead of
  finishing the answer ("Exercise regularly" three times, "because of the way ...
  because of the way ..."). `slm_api.py` already passed `repetition_penalty` but
  the inference CLI never did, so `inference.py` gained `--repetition-penalty`
  (default 1.2) and both `generate()` call sites forward it. 1.2 is enough to
  break the loop without changing the answer; 1.0 keeps the old behaviour.
- Five Mermaid diagrams in `README.md` did not parse, so GitHub rendered them
  as raw code instead of figures: unquoted node labels containing parentheses,
  unterminated parallelogram shapes (`[/health GET]`), and
  `for i from 1 to N ... end` loops with indexed node IDs (`QB[i][...]`).
  Labels are now quoted, parallelograms are closed on both sides, and the
  repeated-block loops are represented by a single labelled node inside a
  counting subgraph. `CONTRIBUTING.md` now documents these constraints.
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
- The image smoke test in CI probes `/health` through `docker exec` and
  asserts the container is still running and `model_loaded` is `false`, so a
  broken image fails the build instead of the next deploy.
- `requirements.lock` was regenerated with `pip freeze --exclude-editable`
  from the project's own environment. The previous file was a freeze of a
  different machine's environment: it carried 88 packages this project never
  installs (anomalib, ultralytics, mujoco, kornia, lightning, ...) alongside
  stale pins for the packages it does use.

## [1.0.0] - 2026-10-20

### Added

- Initial release of the SLM vs Qwen comparison.
- From-scratch GPT training pipeline (81.6M parameters at this tag).
- Qwen2.5-0.5B QLoRA fine-tune with stage-2 steering.
- Live same-prompt comparison generator.
- Forward-pass and generation benchmarks.
- README with architecture diagrams and reproduction steps.
