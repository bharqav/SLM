# Training runbook (custom SLM / adapter)

End-to-end checklist for building a **supportive dialogue** model with **emotion**, **safety**, and **crisis routing**, suitable for student-scale GPU and production realism.

## 1. Define target behavior

**Tasks**

- Supportive dialogue generation (empathy, reflection, no diagnosis)
- Emotion recognition (aux classifier or head)
- Safety refusal + crisis routing (combine classifier + policy layer — see `crisis_policy.py`)

**Metrics**

| Metric | Notes |
|--------|--------|
| Empathy | Human rubric or LLM-judge with calibration |
| Safety precision / recall | Especially **critical self-harm** recall |
| Hallucination rate | Fact-sensitive probes (non-clinical facts only) |
| Latency / cost / tokens | Per-turn from gateway metrics |

## 2. Build datasets

**Sources**

- Open supportive-dialogue corpora (license-checked)
- Synthetic safety/crisis conversations (refusal + escalation)
- Refusal policy examples aligned with your crisis taxonomy

**Splits:** 80% train / 10% val / 10% test — **no leakage** across slices.

**Slice labels:** self-harm, violence, abuse, neutral, multilingual, short vs long context.

## 3. Data governance

- Strip direct PII; normalize unsafe content tags
- Annotation guidelines: empathy quality, unsafe severity, **correct crisis level** (L0–L3)

## 4. Base model & strategy

- **Practical:** 7B / 8B **instruct** model
- **Pipeline:** **SFT** (instruction-response) → **DPO / ORPO** for tone + safety preferences
- **Crisis:** keep a **separate fast classifier** (DistilBERT-style) for routing — matches production controllability

## 5. Train v1 (SFT)

- Format: instruction → response pairs
- **LoRA / QLoRA** for cost control
- Track: LR, rank, epochs, max context — save **checkpoints + metadata** (W&B / MLflow)

## 6. Offline evaluation gate (must-pass)

- Safety **critical recall** ≥ threshold
- Refusal correctness on policy set
- Empathy rubric minimum
- **No regression** vs previous checkpoint on held-out test

Fail → adjust data or hyperparameters; do not ship.

## 7. Red-team / adversarial

- Prompt injection, jailbreaks, self-harm method elicitation, ambiguous distress

Pass required before **canary**.

## 8. Package & serve

- Export adapter or merged weights
- Deploy behind **model gateway** (this repo): max tokens, timeouts, concurrency caps, **fallback model** config

## 9. Canary (5% → 25% → 100%)

- Compare: latency, safety flags, user feedback, abandon/retry
- Auto-rollback if fallback rate or p99 latency exceeds thresholds (`release_registry.json` documents targets)

## 10. Continuous improvement

- **Weekly:** anonymized hard cases → human label → refresh eval slices
- **Monthly:** adapter refresh + full safety suite

**Tools:** Unsloth, Axolotl, TRL, PEFT — pick one stack and standardize experiment IDs tied to `model_version` in `release_registry.json`.

## 11. 24GB VRAM quickstart (implemented)

Use the scaffolded scripts in `scripts/training/`:

1. Install deps: `pip install -r scripts/training/requirements-training.txt`
2. Print commands for a 24GB card: `python scripts/training/model_v2_24gb.py`
3. Train classifiers: `python scripts/training/train_classifier.py ...`
4. Run release gates: `python scripts/training/run_eval_suite.py --predictions <metrics.json> --thresholds scripts/training/thresholds.json`

## 12. 8.5GB VRAM QLoRA path (implemented, slm-ft/)

For the empathetic agent model on the RTX 5060 Laptop (8.5GB, Blackwell):

1. **Stage 1 — empathy SFT**: Qwen2.5-3B-Instruct, 4-bit NF4 QLoRA, 2 epochs
   over EmpatheticDialogues + ESConv (`slm-ft/prepare_data.py` → ~23.4k dialogs).
   See `slm-ft/README.md` for the exact commands and `train_qlora.log` progress.
2. **Stage 2 — policy steering**: continue from the stage-1 adapter with the
   steering set (`slm-ft/steering/build_steering.py`) mixed with a replay slice
   so crisis/safety reply behavior matches `backend/services/crisis_policy.py`
   (L2 support-and-redirect, medical → emergency services, injection refusals).
3. **Automatic chain**: `restart_train_guard.py` restarts stage-1 on crash and
   then runs stage-2 → `evaluate_ft.py` behavioral gate → `merge_adapter.py`
   (fp16 out to `slm-ft/merged_fp16/`).
4. **Serve**: `python slm-ft/serve_model.py --model merged_fp16 --port 8902`
   loads the merged model in 4-bit on the same GPU; point the backend at it
   with `SERVED_MODEL_URL=http://127.0.0.1:8902` (+ optional bearer token).
   The gateway places this tier after the external HF fallback, before the
   canned stub — safety service still gates input/output around it.
5. **License constraint**: EmpatheticDialogues + ESConv are cc-by-nc(-sa).
   The merged artifact inherits non-commercial training data — fine for this
   project; do not push it into a commercial deployment path.

## 13. From-scratch empathetic variant (slm-training code, queued)

The tiny 81.6M GPT (`slm-training/train.py`) is being retrained on the same
validated empathetic dialogs instead of code prose:

- Corpus: `slm-training/build_empathetic_corpus.py` renders `slm-ft/data`
  dialogs to `<sys>persona</sys><user>…</user><assistant>…</assistant>` tag
  streams (~3.1M train / 62k val tokens, EOS-separated, epoch-safe shuffle).
- Isolated run dir `slm-training/empathetic_run/` keeps the original code-prose
  artifacts (best_model.pt, val ppl 19.7) intact — separate bins, checkpoints,
  logs.
- Recipe: 2000 steps @ 8192 tok/step (≈5 epochs of 3.1M tokens), lr 3e-4 with
  200-step warmup, patience-based early stop (800 steps), checkpoint every
  100.
- `launch_empathetic_after_qlora.py` (detached) waits for the QLoRA chain to
  finish, then waits for the GPU to free up and runs the training + a sample
  generation (`empathetic_run/train.log`, `empathetic_run/samples.log`).
