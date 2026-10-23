# slm-ft — QLoRA fine-tune of Qwen2.5-3B-Instruct for empathetic dialogue

**Status (20 Oct 2026): trained to completion.**
Stage-1: 2,928 steps / 2 epochs, final val loss **2.3947** (23,427 dialogs).
Stage-2 policy steering: steering×9 + 500 replay, 192 steps, val 2.47–2.69.
Behavioral gate: **5/5 probes pass** (`adapter_stage2/behavioral_eval.json` —
crisis support-and-redirect, medical → emergency services, injection refusal,
warm benign replies). Deployment artifact: `merged_fp16/` (6.17GB fp16
safetensors, 434 tensors verified).

Turns the pre-trained Qwen2.5-3B-Instruct base into a supportive-conversation
assistant with a LoRA adapter trained on public empathetic-dialogue corpora.
This is the **emotional-SLM story**: unlike `slm-training/` (from-scratch
81.6M GPT trained on code prose), the artifact produced here is intended to
power the chat pipeline's `main` tier once wired to the backend gateway.

## Setup (done once)

    py -V:Astral/CPython3.12.14 -m venv .venv
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\pip install transformers peft datasets accelerate bitsandbytes

Verified working on the RTX 5060 Laptop GPU (Blackwell, sm_120): torch
2.11.0+cu128, bitsandbytes 0.50.2 4-bit NF4 kernels run natively.

## Data

1. EmpatheticDialogues (facebookresearch tarball, ~23k short empathetic
   exchanges, **cc-by-nc-4.0** — non-commercial only):

       Invoke-WebRequest https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz -OutFile ed.tar.gz
       tar -xzf ed.tar.gz -C data\ed_docraw

2. ESConv (thu-coai/esconv from the HF hub, 910 multi-turn emotional-support
   dialogs) — fetched automatically by `prepare_data.py`.

3. Build the canonical chat corpus:

       .venv\Scripts\python prepare_data.py --ed-csv-dir data\ed_docraw
       # -> data/sft_train.jsonl + data/sft_val.jsonl (~23.4k / ~0.5k dialogs)

Conversations are hash-split before writing, so no dialog leaks into val.

## Training

    .venv\Scripts\python train_qlora.py --epochs 2 --out adapter

Defaults: 4-bit NF4 base, bf16 compute, LoRA r=16 alpha=32 on all attention +
MLP projections, seq len 1024, batch 1 × accum 16, AdamW 1e-4 cosine with 3%
warmup, gradient checkpointing. VRAM peak ≈ 5.5–6.5GB — fits the 8.5GB card
(leave the `slm-training` training run idle or finished before launching).

Loss is masked to assistant turns only (`-100` labels on the prompt prefix),
using Qwen's native chat template. Intermediate `train_log.json` records the
loss curve; the adapter is checkpointed after each epoch's val pass.

## Stage 2 — policy steering (better = matches the backend safety rails)

`steering/build_steering.py` generates a small dialog set whose replies mirror
`backend/services/crisis_policy.py` behavior: L1 validation, L2
self-harm support-and-redirect (EN/DE/FR), medical emergencies → emergency
services, injection/jailbreak warm refusals, not-a-therapist identity
framing, PII hygiene. Stage-2 continues from the stage-1 adapter on
`steering×3 + 1200-dialog replay` mix (`data/stage2_sft.jsonl`) so empathy
fluency is not lost:

    .venv\Scripts\python steering\build_steering.py
    .venv\Scripts\python train_qlora.py --data data/stage2_sft.jsonl \
        --init-adapter adapter --epochs 1 --lr 5e-5 --accum 8 --out adapter_stage2

## Behavioral gate

    .venv/Scripts/python evaluate_ft.py --adapter adapter_stage2 --gpu

Replies are generated and checked (support-and-redirect present, no
methods/medical coaching, no prompt leak). Pass/fail report →
`adapter_stage2/behavioral_eval.json`.

## Fully unattended chain

`restart_train_guard.py` (launched detached) drives everything: auto-restarts
stage-1 if it crashes (resuming from `adapter/last_ckpt/`), then automatically
runs stage-2 → behavioral eval → fp16 merge. Logs: `guard.log`,
`train_qlora.log`, per-step `<tag>.out.log` files. After an OS reboot only the
guard launch needs re-running; all other steps resume automatically.

## Deploy

    .venv\Scripts\python merge_adapter.py --adapter adapter_stage2 --out merged_fp16
    .venv\Scripts\python serve_model.py --model merged_fp16 --port 8902

`merged_fp16` is a normal Qwen2.5-3B model folder (fp16 safetensors).
`serve_model.py` loads it in 4-bit NF4 (≈3GB VRAM) and serves
`POST /generate {"prompt", "max_new_tokens"}` on 127.0.0.1 with optional
`--token`. Point the backend gateway at it:

    # backend .env
    SERVED_MODEL_URL=http://127.0.0.1:8902
    SERVE_MODEL_TOKEN=            # must match serve_model.py --token if set

Gateway chain order: cloud → external HF fallback → **served local fine-tune**
→ slm-training code-prose artifact (tier `local`) → canned stub. Safety gates
always wrap the model (input backstops + streaming output backstops) — the
served tier changes wording quality only, not the policy layer.

## License note

EmpatheticDialogues is cc-by-nc-4.0; any merged artifact inherits that
restriction for its *memory* of this data. ESConv is CC BY-NC-SA (nonperm —
same non-commercial caveat). Keep this fine-tune out of any commercial deploy
path, or replace ED with a commercially-licensed dialog corpus and retrain.
