# slm-ft: QLoRA fine-tune of Qwen2.5-0.5B-Instruct for empathetic dialogue

**Status (2026-11-10): trained to completion.** The numbers below are the recorded runs.  
The weight files (`adapter_model.safetensors`, `merged_fp16/model.safetensors`) are
generated artifacts and are gitignored, so a fresh clone carries configs, tokenizers and
metadata only — re-run the steps below to regenerate the weights.
Stage-1: 2,928 steps / 2 epochs, final val loss **2.7650** (23,427 train / 461 val dialogs).  
Stage-2 policy steering: steering×9 + 500 replay (770 dialogs), 96 steps, val loss **2.7655**
(final train loss 3.1131).  
Behavioral gate: **5/5 probes pass** (`adapter_stage2/behavioral_eval.json`, crisis  
support-and-redirect, medical to emergency services, injection refusal, warm benign  
replies, PII hygiene). That record is dated 20 Oct 2026 and predates the 2026-11-10
stage-2 retrain, so re-run the gate before publishing.
Deployment artifact: `merged_fp16/` — real merged fp16 weights (988,097,536 bytes) with a
`config.json` matching the Qwen2.5-0.5B-Instruct base (hidden_size 896, 24 layers).

Turns the pre-trained Qwen2.5-0.5B-Instruct base into a supportive-conversation assistant
with a LoRA adapter trained on public empathetic-dialogue corpora. This is the **emotional-SLM
story**: unlike `slm-training/` (from-scratch 24L/24H/1536D GPT, ~757M params, trained on
code prose), the artifact
produced here is intended to power the chat pipeline's `main` tier once wired to the backend
gateway.

All commands in this document are run from the repository root. Every step also
has a Makefile shortcut: `make ft-download-ed`, `ft-data`, `ft-steering`,
`ft-train`, `ft-train2`, `ft-merge`, or `ft-all` for the whole pipeline.

## Setup (done once)

    py -V:Astral/CPython3.12.14 -m venv .venv
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\pip install transformers peft datasets accelerate bitsandbytes

Verified working on the RTX 5060 Laptop GPU (Blackwell, sm_120): torch 2.11.0+cu128,
bitsandbytes 0.50.2 4-bit NF4 kernels run natively.

## Data

1. EmpatheticDialogues (facebookresearch tarball, ~23k short empathetic exchanges,
   **cc-by-nc-4.0**, non-commercial only):

       python scripts\download_empathetic_dialogues.py --output-dir slm-ft/data/ed_docraw

2. ESConv (thu-coai/esconv from the HF hub, 910 multi-turn emotional-support dialogs), fetched
   automatically by `prepare_data.py`.

3. Build the canonical chat corpus:

       .venv\Scripts\python slm-ft/prepare_data.py --ed-csv-dir slm-ft/data/ed_docraw
       # -> slm-ft/data/sft_train.jsonl + slm-ft/data/sft_val.jsonl (~23.4k / ~0.5k dialogs)

Conversations are hash-split before writing, so no dialog leaks into val.

## Training

    .venv\Scripts\python slm-ft/train_qlora.py --epochs 2 --out slm-ft/adapter \
        --lora-r 8 --lora-alpha 8 --target-modules q_proj,v_proj

Defaults: 4-bit NF4 base, bf16 compute, LoRA r=16 alpha=32 on attention + MLP projections
(q/k/v/o_proj, gate/up/down_proj),
seq len 1024, batch 1 x accum 16, AdamW 1e-4 cosine with 3% warmup, gradient checkpointing.
VRAM peak is about 5.5 to 6.5 GB, which fits the 8.5 GB card (leave the `slm-training` training
run idle or finished before launching).

Both training commands need an interpreter with `bitsandbytes`. The root project
environment deliberately does not carry it on Windows (see the platform marker in
`requirements.txt`); on this host the working interpreter is
`slm-ft\.venv\Scripts\python.exe`. `train_qlora.py` checks for it at startup and exits
with the interpreter to use and the install command, instead of failing later inside
transformers' quantizer dispatch.

The LoRA flags above reproduce the committed stage-1 adapter (r=8/alpha=8 on q_proj,v_proj).
The committed stage-2 adapter records the same config: `--init-adapter` loads the stage-1
adapter and PEFT carries its config forward, so the CLI's own
`--lora-r/--lora-alpha/--target-modules` values (defaults r=16/alpha=32 on
q/k/v/o/gate/up/down) are ignored. `train_log.json` still records the CLI defaults as run
metadata, which is not what the adapter uses.

Loss is masked to assistant turns only (`-100` labels on the prompt prefix), using Qwen's native
chat template. Intermediate `train_log.json` records the loss curve; the adapter is checkpointed
after each epoch's val pass.

## Stage 2, policy steering (better = matches the backend safety rails)

`steering/build_steering.py` generates a small dialog set whose replies mirror the expected safety rails: L1 validation, L2 self-harm support-and-redirect
(EN/DE/FR), medical emergencies to emergency services, injection/jailbreak warm refusals,
not-a-therapist identity framing, PII hygiene. Stage-2 continues from the stage-1 adapter on
`steeringx9 + 500-dialog replay` mix (`data/stage2_sft.jsonl`) so empathy fluency is not lost:

    .venv\Scripts\python slm-ft/steering/build_steering.py
    .venv\Scripts\python slm-ft/train_qlora.py --data slm-ft/data/stage2_sft.jsonl \
        --init-adapter slm-ft/adapter --epochs 1 --lr 5e-5 --accum 8 --out slm-ft/adapter_stage2

## Behavioral gate

The behavioral eval (`adapter_stage2/behavioral_eval.json`) is committed as an
artifact. To re-run it, generate samples from the adapter and apply the
5-probe safety checklist (crisis support-and-redirect, medical to emergency
services, injection refusal, warm benign replies, PII hygiene) using the same
criteria. Pass/fail report is written to `adapter_stage2/behavioral_eval.json`.

## Deploy

    .venv\Scripts\python slm-ft/merge_adapter.py --adapter slm-ft/adapter_stage2 --out slm-ft/merged_fp16

`merged_fp16` is a Qwen2.5-0.5B model folder: real merged fp16 weights
(`model.safetensors`, 988,097,536 bytes) plus a `config.json` matching the
Qwen2.5-0.5B-Instruct base (hidden_size 896, 24 layers). An earlier revision carried a
346-byte JSON stub and stale 3B-era metadata (hidden_size=2048, 36 layers); the 2026-11-10
merge replaced both. The weights are gitignored, so a fresh clone must re-run the merge to
get them. Load with any standard HuggingFace inference path (e.g. `transformers`, vLLM, or
`text-generation-webui`) or serve via the project's `slm_api.py` FastAPI server.

## License note

EmpatheticDialogues is cc-by-nc-4.0; any merged artifact inherits that restriction for its memory
of this data. ESConv is CC BY-NC-SA (non-permissive, same non-commercial caveat). Keep this
fine-tune out of any commercial deploy path, or replace ED with a commercially-licensed dialog
corpus and retrain.
