# slm-ft: QLoRA fine-tune of Qwen2.5-3B-Instruct for empathetic dialogue

**Status (20 Oct 2026): trained to completion.**
Stage-1: 2,928 steps / 2 epochs, final val loss **2.3947** (23,427 dialogs).
Stage-2 policy steering: steering×9 + 500 replay, 192 steps, val 2.47 to 2.69.
Behavioral gate: **5/5 probes pass** (`adapter_stage2/behavioral_eval.json`, crisis
support-and-redirect, medical to emergency services, injection refusal, warm benign
replies). Deployment artifact: `merged_fp16/` (6.17GB fp16 safetensors, 434 tensors
verified).

Turns the pre-trained Qwen2.5-3B-Instruct base into a supportive-conversation assistant
with a LoRA adapter trained on public empathetic-dialogue corpora. This is the **emotional-SLM
story**: unlike `slm-training/` (from-scratch 81.6M GPT trained on code prose), the artifact
produced here is intended to power the chat pipeline's `main` tier once wired to the backend
gateway.

## Setup (done once)

    py -V:Astral/CPython3.12.14 -m venv .venv
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\pip install transformers peft datasets accelerate bitsandbytes

Verified working on the RTX 5060 Laptop GPU (Blackwell, sm_120): torch 2.11.0+cu128,
bitsandbytes 0.50.2 4-bit NF4 kernels run natively.

## Data

1. EmpatheticDialogues (facebookresearch tarball, ~23k short empathetic exchanges,
   **cc-by-nc-4.0**, non-commercial only):

       Invoke-WebRequest https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz -OutFile ed.tar.gz
       tar -xzf ed.tar.gz -C data\ed_docraw

2. ESConv (thu-coai/esconv from the HF hub, 910 multi-turn emotional-support dialogs), fetched
   automatically by `prepare_data.py`.

3. Build the canonical chat corpus:

       .venv\Scripts\python prepare_data.py --ed-csv-dir data\ed_docraw
       # -> data/sft_train.jsonl + data/sft_val.jsonl (~23.4k / ~0.5k dialogs)

Conversations are hash-split before writing, so no dialog leaks into val.

## Training

    .venv\Scripts\python train_qlora.py --epochs 2 --out adapter

Defaults: 4-bit NF4 base, bf16 compute, LoRA r=16 alpha=32 on all attention + MLP projections,
seq len 1024, batch 1 x accum 16, AdamW 1e-4 cosine with 3% warmup, gradient checkpointing.
VRAM peak is about 5.5 to 6.5 GB, which fits the 8.5 GB card (leave the `slm-training` training
run idle or finished before launching).

Loss is masked to assistant turns only (`-100` labels on the prompt prefix), using Qwen's native
chat template. Intermediate `train_log.json` records the loss curve; the adapter is checkpointed
after each epoch's val pass.

## Stage 2, policy steering (better = matches the backend safety rails)

`steering/build_steering.py` generates a small dialog set whose replies mirror the expected safety rails: L1 validation, L2 self-harm support-and-redirect
(EN/DE/FR), medical emergencies to emergency services, injection/jailbreak warm refusals,
not-a-therapist identity framing, PII hygiene. Stage-2 continues from the stage-1 adapter on
`steeringx3 + 1200-dialog replay` mix (`data/stage2_sft.jsonl`) so empathy fluency is not lost:

    .venv\Scripts\python steering\build_steering.py
    .venv\Scripts\python train_qlora.py --data data/stage2_sft.jsonl ^
        --init-adapter adapter --epochs 1 --lr 5e-5 --accum 8 --out adapter_stage2

## Behavioral gate

The behavioral eval (`adapter_stage2/behavioral_eval.json`) is committed as an
artifact. To re-run it, generate samples from the adapter and apply the
5-probe safety checklist (crisis support-and-redirect, medical to emergency
services, injection refusal, warm benign replies, PII hygiene) using the same
criteria. Pass/fail report is written to `adapter_stage2/behavioral_eval.json`.

## Deploy

    .venv\Scripts\python merge_adapter.py --adapter adapter_stage2 --out merged_fp16

`merged_fp16` is a normal Qwen2.5-3B model folder (fp16 safetensors). Load it
with any standard HuggingFace inference path (e.g. `transformers`, vLLM, or
`text-generation-webui`) or serve via the project's `slm_api.py` FastAPI server.

## Reproducibility

## License note

EmpatheticDialogues is cc-by-nc-4.0; any merged artifact inherits that restriction for its memory
of this data. ESConv is CC BY-NC-SA (non-permissive, same non-commercial caveat). Keep this
fine-tune out of any commercial deploy path, or replace ED with a commercially-licensed dialog
corpus and retrain.
