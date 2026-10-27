"""QLoRA fine-tune of Qwen2.5-3B-Instruct on the empathetic-SFT corpus.

Target hardware: 8.5GB VRAM laptop GPU (RTX 5060, Blackwell sm_120).
Loads base in 4-bit NF4 (bitsandbytes), trains LoRA adapters only
(gradient checkpointing on), loss masked to assistant turns.

Usage:
    .venv/Scripts/python train_qlora.py \
        --data data/sft_train.jsonl --val data/sft_val.jsonl \
        --epochs 2 --out adapter/

Artifacts: <out>/adapter_model.safetensors + train_log.json (loss curve).
Merge to fp16 later via merge_adapter.py if you deploy full weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from prepare_data import trim_user_tail
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE = "Qwen/Qwen2.5-3B-Instruct"
FULL_SEQ_LEN = 1024


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--val", default="data/sft_val.jsonl")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=FULL_SEQ_LEN)
    ap.add_argument("--limit", type=int, default=0, help="cap #dialogs (smoke test)")
    ap.add_argument(
        "--init-adapter",
        default=None,
        help="continue training from this adapter dir (stage-2)",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="resume an interrupted run (loads adapter + step state)",
    )
    args = ap.parse_args()
    return args


def build_example(tok, messages, max_len):
    """Assistant-only loss masking: prompt (everything before final turn) -100."""
    final = messages[-1]
    assert final["role"] == "assistant"
    prompt_messages = messages[:-1]
    prompt_text = tok.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    reply_text = final["content"] + tok.eos_token
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tok(reply_text, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + reply_ids
    labels = [-100] * len(prompt_ids) + list(reply_ids)
    if len(ids) > max_len:  # left-truncate keeping the tail (reply)
        cut = len(ids) - max_len
        ids, labels = ids[cut:], labels[cut:]
    return {"input_ids": ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, path, tok, max_len, limit=0):
        self.rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                msgs = trim_user_tail(list(msgs))
                if (
                    not msgs
                    or msgs[-1]["role"] != "assistant"
                    or sum(1 for m in msgs if m["role"] == "user") < 1
                ):
                    continue
                self.rows.append(build_example(tok, msgs, max_len))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for ids, labels, attn in loader:
        ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
        mask = labels != -100
        ntok = mask.sum().item()
        tot += out.loss.item() * ntok
        n += ntok
    model.train()
    return tot / max(1, n)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    steps_done = 0
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        base_dir = Path(args.resume_from)
        state_file = base_dir / "last_ckpt" / "train_state.json"
        if not state_file.exists():
            state_file = (
                base_dir / "train_state.json"
            )  # resume-from may point at last_ckpt itself
        if state_file.exists():
            steps_done = json.loads(state_file.read_text())["steps_done"]
        print(f"resuming from {args.resume_from} at step {steps_done}")
    elif args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print("continuing from adapter:", args.init_adapter)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = SFTDataset(args.data, tok, args.max_len, limit=args.limit)
    val_ds = SFTDataset(args.val, tok, args.max_len)
    log = {
        "config": vars(args),
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "steps": [],
    }
    print(f"train dialogs: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id),
        drop_last=False,
    )

    total_steps = int(len(train_loader) * args.epochs) // args.accum
    warmup = max(1, int(0.03 * total_steps))
    final_target = total_steps + steps_done
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
    )

    def lr_at(step: int) -> float:
        gstep = step + steps_done
        if gstep < warmup:
            return args.lr * (gstep + 1) / warmup
        p = (gstep - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + __import__("math").cos(p * 3.141592653589793))

    ckpt_dir = out / "last_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(tag_step: int) -> None:
        model.save_pretrained(ckpt_dir)
        ckpt_dir.joinpath("train_state.json").write_text(
            json.dumps({"steps_done": steps_done + tag_step}), encoding="utf-8"
        )
        with open(out / "train_log.json", "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2)
        print(f"checkpoint at local step {tag_step} -> {ckpt_dir}", flush=True)

    step = 0
    start = time.perf_counter()
    model.train()
    epoch = 0
    while step < total_steps:
        epoch += 1
        for i, (ids, labels, attn) in enumerate(train_loader):
            lr_now = lr_at(step)
            old_lrs = [g["lr"] for g in opt.param_groups]
            for g in opt.param_groups:
                g["lr"] = lr_now
            outm = model(
                input_ids=ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device),
            )
            outm.loss.backward()
            if (i + 1) % args.accum == 0 or i == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                for g, old in zip(opt.param_groups, old_lrs):
                    g["lr"] = old
                if step % 50 == 0 or step == total_steps:
                    print(
                        f"step {step + steps_done}/{final_target} loss {outm.loss.item():.4f} "
                        f"lr {lr_now:.2e} {time.perf_counter() - start:.0f}s",
                        flush=True,
                    )
                if step % 100 == 0 or step == total_steps:
                    log["steps"].append(
                        {
                            "step": step + steps_done,
                            "train_loss": round(outm.loss.item(), 4),
                            "lr": lr_now,
                            "elapsed_s": round(time.perf_counter() - start, 1),
                        }
                    )
                if step % 200 == 0 or step == total_steps:
                    checkpoint(step)
                if step >= total_steps:
                    break
        # epoch boundary: eval + save
        vl = eval_loss(model, val_loader, device)
        log["steps"].append({"epoch": epoch, "val_loss": round(vl, 4)})
        print(f"epoch {epoch} val_loss {vl:.4f}", flush=True)
        (out / "adapter_model.safetensors").unlink(missing_ok=True)
        model.save_pretrained(out)
        with open(out / "train_log.json", "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2)
    out.joinpath("TRAIN_COMPLETE").write_text("ok", encoding="utf-8")
    print("done; adapter saved to", out)


if __name__ == "__main__":
    main()
