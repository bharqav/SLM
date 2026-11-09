"""Distil Qwen2.5-0.5B-Instruct answers into a JSONL corpus.

For every instruction in a source dataset (Alpaca by default) we ask Qwen for
a completion and record it.  The resulting JSONL is mixed into the training
corpus by ``preprocess_clean.py --distill``.

Usage:
    python distill_qwen.py --limit 64 --batch-size 16          # benchmark
    python distill_qwen.py --batch-size 32 --out qwen_distill.jsonl
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly and "
    "concisely, in at most 3 sentences. State the answer first, then only the "
    "detail the user actually needs. Do not add extra commentary."
)


def build_prompt(tok, item):
    """Turn an Alpaca-style record into a Qwen chat prompt.

    The chat template is used rather than a bare instruction because Qwen is
    an instruct model: with a system message it keeps its answers short, and a
    short answer is one the SLM can imitate without being cut off.
    """
    instr = (item.get("instruction") or "").strip()
    inp = (item.get("input") or "").strip()
    body = f"{instr}\n\n{inp}" if inp else instr
    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="alpaca_data.json")
    ap.add_argument("--out", default="qwen_distill.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole dataset")
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="skip this many records (resume a partial run)",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument(
        "--keep-truncated",
        action="store_true",
        help="keep answers that hit --max-new without an EOS (they end mid-sentence)",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="sample instead of greedy decoding (more varied "
        "answers, less deterministic)",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    if args.limit:
        data = data[args.start : args.start + args.limit]
    elif args.start:
        data = data[args.start :]
    print(f"distil: {len(data)} items from {args.data}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    done = 0
    dropped = 0
    t0 = time.time()
    new_tokens = 0
    out_fh = open(args.out, "w", encoding="utf-8")

    for start in range(0, len(data), args.batch_size):
        batch = data[start : start + args.batch_size]
        prompts = [build_prompt(tok, it) for it in batch]
        enc = tok(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(args.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=args.sample,
                temperature=args.temperature if args.sample else None,
                top_p=args.top_p if args.sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        # Rows that ran to the cap never emitted EOS, so their text is cut
        # mid-sentence.  Training on those teaches the model to trail off, so
        # drop them unless the caller explicitly wants them.
        lengths = (gen != tok.pad_token_id).sum(dim=1).tolist()
        hit_cap = [n >= args.max_new for n in lengths]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for it, text, capped in zip(batch, texts, hit_cap):
            if capped and not args.keep_truncated:
                dropped += 1
                continue
            rec = {
                "instruction": it.get("instruction", ""),
                "input": it.get("input", ""),
                "output": text.strip(),
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        done += len(batch)
        new_tokens += int(sum(lengths))
        elapsed = time.time() - t0
        if done % (args.batch_size * 4) == 0 or done == len(data):
            rate = done / elapsed if elapsed else 0.0
            print(
                f"  {done}/{len(data)}  {rate:.1f} items/s  "
                f"{new_tokens / elapsed:.0f} tok/s  dropped {dropped}  "
                f"elapsed {elapsed:.0f}s",
                flush=True,
            )

    out_fh.close()
    print(
        f"wrote {args.out} ({done - dropped} records, {dropped} truncated dropped)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
