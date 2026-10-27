"""Phase 4.2: Generate text from a trained checkpoint.

Usage
-----
    python inference.py                                  # run the PRD test suite
    python inference.py --prompt "def calculate_" --tokens 60
    python inference.py --checkpoint final_model.pt --temperature 0.8
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig
from slmlog import error, log, phase, warn

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_PATH = os.path.join(HERE, "inference_samples.txt")

TEST_PROMPTS = [
    "def calculate_",
    "# TODO:",
    "class DataProcessor:",
    "async function fetch",
    """This function""",
]


def load_model(checkpoint, device):
    path = checkpoint if os.path.isabs(checkpoint) else os.path.join(HERE, checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)

    old_cfg = ck["model_config"]
    cfg = GPTConfig(
        vocab_size=old_cfg["vocab_size"],
        block_size=old_cfg.get("block_size", 1024),
        n_layer=old_cfg.get("n_layer", 8),
        n_head=old_cfg.get("n_head", 4),
        n_embd=old_cfg.get("n_embd", 1024),
        dropout=old_cfg.get("dropout", 0.0),
    )

    model = GPT.from_config(cfg)
    # The checkpoint was saved by the same model.py, so state_dict keys match
    # natively (transformer.wte, transformer.wpe, transformer.h.N.*, etc.).
    # No key remapping is needed.
    state = ck["model"]
    model_state = model.state_dict()
    load_state = {k: v for k, v in state.items() if k in model_state}
    missing = set(model_state.keys()) - set(load_state.keys())
    if missing:
        warn(f"Missing {len(missing)} keys when loading checkpoint")
    model.load_state_dict(load_state, strict=False)
    model.to(device).eval()

    step = ck.get("step", "?")
    best = ck.get("best_val_loss")
    log(
        "Loaded {} (step {}, best val loss {})".format(
            os.path.basename(path),
            step,
            f"{best:.4f}" if isinstance(best, float) else best,
        )
    )
    return model, cfg, ck


def get_encoder():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def repetition_ratio(text):
    """Fraction of repeated 4-grams -- a cheap degeneracy signal."""
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_model.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    try:
        model, cfg, ck = load_model(args.checkpoint, device)
    except FileNotFoundError as e:
        error(str(e))
        return 1
    enc = get_encoder()

    prompts = [args.prompt] if args.prompt else TEST_PROMPTS
    if not args.prompt:
        phase("Phase 4.2: Inference Testing")

    results, total_tokens, total_time = [], 0, 0.0
    for prompt in prompts:
        for s in range(args.samples):
            ids = enc.encode_ordinary(prompt)
            if not ids:
                ids = [enc.eot_token]
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            t0 = time.time()
            try:
                out = model.generate(
                    x, args.tokens, temperature=args.temperature, top_k=args.top_k
                )
            except Exception as e:
                warn(f"generation failed for {prompt!r}: {e}")
                continue
            dt = time.time() - t0
            text = enc.decode(out[0].tolist())
            completion = text[len(prompt) :]
            total_tokens += args.tokens
            total_time += dt

            rep = repetition_ratio(completion)
            results.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "full": text,
                    "repetition_ratio": round(rep, 3),
                    "tokens_per_second": round(args.tokens / max(dt, 1e-6), 1),
                }
            )
            print("\n" + "=" * 70)
            print(f"PROMPT: {prompt!r}")
            print("-" * 70)
            print(text)
            print("-" * 70)
            print(
                f"{args.tokens} tokens in {dt:.1f}s ({args.tokens / max(dt, 1e-6):.1f} tok/s) | repeated 4-grams: {rep:.0%}"
            )

    if not results:
        error("No samples generated.")
        return 1

    if not args.prompt:
        with open(SAMPLES_PATH, "w", encoding="utf-8") as fh:
            fh.write("Inference samples\n")
            fh.write(f"checkpoint: {args.checkpoint}\n")
            fh.write("step: {}\n".format(ck.get("step")))
            fh.write("best_val_loss: {}\n".format(ck.get("best_val_loss")))
            fh.write(
                f"temperature: {args.temperature} | top_k: {args.top_k} | max_new_tokens: {args.tokens}\n"
            )
            fh.write("=" * 70 + "\n")
            for r in results:
                fh.write("\nPROMPT: {!r}\n".format(r["prompt"]))
                fh.write("-" * 70 + "\n")
                fh.write(r["full"] + "\n")
                fh.write("-" * 70 + "\n")
                fh.write("repeated 4-grams: {:.0%}\n".format(r["repetition_ratio"]))

        high_rep = [r for r in results if r["repetition_ratio"] > 0.5]
        log("")
        log(f"Generated {len(results)} samples -> inference_samples.txt")
        log(
            f"Mean inference speed: {total_tokens / max(total_time, 1e-6):.1f} tokens/s on {device}"
        )
        if high_rep:
            warn(
                f"{len(high_rep)}/{len(results)} samples show heavy repetition (>50% repeated 4-grams)"
            )
        else:
            log("No sample shows degenerate repetition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
