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
from guardrails import (
    check_input,
    check_output,
    clean_dialogue_artifacts,
    is_degenerate,
)
from model import GPT, GPTConfig
from quantize import QuantizationInfo, quantize_model  # noqa: E402
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

    old_cfg = ck.get("model_config") or {}
    cfg = GPTConfig(
        vocab_size=old_cfg.get("vocab_size", 50257),
        block_size=old_cfg.get("block_size", 256),
        n_layer=old_cfg.get("n_layer", 12),
        n_head=old_cfg.get("n_head", 12),
        n_embd=old_cfg.get("n_embd", 768),
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


USER_PREFIX = "User: "
# Turns are trained as "User: <text>\nAssistant: <answer>", and tiktoken merges
# the space after "Assistant:" into the answer's first token (" Hi" is a single
# token, not " " + "Hi").  The cue must therefore stop at the colon: a trailing
# space would append a standalone space token that never occurs in training,
# which scrambled the first-token distribution and made the model open with
# fragments like "ive" and loop.  benchmarks/evaluate_empathy.py already ends the
# cue at the colon.
ASSISTANT_SUFFIX = "\nAssistant:"

# Markers that begin a *following* turn.  A single response ends at the first
# one, so these double as the strings generation should stop at: the corpus is
# multi-turn and the model otherwise keeps writing the next turn, which is
# discarded anyway but costs time and can exhaust max_new_tokens.
TURN_MARKERS = ("\n" + USER_PREFIX, ASSISTANT_SUFFIX)

_CODE_PREFIXES = (
    "def ",
    "class ",
    "import ",
    "from ",
    "return ",
    "if ",
    "elif ",
    "for ",
    "while ",
    "async ",
    "await ",
    "const ",
    "let ",
    "var ",
    "function ",
    "public ",
    "private ",
    "void ",
    "int ",
    "std::",
    "SELECT ",
    "INSERT ",
    "UPDATE ",
    "DELETE ",
    "CREATE ",
    "#include",
    "# TODO",
)


def looks_like_code(text: str) -> bool:
    """Cheap heuristic: is this prompt a code fragment rather than a question?"""
    t = text.lstrip()
    if any(t.startswith(h) for h in _CODE_PREFIXES):
        return True
    stripped = text.rstrip()
    if stripped.endswith((":", "{", "(", "[")):
        return True
    return text.count("{") != text.count("}") or text.count("(") != text.count(")")


def format_prompt(text: str, mode: str) -> str:
    """Render a prompt in the format the model was trained on.

    Dialogue and QA were trained as ``User: <text>\\nAssistant: <answer>``, while
    code completion was trained on raw source. ``auto`` routes each prompt by
    shape so the default behaviour needs no flag.

    The assistant cue ends at the colon, with no trailing space: in the corpus
    that space belongs to the answer's first token (see ``ASSISTANT_SUFFIX``).
    """
    if mode == "code":
        return text
    if mode == "chat":
        return f"{USER_PREFIX}{text}{ASSISTANT_SUFFIX}"
    return text if looks_like_code(text) else f"{USER_PREFIX}{text}{ASSISTANT_SUFFIX}"


def truncate_at_next_turn(text: str) -> str:
    """Cut a completion at the start of the following turn.

    The dialogue corpus is multi-turn, so the model sometimes keeps writing after
    its own answer - either the *user's* reply or a second assistant turn. A
    single response should end there. Both markers are cut, whichever comes
    first one. The result is trimmed as well, because the first generated token
    carries the space the corpus format implies (" Hi" is one token).
    """
    cuts = [i for i in (text.find(m) for m in TURN_MARKERS) if i != -1]
    return text[: min(cuts)].strip() if cuts else text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_clean_model_v2.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help=(
            "Penalise tokens that already appeared in the generation. 1.0 is "
            "off; 1.2 breaks the loops this model falls into on open-ended "
            "prompts and matches the API default."
        ),
    )
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--mode",
        choices=["auto", "chat", "code"],
        default="auto",
        help=(
            "How to format the prompt before generation. 'chat' wraps it as "
            "'User: <text>\\nAssistant:' (the format the model was trained on "
            "for dialogue and QA); 'code' sends it verbatim for code "
            "completion; 'auto' picks per prompt from its shape."
        ),
    )
    ap.add_argument(
        "--quantize",
        choices=["int8", "int4", "gptq", "awq"],
        default=None,
        help="Post-training quantization method (requires CUDA for int8/int4).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    try:
        model, cfg, ck = load_model(args.checkpoint, device)
    except FileNotFoundError as e:
        error(str(e))
        return 1

    qinfo: QuantizationInfo | None = None
    if args.quantize:
        qinfo = QuantizationInfo(
            method=args.quantize,
            applied=False,
            device=device,
            original_params=sum(p.numel() for p in model.parameters()),
        )
        model, qinfo = quantize_model(
            model,
            method=args.quantize,
            device=device,
            checkpoint_path=args.checkpoint,
        )
        log(str(qinfo))

    enc = get_encoder()

    prompts = [args.prompt] if args.prompt else TEST_PROMPTS
    if not args.prompt:
        phase("Phase 4.2: Inference Testing")

    results, total_tokens, total_time = [], 0, 0.0
    for prompt in prompts:
        for s in range(args.samples):
            safety = check_input(prompt)
            if not safety.safe:
                warn(f"Refused prompt: {safety.reason}")
                continue

            fed = format_prompt(prompt, args.mode)
            ids = enc.encode_ordinary(fed)
            if not ids:
                ids = [enc.eot_token]
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            t0 = time.time()
            try:
                out = model.generate(
                    x,
                    args.tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    eos_token_id=enc.eot_token,
                )
            except Exception as e:
                warn(f"generation failed for {prompt!r}: {e}")
                continue
            dt = time.time() - t0
            text = enc.decode(out[0].tolist())
            completion = text[len(fed) :]
            completion = clean_dialogue_artifacts(completion)
            completion = truncate_at_next_turn(completion)
            total_tokens += args.tokens
            total_time += dt

            out_safety = check_output(completion)
            if not out_safety.safe:
                warn(f"Refused output: {out_safety.reason}")
                continue

            rep = repetition_ratio(completion)
            degenerate = is_degenerate(completion)
            results.append(
                {
                    "prompt": prompt,
                    "prompt_sent": fed,
                    "completion": completion,
                    "full": text,
                    "repetition_ratio": round(rep, 3),
                    "degenerate": degenerate,
                    "tokens_per_second": round(args.tokens / max(dt, 1e-6), 1),
                }
            )
            print("\n" + "=" * 70)
            print(f"PROMPT: {prompt!r}")
            if fed != prompt:
                print(f"SENT:   {fed!r}")
            print("-" * 70)
            print(text)
            print("-" * 70)
            print(
                f"{args.tokens} tokens in {dt:.1f}s ({args.tokens / max(dt, 1e-6):.1f} tok/s) | repeated 4-grams: {rep:.0%} | degenerate: {degenerate}"
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
