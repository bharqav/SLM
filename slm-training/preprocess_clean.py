#!/usr/bin/env python3
"""Build a clean, role-marked corpus for continued training.

Why this exists
---------------
The original empathy corpus (``empathy_train.bin``) was produced by flattening
``{"role": ..., "content": ...}`` messages into plain newline-joined text.  Two
problems followed:

1. The model never sees who is speaking.  A turn boundary is just a newline, so
   the model cannot learn "the assistant replies after the user".
2. The source dialogues contain runs of consecutive same-role messages (four
   user turns in a row, for example).  The model therefore learns that after a
   user turn another *user* turn is a valid continuation, which is exactly the
   non-sequitur behaviour seen at inference time.

On top of that the corpus is tiny (4.3 MB, ~2.1M tokens) for a 253M-757M model,
and it contains no general question/answer data at all, so the model cannot
answer "What is the capital of Australia?" even in principle.

This script rebuilds the corpus from two sources:

* the project's own peer-support dialogues (``slm-ft/data/sft_*.jsonl``), and
* the Alpaca instruction/response dataset (``alpaca_data.json``), which supplies
  the general QA ability the empathy corpus lacks.

An optional third source is a JSONL of Qwen-distilled answers
(``--distill``); those records use the same ``instruction``/``input``/
``output`` shape as Alpaca and are rendered identically.

The distillation is generated over the *same* Alpaca instructions, so a
distilled record and an Alpaca record can describe one prompt with two
different answers.  Training on both would teach the model two conflicting
continuations for a single prefix -- the human answer says one thing, Qwen's
says another, and the gradient averages them into a muddled style.  So when
``--distill`` is given, each distilled record *replaces* the Alpaca record with
the same ``(instruction, input)`` key instead of being added alongside it.  The
replacement set is built from all distilled records, including the validation
slice, so no prompt appears in both the training and validation streams.
``--keep-duplicate-alpaca`` restores the old additive behaviour.

Every turn is explicitly role-marked (``User:`` / ``Assistant:``), consecutive
same-role messages are merged into a single turn, and each conversation is
terminated with the end-of-text token (50256) so the model learns to stop
instead of rambling.

Usage
-----
    python preprocess_clean.py \
        --train-jsonl ../slm-ft/data/sft_train.jsonl \
        --val-jsonl   ../slm-ft/data/sft_val.jsonl \
        --alpaca      alpaca_data.json \
        --out-dir     .

Outputs ``clean_train.bin`` / ``clean_val.bin`` (uint16 token streams) and
``clean_stats.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import tiktoken

EOS = 50256  # end-of-text; the same separator the original pipeline used
USER_MARKER = "User: "
ASSISTANT_MARKER = "Assistant: "
NEWLINE = "\n"


# --------------------------------------------------------------------- helpers
def merge_consecutive_turns(messages: Sequence[dict]) -> list[tuple[str, str]]:
    """Collapse runs of same-role messages into single turns.

    The source data has dialogues where the same role speaks several times in a
    row (a quirk of how the conversations were scraped).  Left alone, those
    teach the model to keep talking as the user.  Merging them keeps the
    dialogue natural and makes the role alternation consistent.
    """
    turns: list[tuple[str, str]] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + " " + content)
        else:
            turns.append((role, content))
    return turns


def is_clean_dialogue(turns: Sequence[tuple[str, str]]) -> bool:
    """Reject dialogues that would teach the model bad habits."""
    if len(turns) < 2:
        return False
    # The assistant must reply to a user turn, so the dialogue has to start
    # with one.  Dialogues that open with the assistant are role-play scenarios
    # and are not what we want the model to imitate.
    if turns[0][0] != "user":
        return False
    # Every turn should carry some actual content.
    if any(len(text.split()) < 2 for _, text in turns):
        return False
    return True


def dialogue_tokens(turns: Sequence[tuple[str, str]], enc) -> list[int]:
    """Render a dialogue in the role-marked training format.

    ``User: ...\\nAssistant: ...\\nUser: ...\\nAssistant: `` with the final
    assistant turn terminated by EOS.
    """
    toks: list[int] = []
    for i, (role, content) in enumerate(turns):
        marker = USER_MARKER if role == "user" else ASSISTANT_MARKER
        toks.extend(enc.encode(marker + content))
        if i < len(turns) - 1:
            toks.extend(enc.encode(NEWLINE))
    toks.append(EOS)
    return toks


def alpaca_key(record: dict) -> tuple[str, str]:
    """Identity of an instruction pair, used to deduplicate Alpaca vs distill."""
    return (
        (record.get("instruction") or "").strip(),
        (record.get("input") or "").strip(),
    )


def alpaca_tokens(record: dict, enc, max_output_tokens: int) -> list[int] | None:
    """Render one Alpaca instruction/response pair in the same format."""
    instruction = (record.get("instruction") or "").strip()
    context = (record.get("input") or "").strip()
    output = (record.get("output") or "").strip()
    if not instruction or not output:
        return None
    # Long outputs are usually lists, essays or truncated code dumps; they teach
    # the model to be verbose rather than to the point, so drop them.
    if len(enc.encode(output)) > max_output_tokens:
        return None
    prompt = instruction if not context else f"{instruction}\n{context}"
    return (
        enc.encode(USER_MARKER + prompt)
        + enc.encode(NEWLINE)
        + enc.encode(ASSISTANT_MARKER + output)
        + [EOS]
    )


# ------------------------------------------------------------------------ main
def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Build a clean, role-marked corpus for continued training."
    )
    ap.add_argument(
        "--train-jsonl",
        default=str(here.parent / "slm-ft" / "data" / "sft_train.jsonl"),
    )
    ap.add_argument(
        "--val-jsonl", default=str(here.parent / "slm-ft" / "data" / "sft_val.jsonl")
    )
    ap.add_argument("--alpaca", default=str(here / "alpaca_data.json"))
    ap.add_argument(
        "--distill",
        default=None,
        help="optional JSONL of Qwen-distilled answers "
        "(instruction/input/output). Records replace the Alpaca "
        "record with the same instruction/input instead of being "
        "added alongside it",
    )
    ap.add_argument(
        "--keep-duplicate-alpaca",
        action="store_true",
        help="add distilled records on top of the Alpaca record with "
        "the same instruction instead of replacing it",
    )
    ap.add_argument(
        "--distill-val",
        type=int,
        default=1000,
        help="how many distilled records to hold out for validation",
    )
    ap.add_argument("--out-dir", default=str(here))
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=120,
        help="drop Alpaca answers longer than this (keeps replies to the point)",
    )
    ap.add_argument(
        "--val-alpaca",
        type=int,
        default=2000,
        help="number of Alpaca records held out for validation",
    )
    ap.add_argument(
        "--code-bin",
        default=None,
        help="optional raw code corpus (.bin, EOS-separated) appended to the "
        "training stream so the model can also do code completion",
    )
    ap.add_argument(
        "--code-tokens",
        type=int,
        default=0,
        help="how many tokens of the code corpus to append",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--max-dialogue-tokens",
        type=int,
        default=1024,
        help="drop dialogues longer than this (they are usually noise)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------ load sources
    def load_jsonl(path: Path) -> list[dict]:
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    train_rows = load_jsonl(Path(args.train_jsonl))
    val_rows = load_jsonl(Path(args.val_jsonl))
    with Path(args.alpaca).open(encoding="utf-8") as fh:
        alpaca = json.load(fh)
    print(
        f"Loaded {len(train_rows):,} train dialogues, {len(val_rows):,} val dialogues, "
        f"{len(alpaca):,} Alpaca records"
    )

    # ------------------------------------------------- build the token streams
    def build(
        rows: Iterable[dict],
        alpaca_records: Iterable[dict],
        distill_records: Iterable[dict],
        label: str,
    ):
        toks: list[int] = []
        n_dialogues = 0
        n_turns = 0
        n_merged = 0
        n_alpaca = 0
        n_distill = 0
        n_dropped = 0
        n_replaced = 0

        for row in rows:
            messages = row.get("messages") if isinstance(row, dict) else row
            if not messages:
                n_dropped += 1
                continue
            raw_turns = [(m.get("role", ""), m.get("content", "")) for m in messages]
            n_merged += sum(
                1
                for a, b in zip(raw_turns, raw_turns[1:])
                if a[0] == b[0] and a[0] in {"user", "assistant"}
            )
            turns = merge_consecutive_turns(messages)
            if not is_clean_dialogue(turns):
                n_dropped += 1
                continue
            seq = dialogue_tokens(turns, enc)
            if len(seq) > args.max_dialogue_tokens:
                n_dropped += 1
                continue
            toks.extend(seq)
            n_dialogues += 1
            n_turns += len(turns)

        for record in alpaca_records:
            if replace_alpaca and alpaca_key(record) in distill_keys:
                n_replaced += 1
                continue
            seq = alpaca_tokens(record, enc, args.max_output_tokens)
            if seq is None:
                n_dropped += 1
                continue
            toks.extend(seq)
            n_alpaca += 1

        for record in distill_records:
            seq = alpaca_tokens(record, enc, args.max_output_tokens)
            if seq is None:
                n_dropped += 1
                continue
            toks.extend(seq)
            n_distill += 1

        print(
            f"  {label}: {n_dialogues:,} dialogues ({n_turns:,} turns, "
            f"{n_merged:,} merged same-role runs), {n_alpaca:,} Alpaca pairs, "
            f"{n_distill:,} distilled pairs, {n_dropped:,} dropped "
            f"-> {len(toks):,} tokens"
        )
        if replace_alpaca:
            print(
                f"  {label}: {n_replaced:,} Alpaca pairs replaced by their "
                f"distilled answer"
            )
        return toks, {
            "dialogues": n_dialogues,
            "turns": n_turns,
            "merged_same_role_runs": n_merged,
            "qa_pairs": n_alpaca,
            "distilled_pairs": n_distill,
            "dropped": n_dropped,
            "tokens": len(toks),
            "replaced_by_distill": n_replaced,
        }

    # Hold out a random slice of Alpaca for validation.
    order = rng.permutation(len(alpaca))
    val_idx = set(order[: args.val_alpaca].tolist())
    train_alpaca = [r for i, r in enumerate(alpaca) if i not in val_idx]
    val_alpaca = [alpaca[i] for i in sorted(val_idx)]

    # Distilled Qwen answers are an optional extra stream; hold out a slice for
    # validation the same way.
    distill: list[dict] = []
    train_distill: list[dict] = []
    val_distill: list[dict] = []
    if args.distill:
        distill = load_jsonl(Path(args.distill))
        order_d = rng.permutation(len(distill))
        dv_idx = set(order_d[: args.distill_val].tolist())
        train_distill = [r for i, r in enumerate(distill) if i not in dv_idx]
        val_distill = [distill[i] for i in sorted(dv_idx)]
        print(f"Loaded {len(distill):,} distilled records")

    # A distilled record supersedes the Alpaca record for the same instruction,
    # so build the replacement set over the *whole* distillation (train and val
    # slices alike) to keep a prompt from leaking across the split.
    distill_keys = {alpaca_key(r) for r in distill}
    replace_alpaca = bool(distill_keys) and not args.keep_duplicate_alpaca

    train_toks, train_stats = build(train_rows, train_alpaca, train_distill, "train")
    val_toks, val_stats = build(val_rows, val_alpaca, val_distill, "val")

    # Raw code is a *completion* distribution, not a dialogue, so it is appended
    # verbatim (it is already EOS-separated).  The model distinguishes the two
    # modes by prompt shape: a bare code prefix continues in code, a
    # "User:/Assistant:" prefix continues in dialogue.
    if args.code_bin and args.code_tokens:
        code_path = Path(args.code_bin)
        if not code_path.exists():
            raise FileNotFoundError(f"--code-bin not found: {code_path}")
        code = np.fromfile(code_path, dtype=np.uint16, count=args.code_tokens).tolist()
        train_toks.extend(code)
        train_stats["code_tokens"] = len(code)
        print(f"  train: appended {len(code):,} code tokens from {code_path.name}")

    # ------------------------------------------------------------------ write
    train_path = out_dir / "clean_train.bin"
    val_path = out_dir / "clean_val.bin"
    train_path.write_bytes(np.asarray(train_toks, dtype=np.uint16).tobytes())
    val_path.write_bytes(np.asarray(val_toks, dtype=np.uint16).tobytes())

    stats = {
        "train": train_stats,
        "val": val_stats,
        "eos_token": EOS,
        "format": "User: <text>\\nAssistant: <text>",
        "sources": {
            "dialogues_train": args.train_jsonl,
            "dialogues_val": args.val_jsonl,
            "alpaca": args.alpaca,
            "distill": args.distill,
        },
        "max_output_tokens": args.max_output_tokens,
        "max_dialogue_tokens": args.max_dialogue_tokens,
        "code_bin": args.code_bin,
        "code_tokens": args.code_tokens,
        "seed": args.seed,
    }
    (out_dir / "clean_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(
        f"Wrote {train_path} ({train_path.stat().st_size / 1e6:.1f} MB) and "
        f"{val_path} ({val_path.stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
