"""Build the empathetic-SFT corpus for the Qwen2.5-3B QLoRA fine-tune.

Sources (downloaded/verified by hand, see README):
- ESConv (thu-coai/esconv on HF): 910 multi-turn emotional-support dialogs (JSON in `text`).
- EmpatheticDialogues (facebookresearch tarball): ~25k short open-domain empathetic
  dialogs across 32 emotion contexts. NOTE: cc-by-nc-4.0 -- non-commercial use.

Output: slm-ft/data/sft_train.jsonl + slm-ft/data/sft_val.jsonl
Schema per line: {"messages": [{"role", "content"}, ...]}  (user/assistant pairs)

Conversation-level hash split (no dialog leakage between train/val).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ESCONV_HF = "thu-coai/esconv"
ED_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

MAX_TURNS = 24          # keep dialogues manageable
MIN_USER_TURNS = 2      # need at least one user+assistant exchange pair
VAL_FRACTION = 0.02


def clean(text: str) -> str:
    text = text.replace("_comma_", ",")
    text = text.replace("\t", " ").strip()
    return text


def build_esconv() -> list[list[dict]]:
    """ESConv JSON dialogs: speakers are 'seeker' and 'supporter'."""
    from datasets import load_dataset

    ds = load_dataset(ESCONV_HF, split="train")
    dialogs = []
    for row in ds:
        obj = json.loads(row["text"])
        turns = []
        for u in obj.get("dialog", []):
            text = clean(u.get("text", ""))
            speaker = u.get("speaker", "")
            if not text:
                continue
            if speaker == "sys":
                role = "assistant"
            elif speaker == "usr":
                role = "user"
            else:
                role = None
            if role:
                turns.append({"role": role, "content": text})
        # collapse duplicates of same role sequentially
        merged = []
        for t in turns:
            if merged and merged[-1]["role"] == t["role"] == "assistant":
                merged[-1]["content"] += " " + t["content"]
            else:
                merged.append(dict(t))
        if _ok(merged):
            dialogs.append(merged)
    truncated = []
    for d in dialogs:
        truncated.append(d[:MAX_TURNS])
    return truncated


def _ok(turns: list[dict]) -> bool:
    ut = sum(1 for t in turns if t["role"] == "user")
    at = sum(1 for t in turns if t["role"] == "assistant")
    return ut >= MIN_USER_TURNS and at >= MIN_USER_TURNS


def trim_user_tail(turns: list[dict]) -> list[dict]:
    """SFT dialogs must END on an assistant turn; drop trailing user says."""
    while turns and turns[-1]["role"] == "user":
        turns.pop()
    return turns


def build_ed(csv_dir: Path) -> list[list[dict]]:
    """ED CSVs grouped by conv_id, roles from speaker_idx (0=user, 1=assistant)."""
    dialogs = []
    base_header = [
        "conv_id", "utterance_idx", "context", "prompt",
        "speaker_idx", "utterance", "selfeval", "tags",
    ]
    for name in ("train", "valid", "test"):
        path = csv_dir / f"{name}.csv"
        if not path.exists():
            sys.exit(f"missing EmpatheticDialogues CSV at {path}; run download step in README")
        by_conv: dict[str, dict] = {}
        with open(path, encoding="utf-8", newline="") as fh:
            first_line = fh.readline()
            first = next(csv.reader([first_line]))
            if first and first[0].strip() == "conv_id":
                rows: list[list[str]] = list(csv.reader(fh))
                header = [h.strip() for h in first]
            else:
                # the fbaipublicfiles ed tarball ships train.csv headerless
                rows = list(csv.reader([first_line])) + list(csv.reader(fh))
                header = base_header
            col = {c: i for i, c in enumerate(header)}
            drop_selfflagged = 0
            for parts in rows:
                if len(parts) < max(col["utterance"] + 1, col["speaker_idx"] + 1):
                    continue
                row = {c: parts[i].strip() for c, i in col.items() if i < len(parts)}
                if row.get("selfeval", "") and "not" in row["selfeval"].lower():
                    drop_selfflagged += 1
                    continue  # drop unproficient dialogues
                text = clean(row["utterance"])
                if not text or len(text) < 2:
                    continue
                conv = by_conv.setdefault(
                    row["conv_id"], {"prompt": "", "turns": []}
                )
                conv["prompt"] = conv["prompt"] or row.get("prompt", "")
                conv["turns"].append((int(row["speaker_idx"]), int(row["utterance_idx"]), text))
        for conv in by_conv.values():
            conv["turns"].sort(key=lambda x: x[1])
            first_speaker = conv["turns"][0][0]
            speakers = {spk for spk, _, _ in conv["turns"]}
            if len(speakers) != 2:
                continue  # never count multi-party threads
            second_speakers = sorted(speakers - {first_speaker})
            if not second_speakers:
                continue
            role = {first_speaker: "user", second_speakers[0]: "assistant"}
            merged = []
            for spk, _, text in conv["turns"]:
                t = {"role": role[spk], "content": text}
                if merged and merged[-1]["role"] == t["role"] == "assistant":
                    merged[-1]["content"] += " " + t["content"]
                else:
                    merged.append(t)
            if _ok(merged):
                dialogs.append(merged[:MAX_TURNS])
    return dialogs


def hash_split(dialogs: list[list[dict]], val_fraction: float = VAL_FRACTION):
    train, val = [], []
    for d in dialogs:
        h = int(hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8], 16)
        (val if (h % 100) < val_fraction * 100 else train).append(d)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-csv-dir", required=True, help="dir with train/valid/test.csv (EmpatheticDialogues)")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    es = build_esconv()
    ed = build_ed(Path(args.ed_csv_dir))
    print(f"ESConv dialogs: {len(es)} | EmpatheticDialogues dialogs: {len(ed)}")

    train, val = hash_split(es + ed)
    print(f"total: train={len(train)} val={len(val)}")
    with open(out / "sft_train.jsonl", "w", encoding="utf-8") as fh:
        for d in train:
            fh.write(json.dumps({"messages": d}, ensure_ascii=False) + "\n")
    with open(out / "sft_val.jsonl", "w", encoding="utf-8") as fh:
        for d in val:
            fh.write(json.dumps({"messages": d}, ensure_ascii=False) + "\n")

    with open(out / "corpus_stats.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"esconv": len(es), "empathetic_dialogues": len(ed), "train": len(train), "val": len(val)},
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
