"""Build the empathetic-dialogue corpus for the from-scratch SLM (81.6M GPT).

Consumes the same validated dialogs the QLoRA pipeline uses
(slm-ft/data/sft_*_train.jsonl) and renders each dialog to a tagged text
stream so the tiny GPT learns turn structure alongside supportive phrasing:

    <sys>{persona}</sys>\n<user>...</user>\n<assistant>...</assistant>\n\n<user>...

Dialogs are separated with an EOS token (50256); the training stream is
shuffled at dialog level. Writes uint16 gpt2-token streams:
    empathetic_run/train.bin   (+ val.bin from the holdout split)

Run AFTER corpora exist; safe to run while the QLoRA stage-1 occupies the GPU
(pure CPU work).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SFT_DIR = PROJECT / "slm-ft" / "data"
OUT_DIR = PROJECT / "slm-training" / "empathetic_run"

SYSTEM_PROMPT = (
    "You are a caring peer-support companion. Listen first, validate feelings, "
    "suggest healthy coping steps. You are not a licensed therapist; never give "
    "self-harm methods or medication advice; if someone may be in immediate "
    "danger, urge them to contact local emergency services or a crisis line now."
)

MAX_SYSTEM_CHARS = 800
EOS = 50256


def load_dialogs(path: Path) -> list[list[dict]]:
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            if msgs and msgs[-1]["role"] == "assistant":
                dialogs.append(msgs)
    return dialogs


def render_dialog(msgs: list[dict]) -> str:
    sys_text = ""
    body = []
    for m in msgs:
        if m["role"] == "system":
            sys_text = m["content"]
        else:
            tag = "user" if m["role"] == "user" else "assistant"
            body.append(f"<{tag}>{m['content'].strip()}")
    if not sys_text or not body or body[0].startswith("<user>") is False:
        # always lead with the persona block and first user turn
        sys_text = sys_text or SYSTEM_PROMPT
    if not sys_text.startswith(SYSTEM_PROMPT):
        sys_text = SYSTEM_PROMPT
    text = "<sys>" + sys_text[:MAX_SYSTEM_CHARS] + "</sys>\n\n" + "\n\n".join(body)
    if not text.endswith("</sys>\n\n"):
        pass
    return text


def main() -> None:
    train_dialogs = load_dialogs(SFT_DIR / "sft_train.jsonl")
    val_dialogs = load_dialogs(SFT_DIR / "sft_val.jsonl")
    if not train_dialogs:
        sys.exit("no SFT dialogs found in slm-ft/data; run slm-ft/prepare_data.py")
    rng = random.Random(1337)
    rng.shuffle(train_dialogs)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, dialogs in (("train.bin", train_dialogs), ("val.bin", val_dialogs)):
        ids: list[int] = []
        for d in dialogs:
            toks = enc.encode(render_dialog(d), allowed_special=set())
            ids.extend(toks)
            ids.append(EOS)
        arr = None
        import numpy as np

        arr = np.asarray(ids, dtype=np.uint16)
        out = OUT_DIR / name
        arr.tofile(out)
        print(f"{name}: {len(dialogs)} dialogs -> {len(ids):,} tokens ({out} {out.stat().st_size/1e6:.1f} MB)")

    print("persona sample:", render_dialog(train_dialogs[0])[:220].replace("\n", " | "))


if __name__ == "__main__":
    main()
