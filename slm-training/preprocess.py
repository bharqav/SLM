"""
Phase 2: Tokenize raw_data.txt into flat uint16 binaries for training.

Reads the record-separated corpus produced by scrape_github.py, encodes each
record with the GPT-2 BPE (tiktoken, 50257 vocab), appends an <eos> after every
record, and writes a sequential 90/10 train/val split.

On <pad> and <unk>
------------------
Byte-level BPE is lossless and training packs a contiguous token stream, so
neither a padding nor an unknown token is ever emitted. The only special token
used is <|endoftext|> (id 50256), which serves as <eos>.

Usage
-----
    python preprocess.py [--val-split 0.1] [--seed 1337]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time

from slmlog import error, fmt_dur, human, log, phase, warn

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "raw_data.txt")
TRAIN_BIN = os.path.join(HERE, "train.bin")
VAL_BIN = os.path.join(HERE, "val.bin")
META_PATH = os.path.join(HERE, "tokenizer_meta.json")
STATS_PATH = os.path.join(HERE, "preprocess_stats.json")

RECORD_SEP = "<|endoftext|>"
EOS_ID = 50256
VOCAB_SIZE = 50257
BATCH = 2000  # records per tiktoken batch call
MAX_FAILURE_RATE = 0.01


def get_tokenizer():
    """GPT-2 BPE, with a word-level fallback if tiktoken cannot be loaded."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        log(f"Tokenizer: tiktoken gpt2 (n_vocab={enc.n_vocab})")
        return ("tiktoken", enc)
    except Exception as e:
        warn(f"tiktoken unavailable ({e}); falling back to word-level.")
        return ("word", None)


def iter_records(path):
    """Yield records from the separator-delimited corpus."""
    buf = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == RECORD_SEP:
                rec = "\n".join(buf)
                buf = []
                if rec.strip():
                    yield rec
            else:
                buf.append(line)
    if buf:
        rec = "\n".join(buf)
        if rec.strip():
            yield rec


def build_word_vocab(path, limit=VOCAB_SIZE):
    """Fallback path: frequency-ranked word vocabulary."""
    from collections import Counter

    counter = Counter()
    for rec in iter_records(path):
        counter.update(rec.split())
    words = [w for w, _ in counter.most_common(limit - 3)]
    stoi = {w: i + 3 for i, w in enumerate(words)}
    stoi["<pad>"], stoi["<unk>"], stoi["<eos>"] = 0, 1, 2
    return stoi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    phase("Phase 2: Preprocessing Started")
    if not os.path.exists(RAW_PATH):
        error("raw_data.txt not found. Run scrape_github.py first.")
        return 1

    raw_bytes = os.path.getsize(RAW_PATH)
    log(f"Input: raw_data.txt ({raw_bytes / 1e6:.1f} MB)")

    kind, enc = get_tokenizer()
    if kind == "word":
        stoi = build_word_vocab(RAW_PATH)
        eos_id = stoi["<eos>"]
        unk_id = stoi["<unk>"]
    else:
        eos_id = EOS_ID
        unk_id = None

    t0 = time.time()
    chunks = []
    n_records = n_tokens = n_failed = 0
    lengths = []
    batch = []

    def flush(batch):
        nonlocal n_records, n_tokens, n_failed, chunks
        if not batch:
            return
        try:
            if kind == "tiktoken":
                encoded = enc.encode_ordinary_batch(batch, num_threads=4)
            else:
                encoded = [[stoi.get(w, unk_id) for w in r.split()] for r in batch]
        except Exception as e:
            # Fall back to per-record encoding so one bad record cannot
            # discard a whole batch.
            warn(f"batch encode failed ({e}); retrying individually")
            encoded = []
            for r in batch:
                try:
                    encoded.append(
                        enc.encode_ordinary(r)
                        if kind == "tiktoken"
                        else [stoi.get(w, unk_id) for w in r.split()]
                    )
                except Exception:
                    n_failed += 1
                    encoded.append([])
        flat = []
        for ids in encoded:
            if not ids:
                continue
            ids.append(eos_id)
            flat.extend(ids)
            lengths.append(len(ids))
            n_records += 1
        if flat:
            arr = np.array(flat, dtype=np.uint16)
            chunks.append(arr)
            n_tokens += arr.size

    for rec in iter_records(RAW_PATH):
        batch.append(rec)
        if len(batch) >= BATCH:
            flush(batch)
            batch = []
            if n_records % 200000 < BATCH:
                log(
                    f"  tokenized {human(n_records)} records, {human(n_tokens)} tokens ({time.time() - t0:.0f}s)"
                )
    flush(batch)

    if n_records == 0:
        error("No records tokenized.")
        return 1

    fail_rate = n_failed / max(1, n_records + n_failed)
    if fail_rate > MAX_FAILURE_RATE:
        error(
            f"Encoding failure rate {fail_rate:.2%} exceeds {MAX_FAILURE_RATE:.0%} budget."
        )
        return 1

    tokens = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    del chunks
    assert tokens.max() < VOCAB_SIZE, "token id exceeds vocab size"
    log(
        f"Tokenized {human(n_tokens)} tokens from {human(n_records)} records in {fmt_dur(time.time() - t0)}"
    )

    # Sequential split: validation is the tail of the corpus, so no training
    # record leaks into it via the shared repo it came from.
    n_val = int(len(tokens) * args.val_split)
    n_train = len(tokens) - n_val
    train, val = tokens[:n_train], tokens[n_train:]

    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)

    lengths = np.array(lengths)
    hist_edges = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 10**9]
    hist = {}
    for i in range(len(hist_edges) - 1):
        lo, hi = hist_edges[i], hist_edges[i + 1]
        cnt = int(((lengths >= lo) & (lengths < hi)).sum())
        hist["{}-{}".format(lo, "inf" if hi > 10**8 else hi)] = cnt

    # Integrity: re-read from disk and compare against what we wrote.
    tr = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    va = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    ok = (
        len(tr) == n_train
        and len(va) == n_val
        and len(tr) + len(va) == len(tokens)
        and int(tr[0]) == int(train[0])
        and int(va[-1]) == int(val[-1])
    )
    if not ok:
        error("Integrity check FAILED on train.bin/val.bin.")
        return 1

    stats = {
        "tokenizer": kind,
        "vocab_size": VOCAB_SIZE,
        "eos_id": eos_id,
        "records": n_records,
        "failed_records": n_failed,
        "failure_rate": round(fail_rate, 6),
        "total_tokens": int(n_tokens),
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "val_split": args.val_split,
        "mean_tokens_per_record": round(float(lengths.mean()), 2),
        "median_tokens_per_record": int(np.median(lengths)),
        "p95_tokens_per_record": int(np.percentile(lengths, 95)),
        "max_tokens_per_record": int(lengths.max()),
        "length_histogram": hist,
        "raw_bytes": raw_bytes,
        "chars_per_token": round(raw_bytes / max(1, n_tokens), 3),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tokenizer": kind,
                "encoding": "gpt2",
                "vocab_size": VOCAB_SIZE,
                "eos_id": eos_id,
            },
            fh,
            indent=2,
        )

    log(
        f"train.bin: {human(n_train)} tokens ({n_train * 2 / 1e6:.1f} MB) | val.bin: {human(n_val)} tokens ({n_val * 2 / 1e6:.1f} MB)"
    )
    log(
        "Record length: mean {:.1f}, median {}, p95 {} tokens".format(
            stats["mean_tokens_per_record"],
            stats["median_tokens_per_record"],
            stats["p95_tokens_per_record"],
        )
    )
    log(f"Length histogram: {json.dumps(hist)}")
    log("Integrity check passed. Phase 2: Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
