"""
Phase 1.2: Validate the scraped corpus before spending days training on it.

Checks size, encoding, record quality, source diversity and repetition, then
writes validation_report.json.

The size floor defaults to the corpus this project actually targets rather than
the 500 MB in the original spec -- see --min-mb.

Usage
-----
    python validate_data.py [--min-mb 300] [--max-mb 3000] [--sample 2000]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slmlog import log, warn, error, phase, human

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "raw_data.txt")
STATE_PATH = os.path.join(HERE, "scrape_state.json")
STATS_PATH = os.path.join(HERE, "scrape_stats.json")
REPORT_PATH = os.path.join(HERE, "validation_report.json")

RECORD_SEP = "<|endoftext|>"
MAX_INVALID_RATE = 0.05
MIN_MEAN_RECORD_CHARS = 20

# Structural scaffolding -- comment delimiters and rule lines. These open a
# large share of records by construction (every reconstructed JSDoc block
# starts "/**"), so repetition has to be measured on the first line of actual
# content or the metric just reports the extractor's own formatting.
DELIMITER_RE = __import__("re").compile(
    r"^([/*#\-=_~+.<>|\s]*|\"{3}|'{3}|/\*+|\*+/|%+)$")


def content_head(rec, width=60):
    """First line of a record that carries content rather than scaffolding."""
    for line in rec.split("\n"):
        s = line.strip()
        if s and not DELIMITER_RE.match(s):
            return s[:width]
    return rec.split("\n", 1)[0][:width]


def iter_records(path):
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
    if buf and "\n".join(buf).strip():
        yield "\n".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mb", type=float, default=300.0)
    ap.add_argument("--max-mb", type=float, default=3000.0)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    phase("Phase 1.2: Data Validation")
    if not os.path.exists(RAW_PATH):
        error("raw_data.txt not found.")
        return 1

    checks = {}
    report = {}

    # ------------------------------------------------------------ size check
    size_mb = os.path.getsize(RAW_PATH) / 1e6
    checks["size_within_bounds"] = args.min_mb <= size_mb <= args.max_mb
    report["size_mb"] = round(size_mb, 1)
    log("Corpus size: {:.1f} MB (bounds {:.0f}-{:.0f} MB)".format(
        size_mb, args.min_mb, args.max_mb))

    # ------------------------------------------- full pass: counts + dupes
    total = 0
    total_chars = 0
    by_first_line = collections.Counter()
    by_content_head = collections.Counter()
    hashes = set()
    duplicates = 0
    null_bytes = 0
    nonascii_heavy = 0
    short = 0

    for rec in iter_records(RAW_PATH):
        total += 1
        total_chars += len(rec)
        if "\x00" in rec:
            null_bytes += 1
        if len(rec) < 10:
            short += 1
        if sum(1 for c in rec if ord(c) < 128) / max(1, len(rec)) < 0.90:
            nonascii_heavy += 1
        h = hashlib.blake2b(rec.encode("utf-8"), digest_size=16).digest()
        if h in hashes:
            duplicates += 1
        else:
            hashes.add(h)
        by_first_line[rec.split("\n", 1)[0][:60]] += 1
        by_content_head[content_head(rec)] += 1

    if total == 0:
        error("Corpus contains no records.")
        return 1

    mean_chars = total_chars / total
    dup_rate = duplicates / total
    invalid = null_bytes + short + nonascii_heavy
    invalid_rate = invalid / total

    report.update({
        "records": total,
        "total_chars": total_chars,
        "mean_record_chars": round(mean_chars, 1),
        "estimated_tokens": int(total_chars / 3.6),
        "duplicate_records": duplicates,
        "duplicate_rate": round(dup_rate, 5),
        "null_byte_records": null_bytes,
        "too_short_records": short,
        "non_ascii_heavy_records": nonascii_heavy,
        "invalid_rate": round(invalid_rate, 5),
    })

    checks["mean_record_length_ok"] = mean_chars > MIN_MEAN_RECORD_CHARS
    checks["invalid_rate_under_5pct"] = invalid_rate <= MAX_INVALID_RATE
    checks["no_null_bytes"] = null_bytes == 0
    checks["low_duplication"] = dup_rate < 0.01

    log("Records: {:,} | mean {:.1f} chars | ~{} tokens".format(
        total, mean_chars, human(report["estimated_tokens"])))
    log("Invalid: {} null-byte, {} too short, {} non-ASCII-heavy ({:.2%} of records)"
        .format(null_bytes, short, nonascii_heavy, invalid_rate))
    log("Exact duplicates: {} ({:.3%})".format(duplicates, dup_rate))

    # ------------------------------------------------------ sampled decode
    random.seed(args.seed)
    n_sample = min(args.sample, total)
    picks = set(random.sample(range(total), n_sample))
    decode_failures = 0
    sample_lengths = []
    with open(RAW_PATH, "rb") as fh:
        raw_bytes = fh.read(2_000_000)
    try:
        raw_bytes.decode("utf-8")
        head_decodes = True
    except UnicodeDecodeError:
        # A clean cut may split a multi-byte char; retry ignoring the tail.
        try:
            raw_bytes[:-4].decode("utf-8")
            head_decodes = True
        except UnicodeDecodeError:
            head_decodes = False

    for i, rec in enumerate(iter_records(RAW_PATH)):
        if i in picks:
            sample_lengths.append(len(rec))
            try:
                rec.encode("utf-8").decode("utf-8")
            except UnicodeError:
                decode_failures += 1

    checks["utf8_decodable"] = head_decodes and decode_failures == 0
    report["sample_size"] = n_sample
    report["sample_mean_chars"] = round(sum(sample_lengths) / max(1, len(sample_lengths)), 1)
    report["sample_decode_failures"] = decode_failures
    log("Sampled {} records: mean {:.1f} chars, {} decode failures".format(
        n_sample, report["sample_mean_chars"], decode_failures))

    # ------------------------------------------------------ source diversity
    state = {}
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
    stats = {}
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, encoding="utf-8") as fh:
            stats = json.load(fh)

    n_from_stats = stats.get("repos_ok") or 0
    n_from_state = len(state.get("done_repos", []))
    if n_from_state >= n_from_stats:
        by_kind, n_repos = state.get("by_kind") or {}, n_from_state
    else:
        by_kind, n_repos = stats.get("by_kind") or {}, n_from_stats
    kinds_present = {k for k, v in by_kind.items() if v > 0}

    checks["many_source_repos"] = n_repos >= 20
    checks["mixed_record_kinds"] = len(kinds_present) >= 3
    report["source_repos"] = n_repos
    report["record_kinds"] = by_kind

    log("Sources: {} repositories | record kinds: {}".format(
        n_repos, ", ".join(sorted(kinds_present)) or "unknown"))

    if by_kind:
        tot_k = sum(by_kind.values()) or 1
        for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            log("  {:<14} {:>9,} ({:.1f}%)".format(k, v, 100 * v / tot_k))
        dominant = max(by_kind.values()) / tot_k
        checks["no_single_kind_dominates"] = dominant < 0.95
        report["dominant_kind_share"] = round(dominant, 4)

    # ------------------------------------------------------------ repetition
    raw_top = by_first_line.most_common(5)
    top = by_content_head.most_common(10)
    top_share = top[0][1] / total if top else 0.0
    checks["no_dominant_repeated_line"] = top_share < 0.02
    report["top_repeated_content_heads"] = [
        {"text": t, "count": c, "share": round(c / total, 5)} for t, c in top]
    report["top_raw_first_lines"] = [
        {"text": t, "count": c, "share": round(c / total, 5)} for t, c in raw_top]
    log("Most repeated content line appears {} times ({:.3%} of records)".format(
        top[0][1] if top else 0, top_share))
    for t, c in top[:5]:
        log("  {:>7,}x  {!r}".format(c, t[:60]))
    log("(raw opening lines, scaffolding included: {})".format(
        ", ".join("{!r} x{:,}".format(t[:12], c) for t, c in raw_top[:3])))

    # ---------------------------------------------------------------- verdict
    report["checks"] = checks
    report["passed"] = all(checks.values())
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log("")
    log("Validation checks:")
    for name, ok in checks.items():
        log("  [{}] {}".format("PASS" if ok else "FAIL", name))

    if report["passed"]:
        log("RESULT: corpus passed all validation checks.")
        return 0
    failed = [k for k, v in checks.items() if not v]
    warn("RESULT: validation issues: {}".format(", ".join(failed)))
    # Size and diversity shortfalls are advisory; corruption is not.
    fatal = {"utf8_decodable", "no_null_bytes", "invalid_rate_under_5pct",
             "mean_record_length_ok"}
    if fatal & set(failed):
        error("Fatal data-quality failure; do not train on this corpus.")
        return 1
    warn("Non-fatal shortfalls only; corpus is usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
