# Quickstart

Full pipeline, from an empty directory to generated samples.

## 0. Requirements

- Python 3.9+
- ~2 GB disk for the corpus and binaries, plus ~1 GB per checkpoint
- A GitHub token for the search API (`gh auth login`, or export `GITHUB_TOKEN`)

```bash
pip install -r requirements.txt
```

CUDA is optional. The pipeline detects a GPU and uses it automatically; with no
GPU it trains on CPU, which is slower but works unattended.

---

## 1. Collect the data (~1-3 hours)

```bash
python scrape_github.py --target-tokens 120000000 --min-stars 1000
```

Downloads top-starred Python and JavaScript repos and extracts docstrings,
comments and README prose into `raw_data.txt`.

Safe to interrupt — `scrape_state.json` records finished repos, so re-running
the same command resumes without duplicating anything. Add `--fresh` to start
over.

---

## 2. Tokenize (~1-5 minutes)

```bash
python preprocess.py
```

Produces `train.bin` and `val.bin` (flat `uint16` GPT-2 BPE token streams) plus
`preprocess_stats.json` with token counts and a record-length histogram.

---

## 3. Check throughput before committing (~2 minutes)

```bash
python train.py --benchmark 20
```

Prints seconds/step, tokens/s, and a projected wall-clock time for the full
10,000-step run. Do this first — on CPU the full run takes days, and this tells
you exactly how many.

If the projection is too long, shrink the model:

```bash
python train.py --benchmark 20 --n-embd 384 --n-layer 6 --n-head 6
```

---

## 4. Train (hours to days)

```bash
python train.py
```

Run it detached so it survives a closed terminal:

```bash
nohup python train.py > train_run.log 2>&1 &
```

Monitor it:

```bash
tail -50 train.log
```

Expect a line every 100 steps, a validation pass every 500, and a checkpoint
every 1000. Re-running `python train.py` after any interruption resumes from
the newest checkpoint automatically — no flag needed.

---

## 5. Evaluate

```bash
python evaluate.py
```

Writes `evaluation_report.json` and renders `loss_curves.png`.

---

## 6. Generate

```bash
python inference.py
```

Runs the five test prompts against `best_model.pt` and writes
`inference_samples.txt`. For a single prompt:

```bash
python inference.py --prompt "def calculate_" --tokens 60
```

---

## Troubleshooting

**`codeload throttled (429)`** — GitHub is rate-limiting archive downloads. The
scraper backs off and skips the repo after four attempts; occasional lines are
normal. A continuous stream means you should pause and resume later.

**Training is slower than the benchmark suggested** — something else is using
the CPU. The benchmark and the real run must not overlap with the scraper.

**`train.bin/val.bin missing`** — run `preprocess.py` first.

**Non-finite loss** — handled automatically: the run reloads the best weights
and cuts the LR tenfold, aborting only after five such events. If it keeps
happening, lower `--learning-rate`.

**Out of memory** — handled automatically by halving the batch size and doubling
gradient accumulation. To pre-empt it, start with
`--batch-size 2 --grad-accum-steps 16`.
