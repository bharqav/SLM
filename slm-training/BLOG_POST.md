# I Trained an 81M-Parameter Language Model on a Laptop CPU

> **Draft status.** Everything except the Results section is final. Metrics
> marked `TK` are filled in when training completes — run `python make_report.py`
> and copy the figures from `TRAINING_REPORT.md`. Do not publish with `TK`
> markers in place.

---

Most small language models are trained on general web text. I wanted to try
something narrower: train one entirely on the prose that developers write
*about* code — docstrings, comments, and README text.

No GPU. No cloud bill. An Intel i7-1065G7 with four cores and an integrated
graphics chip that PyTorch cannot train on. Setup took an afternoon; the
training run took `TK` days in the background.

Here is what happened.

## Why code comments?

Three reasons this dataset is more interesting than it first sounds.

**It carries structure.** A Python docstring is preceded by a signature; a JSDoc
block is followed by a declaration. If you extract them as pairs rather than as
loose text, the model sees the relationship:

```python
def calculate_distance(x1, y1, x2, y2):
    """Return the Euclidean distance between two points."""
```

That pairing is the whole point. A comment on its own is just English. A comment
attached to the thing it describes is a supervised signal about what code does.

**It is unusually clean.** Comments in repositories with thousands of stars have
been through code review. Compared to a web crawl, the noise floor is very low —
most of the filtering work is removing license headers and generated-file
banners, not spam.

**It is a distinct distribution.** The register sits between English and code:
identifiers, type names, imperative descriptions of behaviour. That is exactly
what you want for docstring completion or comment generation, and it is
underrepresented in general-purpose training sets.

## Collecting the data

The scraper walks top-starred Python and JavaScript repositories, streams each
one's tarball, and extracts three kinds of record:

- Python docstrings, paired with a signature reconstructed from the AST
- JSDoc blocks, paired with the declaration that follows
- Standalone comment runs, with consecutive lines grouped into one record

Python extraction uses `ast` and `tokenize`, so it is exact. JavaScript uses
regular expressions, which is where the interesting failures live.

**The first thing that broke** was comment detection. A naive `/\*[\s\S]*?\*/`
pattern happily matches a `/*` that appears *inside a string literal*, swallowing
everything up to the next `*/` — which meant chunks of raw code entered the
corpus as "comments." Two fixes: anchor the block pattern to the start of a line,
and add a rejector that discards any record where most lines look like code
(trailing semicolons, braces, arrow functions, `const`/`return` openers).

**The second thing that broke** was GitHub rate limiting. Downloading through
`api.github.com/repos/{repo}/tarball` returns a 302 to `codeload.github.com` —
and `requests` helpfully forwards your `Authorization` header through the
redirect. Codeload's secondary rate limiter does not like that at all. The
scraper stalled at ten repositories, sleeping 60 seconds at a time, while the
core API quota sat completely untouched at 5000/5000.

The fix was to stop being clever: use the token for the search API only, and pull
archives straight from `codeload.github.com` on a separate, unauthenticated
session. Throughput went from zero to roughly a repository every few seconds.

Final corpus: `TK` records from `TK` repositories, about `TK` tokens.

## The model

Standard GPT-2, deliberately:

| | |
| --- | --- |
| Parameters | 81,564,800 |
| Layers | 10 |
| Heads | 10 (64 dims each) |
| Embedding | 640 |
| Context | 256 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |

Pre-norm blocks, learned positional embeddings, a 4× GELU MLP, and weight tying
between the token embedding and the output projection.

**Tying matters enormously at this scale.** The token embedding is 32.2M
parameters — 39% of the entire model — purely because a 50,257-entry vocabulary
is enormous relative to a 640-dimensional model. Sharing it with the output
projection saves those parameters and regularises the model at the same time.

That vocabulary is also the single biggest thing standing between this run and a
faster one, which brings us to the part that surprised me most.

## What actually costs time on a CPU

Before committing to a multi-day run, I benchmarked it: `train.py --benchmark 20`
times twenty real optimizer steps and projects the full run.

The projection came back at roughly **`TK` seconds per step**. At 10,000 steps of
8,192 tokens each — 82M tokens total — that is `TK` days.

Working out where the time goes is clarifying. The final projection from 640
dimensions to 50,257 vocabulary entries costs about as much arithmetic as all ten
transformer blocks combined. On a GPU that matmul disappears into the hardware.
On four CPU cores it is roughly half your wall-clock time.

So the biggest available speedup for CPU training is not a smaller model — it is
a **smaller vocabulary**. A domain-specific 8K-token vocabulary trained on this
corpus would cut the output projection by 6× and barely hurt quality, since the
text is a narrow slice of English plus identifiers. I did not do this, because
the spec called for GPT-2 BPE. It is the first thing I would change.

A second, smaller lesson: **do not benchmark while something else is running.**
My first measurement was taken with the scraper still going and came out about
30% pessimistic. That is the difference between a plausible plan and an
implausible one.

## Surviving a multi-day run

A run measured in days on a laptop *will* be interrupted — sleep, updates,
someone closing a terminal. The training loop assumes this:

- Checkpoints carry model weights, optimizer state, step count, and the full
  loss history. Resuming is the default: run `train.py` again and it picks up
  from the newest checkpoint with no flag.
- A corrupt newest checkpoint falls back to the previous one automatically.
- A non-finite loss reloads the best weights and cuts the learning rate tenfold,
  aborting only after five such events.
- An allocation failure halves the batch size and doubles gradient accumulation,
  preserving the effective batch so the hyperparameters stay honest.
- `Ctrl-C` checkpoints and exits cleanly rather than throwing away hours.

Only the three most recent checkpoints are kept. At roughly a gigabyte each —
weights plus two AdamW moments — that matters more than you would expect.

## Results

> `TK` — fill from `TRAINING_REPORT.md` after the run completes.

| Metric | Value |
| --- | --- |
| Final training loss | `TK` |
| Final validation loss | `TK` |
| Validation perplexity | `TK` |
| Training time | `TK` |
| Throughput | `TK` tokens/s |
| Inference speed | `TK` tokens/s on CPU |

`TK` — loss curve chart (`loss_curves.png`)

`TK` — generated samples for the five test prompts

## What I would do differently

- **Train a domain vocabulary first.** The 50K GPT-2 vocabulary is the dominant
  cost on CPU and is mostly wasted on a corpus this narrow.
- **Split by repository, not by position.** Validation is currently the tail of
  the corpus, so a repository straddling the boundary appears on both sides. A
  repo-level split would give a cleaner generalisation estimate.
- **Measure before choosing the model size.** The parameter count in my original
  plan was off by more than 2× — the configuration I had written down was a 30M
  model, not the 75M I thought I had specified. Ten minutes with the arithmetic
  beforehand would have saved that.

## Try it

The whole pipeline is six Python files and runs on any machine with Python and
about 2 GB of disk:

```bash
pip install -r requirements.txt
python scrape_github.py --target-tokens 120000000
python preprocess.py
python train.py --benchmark 20    # check the projection first
python train.py
```

Repository: `TK` — link
Model weights: `TK` — release link
