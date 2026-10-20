# Training Report: Code-Aware SLM

## Summary

An n/a parameter GPT-2 style transformer is being trained from scratch on 132.3M tokens of docstrings, code comments and README prose scraped from 600 top-starred GitHub repositories. **Training is still in progress** -- the figures below reflect the run so far.

## Data

| Metric | Value |
| --- | --- |
| Repositories scraped | 600 succeeded, 19 skipped |
| Languages | Python, JavaScript/TypeScript |
| Minimum stars | 1,000 |
| Records extracted | 2,539,693 |
| Raw corpus size | 473.1 MB |
| Total tokens | 132.3M |
| Train / validation | 119.1M / 13.2M |
| Characters per token | 3.576 |
| Mean record length | 52.1 tokens |
| Median record length | 25 tokens |

### Record composition

| Source | Records |
| --- | --- |
| Python comments | 1,028,319 (40.5%) |
| Python docstrings (with signature) | 822,946 (32.4%) |
| JavaScript/TypeScript comments | 468,749 (18.5%) |
| JSDoc blocks (with declaration) | 212,175 (8.4%) |
| README prose | 7,504 (0.3%) |

### Record length distribution

| Bucket | Records |
| --- | --- |
| 0-8 tokens | 316,114 |
| 8-16 tokens | 582,403 |
| 16-32 tokens | 563,774 |
| 32-64 tokens | 593,324 |
| 64-128 tokens | 288,915 |
| 128-256 tokens | 130,527 |
| 256-512 tokens | 43,338 |
| 512-1024 tokens | 14,234 |
| 1024-inf tokens | 7,065 |

## Model

| Property | Value |
| --- | --- |
| Parameters | 81,564,800 |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Context length | 256 |
| Layers | 10 |
| Attention heads | 10 |
| Embedding dimension | 640 |
| Head dimension | 64 |
| Dropout | 0.1 |
| Weight tying | token embedding shared with output projection |

## Training configuration

| Hyperparameter | Value |
| --- | --- |
| Optimizer | AdamW (betas 0.9, 0.95) |
| Peak learning rate | 0.0006 |
| Minimum learning rate | 1e-05 |
| Schedule | linear warmup 1000 steps, then cosine decay |
| Weight decay | 0.1 (dim >= 2 tensors only) |
| Gradient clipping | 1.0 |
| Micro-batch | 4 sequences |
| Gradient accumulation | 8 |
| Effective batch | 32 sequences (8192 tokens) |
| Max steps | 10000 |
| Tokens over full run | 81.9M |
