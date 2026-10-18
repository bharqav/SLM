# Training Report: Code-Aware SLM

## Summary

An n/a parameter GPT-2 style transformer is being trained from scratch on 2.6M tokens of docstrings, code comments and README prose scraped from 10 top-starred GitHub repositories. **Training is still in progress** -- the figures below reflect the run so far.

## Data

| Metric | Value |
| --- | --- |
| Repositories scraped | 10 succeeded, 0 skipped |
| Languages | Python, JavaScript/TypeScript |
| Minimum stars | 1,000 |
| Records extracted | 22,774 |
| Raw corpus size | 3.7 MB |
| Total tokens | 2.6M |
| Train / validation | 2.3M / 256.8K |
| Characters per token | 3.687 |
| Mean record length | 49.47 tokens |
| Median record length | 21 tokens |

### Record composition

| Source | Records |
| --- | --- |
| JavaScript/TypeScript comments | 18,501 (81.2%) |
| JSDoc blocks (with declaration) | 3,287 (14.4%) |
| Python comments | 428 (1.9%) |
| Python docstrings (with signature) | 344 (1.5%) |
| README prose | 214 (0.9%) |

### Record length distribution

| Bucket | Records |
| --- | --- |
| 0-8 tokens | 6,304 |
| 8-16 tokens | 14,262 |
| 16-32 tokens | 12,957 |
| 32-64 tokens | 10,036 |
| 64-128 tokens | 4,830 |
| 128-256 tokens | 2,075 |
| 256-512 tokens | 953 |
| 512-1024 tokens | 296 |
| 1024-inf tokens | 189 |

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
