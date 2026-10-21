# Training Report: Code-Aware SLM

## Summary

An 81.6M parameter GPT-2 style transformer was trained from scratch on 132.3M tokens of docstrings, code comments and README prose scraped from 600 top-starred GitHub repositories. Training ran for 10000 steps on CUDA, finishing with a training loss of 2.5313 and a validation loss of 2.9816 (perplexity 19.72). The run met every convergence criterion.

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

## Run

| Metric | Value |
| --- | --- |
| Device | CUDA |
| Torch threads | 8 |
| Steps completed | 10000 / 10000 |
| Stop reason | reached max_steps |
| Wall-clock time | 1h00m |
| Throughput | 22397.1 tokens/s |
| Tokens seen | 81.9M |
| Non-finite loss events | 0 |
| OOM events | 0 |

## Results

| Metric | Value |
| --- | --- |
| Final training loss | 2.5313 |
| Final validation loss | 2.9816 |
| Best validation loss | 2.9816 (step 10000) |
| Training perplexity | 12.57 |
| Validation perplexity | 19.72 |
| Total loss reduction | 1.9601 |
| Monotonic fraction | 1.0 |

### Convergence criteria

| Criterion | Result |
| --- | --- |
| train loss below 3.0 | PASS |
| val loss below 3.5 | PASS |
| no divergence | PASS |
| loss plateaued | PASS |
| generally decreasing | PASS |

### Loss history (train / validation)

| Step | Loss |
| --- | --- |
| 500 | 4.4914 / 4.7404 |
| 1,000 | 3.9144 / 4.2380 |
| 1,500 | 3.5639 / 3.9428 |
| 2,000 | 3.3106 / 3.7165 |
| 2,500 | 3.1550 / 3.5522 |
| 3,000 | 3.0385 / 3.4455 |
| 3,500 | 2.9682 / 3.3675 |
| 4,000 | 2.8899 / 3.3150 |
| 4,500 | 2.8359 / 3.2496 |
| 5,000 | 2.7835 / 3.2045 |
| 5,500 | 2.7455 / 3.1632 |
| 6,000 | 2.6984 / 3.1322 |
| 6,500 | 2.6562 / 3.0864 |
| 7,000 | 2.6198 / 3.0580 |
| 7,500 | 2.5945 / 3.0379 |
| 8,000 | 2.5679 / 3.0153 |
| 8,500 | 2.5526 / 2.9994 |
| 9,000 | 2.5433 / 2.9898 |
| 9,500 | 2.5353 / 2.9849 |
| 10,000 | 2.5313 / 2.9816 |
