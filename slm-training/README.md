# Code-Aware SLM

A small language model trained from scratch on the *prose inside code*,
docstrings, comments and README text scraped from top-starred GitHub
repositories. ~757M parameters (24L/24H/1536D), GPT-2 architecture, trained with gradient checkpointing on CUDA.

No GPU, no cloud bill, no pretrained weights. The whole pipeline is six Python
files.

---

## Why comments and docstrings?

Most small language models train on general web text. This one trains on the
natural language that developers write *about* code:

- **It has structure.** A docstring is preceded by a signature; a JSDoc block is
  followed by a declaration. Those pairs are preserved as single training
  documents, so the model sees `def calculate_distance(x1, y1, x2, y2):`
  attached to the English that explains it.
- **It is unusually clean.** Comments in heavily-starred repositories are
  reviewed, edited prose. There is very little of the noise that dominates a
  web crawl.
- **It is a distinct distribution.** The vocabulary sits between English and
  code, identifiers, types, imperative descriptions of behaviour, which is
  exactly the register you want for docstring completion, comment generation
  and commit-message drafting.

## What the pipeline does

```
scrape_github.py  →  raw_data.txt      docstrings + comments + README prose
preprocess.py     →  train.bin/val.bin GPT-2 BPE tokens, 90/10 sequential split
train.py          →  best_model.pt     transformer training with resume + recovery
evaluate.py       →  loss_curves.png   convergence analysis
inference.py      →  samples           generation from prompts
make_report.py    →  TRAINING_REPORT.md
```

Each stage writes files the next stage reads, so any stage can be re-run alone.

See **[QUICKSTART.md](QUICKSTART.md)** to run it and **[API.md](API.md)** for
every flag.

---

## Architecture

| Property | Value |
| --- | --- |
| Parameters | 81,564,800 |
| Non-embedding parameters | 49,236,480 |
| Layers | 10 |
| Attention heads | 10 (64 dims each) |
| Embedding dimension | 640 |
| Context length | 256 tokens |
| Vocabulary | 50,257 (GPT-2 BPE via `tiktoken`) |
| Dropout | 0.1 |

Standard pre-norm GPT-2: learned positional embeddings, causal self-attention
through `scaled_dot_product_attention`, a 4× GELU MLP, and weight tying between
the token embedding and the output projection. Residual projections use the
GPT-2 scaled initialisation (`std = 0.02/sqrt(2·n_layer)`).

Weight tying matters at this scale. The token embedding alone is 32.2M
parameters, 39% of the model, so sharing it with the output projection is both
a large memory saving and a real regulariser.

---

## Status

**Full 10,000-step training run launched 19 Oct 2026 on CUDA** (NVIDIA RTX 5060
Laptop GPU, bf16, ~23k tokens/s, 0.4 s/step, ETA ≈ 60 minutes, 90× faster than
the CPU projection). Live progress: `slm-training/train_cuda_run.log`, earlier
checkpoints rotate at `checkpoint_step_*.pt`, best weights land in
`best_model.pt` (val-loss tracked), and a monitor-friendly log line appears
every 100 steps. Follow-up after completion: `python evaluate.py`, then
`python make_report.py` to regenerate TRAINING_REPORT.md from the run artifacts.

Setup used on this machine (Blackwell sm_120 needs cu128+ wheels):

    py -V:Astral/CPython3.12.14 -m venv .venv
    .venv\Scripts\pip install numpy tiktoken
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\python train.py            # auto-detects CUDA + bf16

Previous status (for the record): the corpus was collected and validated
23 Aug 2026, 120.7M tokens from 600 repositories, all ten validation checks
passed, tokenized to 119.1M train / 13.2M val tokens. The first CPU attempt was
stopped at step ~200 (loss 10.82 → 6.79, warmup tracking exactly); the recipe
was sound, only the hardware changed.

Training was started on CPU and deliberately stopped at step ~200. It was
projected to take **six days** at 52.5 s/step, and the run will instead be done
on a CUDA GPU where the same 10,000 steps take roughly **30 minutes**. The loss
was falling correctly when it was stopped, 10.82 to 6.79 over the first 100
steps, with the learning-rate warmup tracking exactly, so the recipe is sound;
only the hardware changed.

To run it, see the CUDA section of [QUICKSTART.md](QUICKSTART.md). Results, loss
curves and generated samples land in
**[TRAINING_REPORT.md](TRAINING_REPORT.md)**, regenerated from the run's JSON
artifacts with `python make_report.py`.

---

## Notes on training without a GPU

The run this repository documents used an Intel i7-1065G7 (4 cores) with no
usable GPU backend, an Iris Plus iGPU has no CUDA and no practical PyTorch
training path. A few things that mattered:

- **Benchmark before committing.** `python train.py --benchmark 20` reports
  seconds/step and projects the full run. On CPU the difference between a
  30M and an 80M parameter model is days of wall-clock time.
- **The output projection dominates.** At a 50,257 vocabulary and a 640
  embedding, the final matmul is roughly as expensive as all ten transformer
  blocks put together. Shrinking the vocabulary is the single biggest lever on
  CPU throughput.
- **Checkpoint aggressively and always resume.** A multi-day CPU run will be
  interrupted. Checkpoints carry optimizer state, step count and loss history,
  and `train.py` resumes from the newest one with no flag.
- **Don't run anything else.** Scraping while benchmarking made each step ~30%
  slower and produced a misleading projection.
- **Check the power cable before you measure anything.** On battery, Windows
  capped this CPU to 998 MHz, below its 1498 MHz base, and a step took
  102.8 s. On AC with a warm page cache the same step took 54.5 s. That single
  factor was worth more than every other optimisation combined: 11.9 days
  versus 6.3.
- **Thread count barely matters here.** 4, 6 and 8 threads span 4%
  (56.8 / 54.5 / 54.9 s per step). The workload is bound by memory bandwidth,
  not core count.

### The integrated GPU does not help

Worth documenting, because it looks promising until you measure it. The Iris
Plus is reachable from PyTorch through `torch-directml`, and it does run this
model's forward and backward passes correctly. It fails on the first optimizer
step:

```
RuntimeError: Not enough memory resources are available to complete this operation.
```

The GPU exposes about 1024 MB. In fp32 this model needs:

| Allocation | Size |
| --- | --- |
| Weights | 326 MB |
| Gradients | 326 MB |
| AdamW `exp_avg` + `exp_avg_sq` | 653 MB |
| **Total, before activations** | **1305 MB** |

Warmup survives because AdamW's state tensors are not allocated until the first
`opt.step()`. The 281 MB shortfall is a fixed cost; it does not shrink with
batch size, so there is nothing to tune.

Even with enough memory it would be the wrong choice. Synchronised fp32 matmul
measures **115.6 GFLOPS on the iGPU against 89.3 GFLOPS on the CPU**, only
1.29x, because both share the same ~58 GB/s LPDDR4x bus. The FLOPS advantage
never materialises.

One trap worth naming: DirectML dispatches asynchronously, so a naive timing
loop measures queue submission rather than execution. Unsynchronised, that same
matmul reported **933 GFLOPS**, about eight times the real number, and exactly
the figure that would talk you into a rewrite. Force a sync (pull a value back
with `.cpu()`) before timing anything.

`torch.compile` is also unavailable here: inductor needs MSVC's `cl.exe`.

---

## Known limitations

- **One epoch, one domain.** The corpus is a single pass over Python and
  JavaScript prose. The model has never seen other languages, and nothing about
  it generalises to general English.
- **256-token context.** Long docstrings are truncated at the context boundary
  during training.
- **No instruction tuning.** This is a base model. It completes text; it does
  not follow instructions.
- **Sequential split.** Validation is the tail of the corpus rather than a
  random sample, so the split is by position rather than by repository. Records
  from a repo that straddles the boundary can appear on both sides.
- **Extraction is heuristic for JavaScript.** Python uses `ast` and `tokenize`,
  which are exact. JavaScript uses regular expressions, which are not: a
  line-anchored block-comment pattern plus a code-shape rejector removes most
  false positives, but not all.

## Next steps

- Quantize to INT8 for faster CPU inference
- Fine-tune on a single repository or framework for domain-specific completion
- Train a smaller domain vocabulary to cut the output-projection cost
- Extend the context window beyond 256 tokens

---

## Repository layout

```
slm-training/
├── scrape_github.py       Phase 1  data collection
├── preprocess.py          Phase 2  tokenization + split
├── model.py                        GPT implementation
├── train.py               Phase 3  training loop
├── evaluate.py            Phase 4  convergence analysis
├── inference.py           Phase 4  generation
├── make_report.py                  report generation
├── slmlog.py                       shared logging
├── requirements.txt
├── QUICKSTART.md                   how to run it
├── API.md                          every script's options
├── TRAINING_REPORT.md              results
└── LICENSE                         MIT
```

Model weights are published as GitHub release assets rather than committed,
`best_model.pt` is ~330 MB.

## License

MIT
