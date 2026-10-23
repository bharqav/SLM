# API Reference

Every stage is a standalone CLI script. They communicate through files in the
project directory, so any stage can be re-run independently.

```
scrape_github.py  ->  raw_data.txt
preprocess.py     ->  train.bin, val.bin, tokenizer_meta.json
train.py          ->  best_model.pt, final_model.pt, checkpoint_step_*.pt, losses.json
evaluate.py       ->  evaluation_report.json, loss_curves.png
inference.py      ->  inference_samples.txt
```

All scripts append timestamped lines to `train.log` in addition to stdout.

---

## `scrape_github.py`

Collects comments, docstrings and README prose from top-starred GitHub repos.

| Option | Default | Meaning |
| --- | --- | --- |
| `--target-tokens` | `120000000` | Stop once the corpus reaches this estimated token count. |
| `--min-stars` | `1000` | Minimum stars for a repo to enter the queue. |
| `--max-repos-per-lang` | `600` | Cap on repos discovered per language (GitHub search returns at most 1000). |
| `--languages` | `python,javascript` | Comma-separated language filters. |
| `--fresh` | off | Delete `raw_data.txt` and `scrape_state.json` before starting. |

**Authentication.** Reads `GITHUB_TOKEN`/`GH_TOKEN`, else falls back to
`gh auth token`. The token is used only for the search API; archives are pulled
unauthenticated from `codeload.github.com`, because forwarding API credentials
to codeload trips its secondary rate limiter.

**Output.** `raw_data.txt` — UTF-8 records separated by a line containing
exactly `<|endoftext|>`. `scrape_state.json` tracks completed repos so an
interrupted run resumes without re-downloading or duplicating.
`scrape_stats.json` holds final counts by record kind.

```bash
python scrape_github.py --target-tokens 120000000 --min-stars 1000
```

---

## `preprocess.py`

Tokenizes the corpus and writes the train/val split.

| Option | Default | Meaning |
| --- | --- | --- |
| `--val-split` | `0.1` | Fraction of tokens held out for validation. |
| `--seed` | `1337` | Reproducibility seed. |

**Output.** `train.bin` and `val.bin` (flat `uint16` token streams),
`tokenizer_meta.json`, `preprocess_stats.json` (token counts, record-length
histogram, chars-per-token).

The split is *sequential*, not shuffled: validation is the tail of the corpus,
so records from one repo do not straddle both sides of the split.

```bash
python preprocess.py --val-split 0.1
```

---

## `model.py`

Importable module, not a CLI.

```python
from model import GPT, GPTConfig

cfg = GPTConfig(vocab_size=50257, block_size=256,
                n_layer=10, n_head=10, n_embd=640, dropout=0.1)
model = GPT(cfg)

model.num_params()                 # 81,564,800
model.num_params(non_embedding=True)
model.param_breakdown()            # dict of component -> parameter count

logits, loss = model(idx, targets) # targets=None returns last-position logits
out = model.generate(idx, max_new_tokens=50, temperature=0.7, top_k=50)
opt, n_decay, n_nodecay = model.configure_optimizers(0.1, 6e-4, (0.9, 0.95))
```

`GPTConfig` fields: `vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`,
`dropout`, `bias`. The token embedding and output projection share weights.

---

## `train.py`

| Option | Default | Meaning |
| --- | --- | --- |
| `--batch-size` | `4` | Sequences per micro-batch. |
| `--grad-accum-steps` | `8` | Micro-batches per optimizer step (effective batch 32). |
| `--block-size` | `256` | Context length. |
| `--n-layer` / `--n-head` / `--n-embd` | `10` / `10` / `640` | Architecture. |
| `--dropout` | `0.1` | Dropout probability. |
| `--learning-rate` | `6e-4` | Peak LR. |
| `--min-lr` | `1e-5` | Cosine floor. |
| `--warmup-steps` | `1000` | Linear warmup length. |
| `--max-steps` | `10000` | Total optimizer steps. |
| `--weight-decay` | `0.1` | Applied to dim >= 2 tensors only. |
| `--beta1` / `--beta2` | `0.9` / `0.95` | AdamW betas. |
| `--grad-clip` | `1.0` | Max gradient norm. |
| `--eval-interval` | `500` | Steps between validation passes. |
| `--eval-iters` | `40` | Batches averaged per validation pass. |
| `--checkpoint-interval` | `200` | Steps between full checkpoints. |
| `--log-interval` | `100` | Steps between progress lines. |
| `--patience-steps` | `5000` | Early stop if val loss stalls this long. |
| `--threads` | `0` | Torch threads; `0` means physical core count (CPU only). |
| `--device` | `auto` | `auto` picks CUDA when available, else CPU. |
| `--dtype` | `auto` | `auto` = bfloat16 on CUDA if supported, float16 on older CUDA, float32 on CPU. |
| `--compile` | off | Wrap in `torch.compile`. Needs a C++ compiler; unavailable on stock Windows. |
| `--resume` | off | Force resume (a checkpoint is auto-detected anyway). |
| `--benchmark N` | `0` | Time N optimizer steps, print throughput, exit. |

**Resuming.** `train.py` always resumes from the newest
`checkpoint_step_*.pt` when one exists. Checkpoints carry model weights,
optimizer state, step, best val loss and full loss history. If the newest file
is corrupt it falls back to the previous one automatically. Only the 3 most
recent checkpoints are kept.

**Recovery.** A non-finite loss reloads the best weights and scales the LR by
0.1 (aborting after 5 such events). An allocation failure halves the batch size
and doubles gradient accumulation, preserving the effective batch. `Ctrl-C`
checkpoints and exits cleanly.

```bash
python train.py --benchmark 20      # measure throughput first
python train.py                     # start or resume
```

**On a CUDA GPU.** The defaults are sized for a CPU, where a micro-batch of 4
sequences is all the memory bandwidth will carry. A discrete GPU is starved by
that -- it goes latency-bound rather than compute-bound -- so raise the batch
and drop the accumulation to keep the same effective batch:

```bash
python train.py --batch-size 32 --grad-accum-steps 1   # 8192 tokens/step
```

With 24 GB of VRAM there is room for far more. The model, gradients and AdamW
state together are only ~1.3 GB in fp32, so the remaining headroom is all
activations: `--batch-size 64 --grad-accum-steps 2` gives a 32768-token step and
still fits comfortably.

`--dtype auto` selects bfloat16 on Ampere and newer, which needs no loss scaler
and keeps fp32 range. TF32 matmuls are enabled automatically on CUDA.

---

## `evaluate.py`

No options. Reads `losses.json` + `training_stats.json`.

Checks the convergence criteria (final train loss < 3.0, val loss < 3.5, no
eval-to-eval spike above 0.5, train loss moving less than 0.1 over the last
1000 steps, and a generally decreasing curve), writes
`evaluation_report.json`, and renders `loss_curves.png`.

```bash
python evaluate.py
```

---

## `inference.py`

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | `best_model.pt` | Checkpoint to load. |
| `--prompt` | none | Single prompt; omit to run the 5-prompt test suite. |
| `--tokens` | `50` | Tokens to generate. |
| `--temperature` | `0.7` | Sampling temperature. |
| `--top-k` | `50` | Top-k truncation. |
| `--samples` | `1` | Samples per prompt. |
| `--seed` | `1337` | Sampling seed. |

With no `--prompt`, it runs the five PRD test prompts and writes
`inference_samples.txt`, reporting tokens/s and the repeated-4-gram ratio as a
degeneracy check.

```bash
python inference.py
python inference.py --prompt "def calculate_" --tokens 60 --temperature 0.8
```
