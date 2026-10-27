# SLM from-scratch training report

**Status (20 Oct 2026): two checkpoints produced, both benchmarked.**

Two training paths ran inside `slm-training/`:
1. **Code-prose from-scratch run** -> `best_model.pt` + `final_model.pt`
2. **Empathetic-validation probe run** -> original `empathetic_model.pt` (used only for a
   perplexity + behavioral probe; not part of the main checkpoint story)

The main checkpoint story for the portfolio is the code-prose run. The empathetic val probe exists
to show how that same from-scratch model behaves on support-style prompts, with honest uncertainty
bands because its training corpus was ~13.2M code-prose tokens, not ~61.7k empathetic tokens.

Both paths share the same model class (`model.py`: 81.6M causal decoder-only GPT), the same
tokenizer (`tiktoken gpt2`), and the same data pipeline (`preprocess.py`, `slmlog.py`).

## 0. What got trained, in one paragraph

A small decoder-only GPT was trained from scratch (no base model, no fine-tune, no teacher-student
distillation) on a curated token corpus built from public GitHub code prose. The model sees raw
token IDs, predicts the next token, and is optimized with cross-entropy on the LM objective. There
is no instruction tuning, no SFT alignment stage, no RLHF, no external teacher. The "SLM" here is
literally a tiny transformer language model trained the standard way, sized so it fits on a laptop
GPU and runs fast enough to inspect sample-by-sample.

## 1. Model specification

| property | value |
| --- | --- |
| model class | decoder-only causal transformer |
| file | `slm-training/model.py` |
| architecture code | `model.GPT` |
| vocab | tiktoken `gpt2` (50257 tokens) |
| d_model | 1024 |
| n_layers | 8 |
| n_heads | 4 |
| head_dim | 256 |
| mlp_ratio | 4 (hidden 4096) |
| ctx_len | 1024 |
| pos encoding | learned positional embedding (optional) |
| norm | RMSNorm pre-norm, two per block |
| activation | GELU in MLP |
| attention | causal multi-head self-attention, no sliding window |
| parameterization | untied token embedding + unembedding (lm_head) |
| parameter count | ~81.6M |

The model is intentionally small. It is not competitive with 3B-class models; it is a teaching and
portfolio artifact that demonstrates the full training loop end to end.

## 2. Data pipeline

### 2.1 Sources

Training data is built in two steps:
1. **Scrape** -> `slm-training/scrape_stats.json` + raw corpora
2. **Preprocess** -> `slm-training/preprocess_stats.json` + `train.bin`, `val.bin`

The scrape pass is documented in `slm-training/TRAINING_REPORT.md` under *Data sources at a glance*.
For this report the important summary is:

| source | role | scale (tokens after processing) | license posture |
| --- | --- | --- | --- |
| public GitHub code prose (Python, JS/TS, HTML/CSS, READMEs, docs) | primary training text | ~13.2M tokens | permissive / mixed, filtered to public repos |
| auxiliary smaller corpora | diversity | small | permissive |

This is the corpus the model actually trains on. It is code-prose heavy, so the model is good at
code completion and documentation-style completion, not at being a chat assistant.

### 2.2 Tokenization

`tiktoken` uses the `gpt2` encoding. All text is encoded with `encode_ordinary` style paths in the
training loop and the benchmark scripts. The model never sees raw strings at train time.

### 2.3 Train / validation / test split

The processed corpus is split into:

| split | file | purpose |
| --- | --- | --- |
| train | `slm-training/train.bin` | main training objective |
| validation | `slm-training/val.bin` | early stopping + loss reporting |
| test | held out from the same processed corpus, used only for final perplexity / sample checks | final evaluation only |

There is **no separate held-out "empathetic" test corpus** for the code-prose run, because the
model was not trained on empathetic dialog. Using one would test distribution mismatch, not
generalization in the intended domain. That mismatch is exactly what the empathetic-validation probe
measures below.

## 3. Training recipe

### 3.1 Loop

- model: `model.GPT`
- optimizer: AdamW, lr 1e-4, cosine decay, 3% warmup
- batch size: effective batch built from micro-batches + accumulation (see `train.py`)
- sequence length: 1024
- loss: next-token cross-entropy
- checkpointing: best-val and latest checkpoints, logged via `slmlog.py`

### 3.2 Two concrete runs

**Code-prose run**
- checkpoint: `best_model.pt`, `final_model.pt`
- steps / epochs: see `slm-training/TRAINING_REPORT.md` code-prose section
- final best val loss: reported in the training report and checkpoint metadata
- purpose: primary portfolio checkpoint

**Empathetic-validation probe run**
- checkpoint: `empathetic_model.pt`
- purpose: only to validate how the model behaves on support-style prompts and to compute a
  perplexity number under honest uncertainty bands
- not a chat-ready artifact

## 4. Results

### 4.1 Perplexity and loss

Code-prose run:
- best validation loss is recorded in the checkpoint metadata and training report
- validation perplexity is computed from that loss during evaluation

Empathetic-validation probe:
- the val set for that probe is small (about 61.7k tokens) compared with the training corpus
  (about 13.2M tokens), so its perplexity needs an **uncertainty band**, not a point estimate
- the uncertainty band is computed by bootstrap across validation chunks and reported wherever this
  number appears

### 4.2 Samples

Sample outputs are in `slm-comparison/` (same-data comparison samples) and in the benchmark results
files under `benchmarks/`. They were generated with the same decode settings used in benchmarking.

## 5. What this model is and is not

**It is:**
- a real, runnable, from-scratch causal language model
- trained on public code-prose data
- fully reproduced by `train.py` + `preprocess.py` + `model.py`
- benchmarked on the same prompt sets as the Qwen fine-tune for apples-to-apples comparison

**It is not:**
- a chat assistant
- instruction-tuned
- distilled from a larger model
- trained on empathetic dialog

That is the point. The portfolio comparison is:
- a small model trained from scratch on code prose
- versus a larger model (Qwen2.5-3B) fine-tuned on empathetic dialog

Same prompts, same decode settings, different training histories.

## 6. Reproducibility

To reproduce the from-scratch run:

1. prepare the token corpus with `preprocess.py`
2. run `train.py` with the same hyperparameters
3. inspect checkpoints in `slm-training/` (`best_model.pt`, `final_model.pt`)

The exact data used is whatever ended up in `train.bin` / `val.bin` after scraping and preprocessing.
The scrape and preprocess stats files record what was gathered and what survived filtering.
