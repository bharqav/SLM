# SLM Architecture

Two models live in this repo. Both are decoder-only transformers; they differ in how they were
built. Source of truth for the from-scratch model is `slm-training/model.py`.

## From-scratch SLM (`slm-training/`)

```
token ids (B, T), T <= 256
        |
  wte: Embedding(50257, d)  +  wpe: Embedding(256, d)     learned absolute positions
        |
     Dropout(0.1)
        |
  +-----------------------------------------------------+
  |  Block x n_layer                                    |
  |    x = x + Attention(LayerNorm(x))      pre-norm    |
  |    x = x + MLP(LayerNorm(x))            pre-norm    |
  |                                                     |
  |  Attention: fused QKV Linear(d, 3d), n_head heads,  |
  |             causal scaled dot-product, Linear(d, d) |
  |  MLP:       Linear(d, 4d) -> GELU(tanh) -> Linear(4d, d) |
  +-----------------------------------------------------+
        |
  final LayerNorm
        |
  lm_head: Linear(d, 50257, bias=False)   weight tied to wte
        |
logits over the vocabulary
```

| Setting | 24L default | 16L checkpoint |
|---------|-------------|----------------|
| `n_layer` / `n_head` / `n_embd` | 24 / 24 / 1536 | 16 / 16 / 1024 |
| Head dimension | 64 | 64 |
| MLP hidden width | 6144 | 4096 |
| Parameters (tied, deduplicated) | 757,547,520 | 253,266,944 |
| Vocabulary | 50,257 (tiktoken GPT-2) | 50,257 |
| Context length | 256 | 256 |
| Dropout | 0.1 | 0.1 |

Implementation notes:

- Weight tying between `wte` and `lm_head` saves about 77M parameters at the 24L width.
- Residual projections (`c_proj`) use the GPT-2 scaled init, `0.02 / sqrt(2 * n_layer)`.
- Weight decay applies only to tensors with two or more dimensions (matmuls and embeddings).
- Inference returns logits for the last position only, and there is no KV cache: every
  generated token re-encodes the whole context. This is why the 24L model is slower than Qwen at
  generation (see `README.md` section 9.2).
- Sampling supports temperature, top-k, top-p, repetition penalty and stop strings.

## Qwen2.5-0.5B fine-tune (`slm-ft/`)

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Layers / hidden / MLP | 24 / 896 / 4864 |
| Attention | 14 query heads, 2 KV heads (grouped-query), head dim 64 |
| Normalization / activation | RMSNorm / SwiGLU |
| Positions | RoPE, context 32,768 |
| Vocabulary | 151,936, tied input and output embeddings |
| Adapter | LoRA r=8, alpha=8, dropout 0.05 on `q_proj` and `v_proj` (540,672 trainable scalars) |
| Quantization during training | 4-bit NF4, bf16 compute |
| Deployed artifact | adapter merged into fp16 weights (`slm-ft/merged_fp16/`) |

## Training data

- From-scratch: code-prose corpus (2,539,694 records, about 132.3M tokens) and empathetic dialogue
  corpora, depending on the checkpoint. Per-checkpoint details are in `MODEL_COMPARISON.md`.
- Fine-tune: 23,427 train and 461 validation dialogs (ESConv and EmpatheticDialogues), about
  2.03M tokens, then a 770-dialog stage-2 safety-steering mix.
- EmpatheticDialogues is CC BY-NC-4.0 and ESConv is CC BY-NC-SA, so artifacts trained on them
  are non-commercial.
