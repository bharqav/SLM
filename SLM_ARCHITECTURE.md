# SLM Architecture Diagram

## From-Scratch SLM (slm-training/)

```
┌──────────────────────────────────────────────────────────────┐
│                    GPT-style Model                     │
├──────────────────────────────────────────────────────────────┤
│  Input: Token IDs (vocab_size=50257)                      │
│  └─────────────────────┬───────────────────────┘          │
│                       │                       │          │
│               Token Embedding               Position Embedding
│       (50257 × 768)            (256 × 768)           │
│                       │                       │          │
│                       └───────►  Dropout ──────┘          │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Transformer Blocks (12 layers)                        │
│  │  Each block:                                    │
│  │  ├─ LayerNorm (pre-norm)                              │
│  │  ├─ Causal Self-Attention (12 heads, head_dim=64)  │
│  │  │  └─ QKV projections, scaled dot-product attention │
│  │  ├─ LayerNorm (post-attention)                            │
│  │  └─ MLP (4× expansion, GELU activation)               │
│  │       ├─ Linear(768, 3072)                   │
│  │       └─ Linear(3072, 768)                   │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  └─ Final LayerNorm                                                       │
│                                                             │
│  └─ Language Modeling Head (tie weights with token embedding)│
│       Linear(768, 50257)                                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      Output: Logits over vocabulary

## Key Hyperparameters
- **vocab_size**: 50257 (tiktoken GPT-2 tokenizer)
- **n_layers**: 12
- **n_heads**: 12
- **head_dim**: 64
- **d_model**: 768
- **ctx_len**: 256
- **mlp_ratio**: 4.0
- **total parameters**: ~124M

## Training Data
- Source: Public code prose dataset + empathetic dialogues
- Volume: ~121M tokens (code) + ~2.4M tokens (empathy)
- Training tokens: ~107M (code) + ~2.1M (empathy)
- Validation tokens: ~13M (code) + ~49K (empathy)
- Data format: Clean text files, source code, documentation, dialogue
- License: Various open-source licenses + CC BY-NC-4.0 (empathetic data)

## Empathetic Fine-Tune
- Source: ESConv + EmpatheticDialogues
- Volume: ~2.4M tokens
- Training dialogs: 23,427
- Validation dialogs: 461
- Checkpoint: `best_empathy_model.pt` (after training)

## Inference
- Autoregressive generation
- Temperature sampling with top-k filtering
- Supports context window up to 256 tokens