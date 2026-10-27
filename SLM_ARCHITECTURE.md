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
│       (50257 × 1024)            (1024 × 1024)           │
│                       │                       │          │
│                       └───────►  Dropout ──────┘          │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Transformer Blocks (8 layers)                        │     │
│  │  Each block:                                    │     │
│  │  ├─ RMSNorm (pre-norm)                              │     │
│  │  ├─ Causal Self-Attention (4 heads, head_dim=256)  │     │
│  │  │  └─ QKV projections, scaled dot-product attention │     │
│  │  ├─ RMSNorm (post-attention)                            │     │
│  │  └─ MLP (4× expansion, GELU activation)               │     │
│  │       ├─ Linear(d_model, d_model×4)                   │     │
│  │       └─ Linear(d_model×4, d_model)                   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  └─ Final RMSNorm                                                       │
│                                                             │
│  └─ Language Modeling Head (tie weights with token embedding)│
│       Linear(1024, 50257)                                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      Output: Logits over vocabulary

## Key Hyperparameters
- **vocab_size**: 50257 (tiktoken GPT-2 tokenizer)
- **d_model**: 1024
- **n_heads**: 4
- **head_dim**: 256
- **n_layers**: 8
- **ctx_len**: 1024
- **mlp_ratio**: 4.0
- **total parameters**: ~81.6M

## Training Data
- Source: Public code prose dataset
- Volume: ~13.2M tokens (estimated)
- Training tokens: ~12.5M
- Validation tokens: ~700K
- Data format: Clean text files, source code, documentation
- License: Various open-source licenses

## Inference
- Autoregressive generation
- Temperature sampling with top-k filtering
- Supports context window up to 1024 tokens