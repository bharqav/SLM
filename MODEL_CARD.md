# SLM Project Model Card

This repository ships two model families. They are different artifacts with different
training recipes, and every number below is traceable to a file in this repo.

| | From-scratch SLM | Qwen2.5-0.5B QLoRA fine-tune |
|---|---|---|
| Code | `slm-training/` | `slm-ft/` |
| Architecture | GPT-2 style decoder-only transformer | Qwen2.5-0.5B-Instruct (GQA, RoPE, RMSNorm, SwiGLU) |
| Parameters | 253.3M (16L/16H/1024D) or 757.5M (24L/24H/1536D), tied embeddings | 494.0M base, 540,672 trainable LoRA scalars per adapter |
| Context length | 256 tokens | 32,768 tokens |
| Tokenizer | tiktoken GPT-2 BPE, 50,257 | Qwen2.5 BPE, 151,936 |
| Training | from random initialization | 4-bit NF4 QLoRA, merged to fp16 |
| License | MIT (code) | MIT (code); base weights under Qwen's license |

## Intended use

An educational and portfolio demonstration of two ways to build a small language model for
supportive conversation: train one from scratch, or specialize a pretrained instruct model.

This is not a therapist, not a medical device and not a crisis service. It must not be used
as a substitute for professional help.

## Training data

- **From-scratch SLM:** a code-prose corpus (comments, docstrings and READMEs from permissively
  licensed public repositories; 2,539,694 records, about 132.3M tokens) and empathetic dialogue
  corpora, depending on the checkpoint. See `MODEL_COMPARISON.md` for the per-checkpoint inventory.
- **Qwen fine-tune:** 23,427 training and 461 validation dialogs built from ESConv (910
  dialogs) and EmpatheticDialogues (about 23k), about 2.03M tokens. Stage 2 adds 30 hand-written
  safety-steering dialogs (upsampled 9x) plus 500 replay dialogs.

EmpatheticDialogues is CC BY-NC-4.0 and ESConv is CC BY-NC-SA. Artifacts trained on them
(`slm-ft/merged_fp16/`, `slm-training/best_empathy_model_v2.pt`) inherit a non-commercial
restriction regardless of the MIT license on the code.

## Measured performance

CPU, float32, 8 threads, 5 prompts, 40 new tokens, seed 1337
(`benchmarks/generation_matrix.json`, `benchmarks/forward_matrix.json`):

| Model | Params | Generation | Forward pass (seq 64) | RSS after load |
|-------|--------|-----------|-----------------------|----------------|
| SLM 16L (`final_model.pt`) | 253.3M | 12.54 tok/s | 9.1 passes/s | 2,907 MB |
| SLM 24L (`best_empathy_model_v2.pt`) | 757.5M | 4.69 tok/s | 3.3 passes/s | 6,078 MB |
| Qwen2.5-0.5B-Instruct (base) | 494.0M | 9.92 tok/s | 3.7 passes/s | 3,631 MB |
| Qwen2.5-0.5B + QLoRA stage 2 (merged) | 494.0M | 10.62 tok/s | 3.5 passes/s | 3,923 MB |

Validation loss (not comparable across rows: different tokenizers, data and loss masking):

| Model | Best val loss | Notes |
|-------|---------------|-------|
| SLM 16L (`final_model.pt`) | 2.9197 | step 3900 of 4000, clean-empathy corpus |
| SLM 24L (`best_empathy_model_v2.pt`) | 3.9259 | step 1000, perplexity about 50.7 |
| Qwen QLoRA stage 1 | 2.7650 | assistant tokens only |
| Qwen QLoRA stage 2 | 2.7655 | 770 dialogs, 96 steps |

On the same prompts, neither from-scratch model matches Qwen's output quality. Qwen is coherent
and on topic, including valid code completions. The from-scratch models are short, repetitive
or off-topic on prompts outside their training distribution. The 24L model has no KV cache, so
it is slower than Qwen at generation.

## Limitations

- 256-token context for the from-scratch models; long generations degrade.
- No standard academic benchmark scores are recorded for the from-scratch models.
- The behavioral safety gate for the fine-tune is a 5-probe smoke test, not a coverage
  guarantee, and its record predates the latest stage-2 retrain. Re-run it before any release.
- Trained weights are large and gitignored; a fresh clone carries code, configs and tokenizers
  only. Regenerate weights with the steps in `slm-ft/README.md` and `slm-training/README.md`.
- Empathetic-dialogue corpora are mostly English and reflect their annotators' views of empathy.

## How to use the from-scratch model

```python
import sys
import torch
import tiktoken

sys.path.insert(0, "slm-training")
from model import GPT, GPTConfig

ckpt = torch.load(
    "slm-training/best_empathy_model_v2.pt", map_location="cpu", weights_only=False
)
model = GPT(GPTConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()

enc = tiktoken.get_encoding("gpt2")
x = torch.tensor([enc.encode("I have been feeling really down lately")])
out = model.generate(x, max_new_tokens=50, temperature=0.7, top_k=20)
print(enc.decode(out[0].tolist()))
```

## Comparison

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the full evidence layer.

## Citation

```bibtex
@misc{slm2026,
  title={From-Scratch SLM vs a QLoRA-Tuned Qwen2.5-0.5B for Empathetic Dialogue},
  year={2026},
  url={https://github.com/bharqav/emotional-slm}
}
```
