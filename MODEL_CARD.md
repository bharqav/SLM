# SLM Project Model Card

## Model Details

**Model Name:** SLM from-scratch (code-prose variant + empathetic fine-tune)
**Version:** 2.0.0
**Architecture:** GPT-style decoder-only transformer
**Parameters:** ~757M (24L/24H/1536D default; legacy 253M 16L/16H/1024D checkpoint available)
**License:** MIT

### Model Description

This is a from-scratch GPT-style language model trained on a corpus of code prose extracted from permissively licensed public GitHub repositories. The model is **not** instruction-tuned and is intended as a teaching/portfolio artifact demonstrating end-to-end training.

### Training Data

- **Source:** Public GitHub repositories (MIT, Apache 2.0, BSD licenses)
- **Content:** Code comments, docstrings, README text, technical documentation
- **Size:** ~2.54M records, ~2.1M train tokens / ~44K val tokens (empathy corpus); 256-token context window

### Intended Use

**Primary:** Educational demonstration of training a language model from scratch.
**Secondary:** Code prose completion, docstring generation, technical documentation assistance.

### Limitations

- Trained on code prose only; produces incoherent output on general dialogue/empathetic prompts
- 256-token context window; long-context generation is unreliable
- GPU-optimized; benchmarked on CUDA (RTX 5060, bf16, ~970 tokens/s for 757M model)
- Instruction-tuned variant available (`best_empathy_model_v2.pt`) for empathetic conversation

### Performance

| Metric | Value |
|--------|-------|
| Parameters | ~757M |
| Context Length | 256 tokens |
| Validation Loss | 3.39 (757M empathy model, step 1175) |
| Throughput (CPU) | ~37 tok/s |
| Load Latency (CPU) | ~2.6s |

### How to Use

```python
from model import GPT, GPTConfig
import torch

cfg = GPTConfig()
model = GPT(cfg)
state = torch.load("best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
model.eval()

from tiktoken import get_encoding

enc = get_encoding("gpt2")
tokens = enc.encode("def preprocess(text):")
x = torch.tensor([tokens], dtype=torch.long)
out = model.generate(x, max_new_tokens=50, temperature=0.7, top_k=20)
print(enc.decode(out[0].tolist()))
```

### Empathetic Variant

An additional checkpoint (`best_empathy_model.pt`) is available after training on
23,427 empathetic dialogues (ESConv + EmpatheticDialogues, ~2.4M tokens). This
variant is optimized for supportive conversation and outperforms the Qwen2.5-0.5B
base model on empathetic response quality, while remaining ~50x faster on CPU.

### Citation

If you use this model in research, please cite:

```bibtex
@misc{slm2026,
  title={From-Scratch SLM: A Code-Prose Language Model},
  year={2026},
  url={https://github.com/bharqav/emotional-slm}
}
```

## Model Comparison Summary

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for detailed benchmark tables comparing this model against Qwen2.5-0.5B-Instruct.
