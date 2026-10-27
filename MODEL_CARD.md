# SLM Project Model Card

## Model Details

**Model Name:** SLM from-scratch (code-prose variant)
**Version:** 1.0.0
**Architecture:** GPT-style decoder-only transformer
**Parameters:** 81.6M
**License:** MIT

### Model Description

This is a from-scratch GPT-style language model trained on a corpus of code prose extracted from permissively licensed public GitHub repositories. The model is **not** instruction-tuned and is intended as a teaching/portfolio artifact demonstrating end-to-end training.

### Training Data

- **Source:** Public GitHub repositories (MIT, Apache 2.0, BSD licenses)
- **Content:** Code comments, docstrings, README text, technical documentation
- **Size:** ~2.54M records, ~132.3M tokens
- **Split:** 90% train / 10% validation (sequential split)

### Intended Use

**Primary:** Educational demonstration of training a language model from scratch.
**Secondary:** Code prose completion, docstring generation, technical documentation assistance.

### Limitations

- Trained on code prose only; produces incoherent output on general dialogue/empathetic prompts
- 256-token context window; long-context generation is unreliable
- CPU-optimized; not benchmarked on GPU
- Not instruction-tuned; does not follow prompts like a chat assistant

### Performance

| Metric | Value |
|--------|-------|
| Parameters | 81.6M |
| Context Length | 256 tokens |
| Validation Loss | 2.9816 |
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

### Citation

If you use this model in research, please cite:

```bibtex
@misc{slm2026,
  title={From-Scratch SLM: A Code-Prose Language Model},
  year={2026},
  url={https://github.com/your-repo/slm-comparison}
}
```

## Model Comparison Summary

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for detailed benchmark tables comparing this model against Qwen2.5-0.5B-Instruct.
