# SLM Project — Comprehensive Benchmark Report

**Generated:** 2026-10-25  
**Environment:** CPU-only (Intel/AMD x64), Python 3.13, PyTorch 2.14.1  
**Note:** The Qwen merged-fp16 artifact in `slm-ft/merged_fp16/` is a placeholder stub.  
The Qwen numbers below are from the base model `Qwen/Qwen2.5-0.5B-Instruct` loaded from HuggingFace,  
used as the closest available proxy for the intended fine-tune.

## 1. Forward-Pass Throughput Benchmarks

| Configuration | Parameters | 20 Forward Passes (s) | Throughput (passes/s) | Avg Latency (ms/pass) |
|---------------|------------|----------------------|----------------------|----------------------|
| Tiny (4L, 4H, 512D) | 38.5M | 0.302 | 66.3 | 15.08 |
| Small (6L, 6H, 768D) | 81.3M | 0.651 | 30.7 | 32.57 |
| **SLM Actual (10L, 10H, 640D)** | **81.6M** | **0.772** | **25.9** | **38.62** |
| Medium (8L, 4H, 1024D) | 152.5M | 1.324 | 15.1 | 66.22 |
| Large (12L, 6H, 1536D) | 417.6M | 3.859 | 5.2 | 192.95 |
| **Qwen2.5-0.5B-Instruct** | **494.0M** | **6.468** | **3.1** | **323.39** |

**Key insight:** The 81.6M SLM is ~8x faster per forward pass than the 494M Qwen base on CPU,  
because CPU inference is memory-bandwidth bound and the SLM has a smaller activation footprint.

## 2. Text Generation Benchmarks (Same 9 Prompts)

| Model | Total Tokens Generated | Total Time (s) | Mean Throughput (tok/s) | Mean Prompt Latency (s) |
|-------|----------------------|----------------|------------------------|------------------------|
| SLM from-scratch (81.6M) | 720 | ~19.4 | ~37.1 | ~2.15 |
| Qwen2.5-0.5B-Instruct (494M) | 720 | ~73.0 | ~9.9 | ~8.11 |

**Generation settings:** max_new_tokens=80, temperature=0.7, top_k=20

## 3. Per-Prompt Generation Comparison (Selected)

### Prompt: "def preprocess(text):"

| Tier | Tokens/s | Latency (s) | Completion (first 120 chars) |
|------|----------|-------------|------------------------------|
| SLM | 37.4 | 2.14 | `"""\n    Preprocesses text into a text file.\n\n    Args:\n        text (str): The text to preprocess.\n\n    Returns:\n  ...` |
| Qwen2.5-0.5B | 10.0 | 8.00 | `"""\n    This function takes a string as input and returns the preprocessed text. It performs the following operat...` |

### Prompt: "I have been feeling really down lately"

| Tier | Tokens/s | Latency (s) | Completion (first 120 chars) |
|------|----------|-------------|------------------------------|
| SLM | 38.6 | 2.07 | `, and it is a different case, so it is the same as the original test.<|endoftext|>the \`test\` function is the result of e...` |
| Qwen2.5-0.5B | 9.2 | 8.74 | ` and I'm not sure if it's because of the pandemic or just because I've been having a tough time. Can you give me some ti...` |

## 4. Load Latency & Memory

| Model | Load Latency (s) | Peak CPU Memory | Device |
|-------|-----------------|-----------------|--------|
| SLM from-scratch | ~0.02 | ~326 MB (fp32 weights) | CPU |
| Qwen2.5-0.5B-Instruct | ~0.05 | ~1.9 GB (fp32 weights) | CPU |

## 5. Quality Observations

- **SLM:** Generates plausible-looking docstrings and code completions because it was trained on code prose. Outputs on empathetic prompts are incoherent (expected for a code-prose model).
- **Qwen2.5-0.5B:** Generates coherent English and structured code. On empathetic prompts it gives generic supportive replies. On code prompts it produces more syntactically correct completions than the SLM.
- **Speed:** SLM is faster on CPU. On GPU, Qwen would likely be faster in absolute tok/s due to better optimization, but the SLM's small size still gives it low latency.

## 6. Reproducibility

All benchmarks are reproducible with:

```bash
cd /c/SubD/Transfer/Projects/SLM
python slm-comparison/generate_live_comparison.py
```

Outputs:
- `slm-comparison/comparison_samples.json`
- `slm-comparison/comparison_samples.md`
- `benchmarks/slm_live_results.json`
- `benchmarks/qwen_base_results.json`
- `benchmarks/slm_forward_bench.json`
- `benchmarks/qwen_forward_bench.json`

Random seeds are fixed where applicable.
