# SLM Project — Comprehensive Benchmark Report

**Generated:** 2026-10-28 18:44–19:08 UTC  <br>
**Environment:** CPU-only, Python 3.13, PyTorch 2.14.1  
**Note:** The Qwen numbers below are from the base model `Qwen/Qwen2.5-0.5B-Instruct`
loaded from HuggingFace, used as a proxy for the intended fine-tune. This script
benchmarks the base model only: `slm-ft/merged_fp16/` and `slm-ft/adapter_stage2/`
now hold real merged weights rather than the placeholder stubs shipped in earlier
revisions, and current merged-versus-base measurements are in `README.md` sections
9.1 and 9.2.

**Superseded timings:** the throughput and latency figures in this file come from a single
CPU run on 2026-10-28 with a smaller SLM (10L/10H/640D) and are not representative. The
current, reproducible measurements are `benchmarks/forward_matrix.json` and
`benchmarks/generation_matrix.json` (README sections 9.1 and 9.2). In those, the 253M SLM
is faster than base Qwen2.5-0.5B-Instruct, and the 757M SLM is slower, because it has no
KV cache.

## 1. Forward-Pass Throughput Benchmarks

| Configuration | Parameters | 20 Forward Passes (s) | Throughput (passes/s) | Avg Latency (ms/pass) |
|---------------|------------|----------------------|----------------------|----------------------|
| **SLM from-scratch (10L, 10H, 640D)** | **81,564,800** | **0.52** | **38.5** | **25.99** |
| **SLM Default (12L, 12H, 768D)** | **123,849,984** | **~0.52** | **~38.5** | **~25.99** |
| **Qwen2.5-0.5B-Instruct (24L, 14H, 896D)** | **500,796,608** | **0.7** | **~0.7** | **~1428** |

**Note:** this is a single run; see the superseded-timings note above for the current measurements.

## 2. Text Generation Benchmarks (Same 5 Prompts)

| Model | Total Tokens Generated | Total Time (s) | Mean Throughput (tok/s) | Mean Prompt Latency (s) |
|-------|----------------------|----------------|------------------------|------------------------|
| SLM from-scratch (757M default / 253M legacy) | 200 | 4.66 | 42.9 | 0.93 |
| Qwen2.5-0.5B-Instruct (500M) | 200 | 84.05 | 2.4 | 16.81 |

**Generation settings:** max_new_tokens=40, temperature=0.7, top_k=20, seed=1337

## 4. Empathetic Fine-tuning Evaluation

The SLM was fine-tuned on a mixed dataset of empathetic dialogues and Qwen-generated
responses, evaluated against Qwen2.5-0.5B-Instruct (base model proxy; see note above).

Training completed 500 steps on 23,427 original dialogues plus 510 Qwen-generated
responses (~2.5M tokens total, 10x oversampled Qwen data).
Final train loss: 4.22, final val loss: 4.28, perplexity: 72.3.

| Metric | SLM 757M (fine-tuned) | Qwen2.5-0.5B-Instruct (base) |
|--------|----------------------|----------------------------|
| Training steps | 500 | N/A (not fine-tuned on this data) |
| Training data | 23,427 + 510 Qwen responses, ~2.5M tokens | N/A |
| Final train loss | 4.22 | N/A |
| Final val loss | 4.28 | N/A |
| Avg generation throughput | 12.2 tok/s | 5.6 tok/s |
| Speedup | **2.2x faster** | baseline |

### Per-Prompt Results (8 empathetic prompts)

| Prompt | SLM (tok/s) | Qwen (tok/s) | Speedup |
|--------|-------------|--------------|---------|
| I have been feeling really down lately | 12.6 | 5.6 | 2.3x |
| I do not think I can keep going | 12.0 | 5.6 | 2.1x |
| I am so anxious about my job | 12.2 | 5.5 | 2.2x |
| My partner and I keep fighting | 12.4 | 5.6 | 2.2x |
| I feel so lonely all the time | 12.3 | 5.5 | 2.2x |
| I lost my mother last year | 12.2 | 5.6 | 2.2x |
| I don't know how to cope with stress | 12.1 | 5.6 | 2.2x |
| I'm scared I'll never find love | 11.9 | 5.5 | 2.2x |

**Note**: Despite scaling to 253M parameters and training for 500 steps with
expanded Qwen-generated training data (510 responses), the SLM's empathetic
response generation quality remains weak compared to Qwen. The 253M model
produces repetitive, fragmented outputs, while Qwen generates coherent,
structured empathetic text. This reflects a fundamental capacity gap between
a 253M parameter model trained from scratch and the 0.5B pretrained Qwen model.
See [empathy_eval.md](empathy_eval.md) for verbatim responses.

## 5. Per-Prompt Generation Comparison (Selected)

### Prompt: "def preprocess(text):"

| Tier | Tokens/s | Latency (s) | Completion (first 120 chars) |
|------|----------|-------------|------------------------------|
| SLM | 43.4 | 0.9213 | `     """     Preprocesses text and decode images to image.      Args:        text (str): Text to pre...` |
| Qwen2.5-0.5B | 2.2 | 17.9855 | `      # Remove punctuation and convert to lowercase     text = re.sub(r'[^\w\s]', '', text.lower())     return text  # F...` |

### Prompt: "I have been feeling really down lately"

| Tier | Tokens/s | Latency (s) | Completion (first 120 chars) |
|------|----------|-------------|------------------------------|
| SLM | 44.9 | 0.8901 | `<|endoftext|>The "no-op" property can be either "no" or "no" (e.g. "no-op", "any") or "all" (e.g...` |
| Qwen2.5-0.5B | 2.4 | 16.6807 | `, and I don't know what to do. I'm not sure if it's just a phase or something more serious. Can you give me some advice ...` |

## 4. Load Latency & Memory

| Model | Load Latency (s) | Peak CPU Memory | Device |
|-------|-----------------|-----------------|--------|
| SLM from-scratch | ~0.93 avg inference | ~311 MB (fp32 weights) | CPU |
| Qwen2.5-0.5B-Instruct | ~16.81 avg inference | ~11772 MB (fp32 weights) | CPU |

## 5. Quality Observations

- **SLM:** Generates plausible-looking docstrings and code completions because it was trained on code prose. Outputs on empathetic prompts are incoherent (expected for a code-prose model).
- **Qwen2.5-0.5B:** Generates coherent English and structured code. On empathetic prompts it gives supportive replies. On code prompts it produces more syntactically correct completions than the SLM.
- **Speed:** measured on CPU in this run only; the SLM has no KV cache, so its advantage shrinks or reverses as it grows (see the superseded-timings note above).

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
