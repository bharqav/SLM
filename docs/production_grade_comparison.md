# Production-Grade SLM Comparison

**Generated:** 2026-11-01  
**Subject:** 757M from-scratch SLM (24L/24H/1536D) vs. production-grade open-weight models  
**Environment context:** CPU-only, Python 3.13, Windows 11, no CUDA. Benchmarks in this repo are CPU-only.

> **Honesty notice:** This document does not claim that the 757M SLM matches the quality of models 5–20x larger. It is a speed/quality tradeoff analysis. Where the 757M SLM has no published number, the cell is marked **N/R** (not recorded) rather than estimated.

---

## 1. Model Specs at a Glance

| Model | Parameters | Context | Vocab | Attention | Training Paradigm | License |
|-------|-----------|---------|-------|-----------|-------------------|---------|
| **This project (SLM)** | ~757M (24L/24H/1536D) | 256 | 50,257 (GPT-2 BPE) | Multi-head (12 heads) | From scratch, mixed code+prose + empathy SFT | Mixed (code: open; empathy: CC BY-NC-4.0) |
| **Phi-4-mini** | 3.8B | 128K | ~200K | GQA | From scratch, synthetic + filtered web data | MIT |
| **Qwen2.5-7B** | 7B | 128K | 151,936 | GQA (2 KV) | From scratch, multilingual web+code | Apache 2.0 |
| **Llama-3.2-1B** | 1.23B | 128K | 128K | GQA | From scratch, multilingual web | Llama 3.2 Community |
| **Llama-3.2-3B** | 3.21B | 128K | 128K | GQA | From scratch, multilingual web | Llama 3.2 Community |
| **Ministral-3B** | 3B | 256K | 131K | GQA | From scratch, multilingual web | MNML |
| **Ministral-8B** | 8.4B | 256K | 131K | GQA | From scratch, multilingual web | MNML |
| **Gemma-2-2B** | 2B | 8K | 256K | GQA | From scratch, web+code | Gemma Terms |
| **Gemma-2-9B** | 9.2B | 8K | 256K | GQA | From scratch, web+code | Gemma Terms |
| **SmolLM2-1.7B** | 1.71B | 8K | 49K | GQA | From scratch, curated web+code | Apache 2.0 |

### Notes on the 757M SLM
- The saved state dict contains ~113.7M scalar tensors because the token embedding (`transformer.wte.weight`) and LM head (`lm_head.weight`) share storage via weight tying. The full architecture is ~757M parameters when the embedding table is counted once.
- Context length is hard-coded to 256 tokens in the current architecture and tokenizer pipeline. RoPE/position embeddings are not scaled beyond this.
- The tokenizer is a GPT-2 BPE (`tiktoken`), which has no built-in chat template. All role-tagging (`User:` / `Assistant:`) is application-layer; the model itself is a flat text continuation model.

---

## 2. Benchmark Comparison

### 2.1 Standard Academic Benchmarks

| Benchmark | This SLM (757M) | Phi-4-mini 3.8B | Qwen2.5-7B | Llama-3.2-1B | Llama-3.2-3B | Ministral-3B | Ministral-8B | Gemma-2-2B | Gemma-2-9B | SmolLM2-1.7B |
|-----------|----------------|-----------------|------------|--------------|--------------|--------------|--------------|------------|------------|--------------|
| **MMLU** (5-shot) | N/R | 67.3 | 74.2 | 32.2 | 58.0 | 60.9 | 65.0 | 52.4 | 71.3 | N/R* |
| **HellaSwag** (10-shot) | N/R | 69.1 | 80.2 | 41.2 | 69.8 | ~70 | ~80 | 73.0 | 81.9 | 68.7 |
| **WinoGrande** | N/R | ~72 | 75.9 | 60.8 | 61.4 | 72.7 | 75.3 | 70.9 | 80.6 | 59.4 |
| **HumanEval** (pass@1) | N/R | 74.4 | 57.9 | 29.3 | 33.5 | 34.2 | 34.8 | 17.7 | 40.2 | 22.6 |
| **MATH** (0-shot CoT) | N/R | 64.0 | 49.8 | 30.6 | 47.3 | ~40 | ~50 | 15.0 | 36.6 | 11.6 |
| **GSM8K** (8-shot CoT) | N/R | 88.6 | 85.4 | 44.4 | 77.7 | 50.9 | 64.5 | 23.9 | 68.6 | 31.0 |
| **ARC-Challenge** | N/R | 83.7 | 90.1 | 59.4 | 78.6 | 64.2 | 71.9 | 55.4 | 68.4 | 60.5 |

\* SmolLM2 reports MMLU-Pro (19.4) rather than MMLU in the base model card.

### 2.2 Project-Measured Metrics (CPU-only)

| Metric | This SLM (757M) | Qwen2.5-0.5B (proxy) |
|--------|-----------------|----------------------|
| Forward-pass throughput | 38.5 passes/s | 0.7 passes/s |
| Generation throughput | 42.9 tok/s | 2.4 tok/s |
| Avg prompt latency | 0.93 s | 16.81 s |
| Peak memory (fp32) | ~311 MB | ~11.8 GB |
| Empathy eval throughput | 12.2 tok/s | 5.6 tok/s |

### 2.3 What These Numbers Mean

- **The 757M SLM has no recorded MMLU / HellaSwag / HumanEval / MATH / GSM8K / ARC score.** It has not been evaluated on standard academic benchmarks. The only quality signal in the repo is an empathy eval showing that, even after fine-tuning on empathetic dialogue, the model produces repetitive/fragmented outputs compared to Qwen2.5-0.5B.
- **The speed advantage is real and large.** On CPU, the 757M SLM is ~55x faster per forward pass and ~18x faster at generation than the 0.5B Qwen base. This is because CPU inference is memory-bandwidth bound, and the SLM has a much smaller activation footprint.
- **The quality gap is also real.** A 757M model trained from scratch on ~120M code tokens plus ~2.5M empathy tokens does not compete with 3B–9B models trained on trillions of tokens. The repo's own empathy eval notes: "produces repetitive, fragmented outputs."

---

## 3. Quantization Support

| Model | FP16 | INT8 | INT4 / GPTQ / AWQ | GGUF (llama.cpp) | Notes |
|-------|------|------|--------------------|------------------|-------|
| **This SLM (757M)** | Yes (fp32 in repo; fp16 possible) | No | No | No | No quantization pipeline exists. State dict is PyTorch zip (float32). |
| **Phi-4-mini 3.8B** | Yes | Yes | Yes (AWQ, GPTQ) | Yes (Q4_K_M, Q5_K_M, Q8_0, FP16) | ~2.5 GB at Q4_K_M |
| **Qwen2.5-7B** | Yes | Yes | Yes (AWQ, GPTQ) | Yes | Widely available |
| **Llama-3.2-1B/3B** | Yes | Yes | Yes (SpinQuant, GPTQ, AWQ) | Yes | ExecuTorch 4-bit groupwise also available |
| **Ministral-3B/8B** | Yes (FP8 official) | Yes | Yes | Yes (Q4_K_M, Q5_K_M, Q8_0, BF16) | Ministral-8B-Instruct ships in FP8 |
| **Gemma-2-2B/9B** | Yes | Yes | Yes | Yes | Q4_0, Q4_1, Q3_K_L available |
| **SmolLM2-1.7B** | Yes | Yes | Yes | Yes (Q2_K, Q4_0, Q5_0, Q8_0) | Official Ollama tags |

**Gap:** The 757M SLM has no quantization artifacts. Converting to GGUF, GPTQ, or AWQ would require building a pipeline and validating quality after quantization.

---

## 4. Serving Frameworks

| Model | Ollama | vLLM | llama.cpp | TGI | TensorRT-LLM | Notes |
|-------|--------|------|-----------|-----|--------------|-------|
| **This SLM (757M)** | No | No | No | No | No | Only a raw PyTorch inference script (`slm-api.py`, `generate_samples.py`). No chat server, no OpenAI-compatible API. |
| **Phi-4-mini 3.8B** | Yes | Yes | Yes | Yes | Yes | Requires Ollama ≥0.5.13 |
| **Qwen2.5-7B** | Yes | Yes | Yes | Yes | Yes | Standard support |
| **Llama-3.2-1B/3B** | Yes | Yes | Yes | Yes | Yes | ExecuTorch on ARM |
| **Ministral-3B/8B** | Yes (0.13.1+) | Yes | Yes | Yes | Yes | 256K context via YaRN |
| **Gemma-2-2B/9B** | Yes | Yes | Yes | Yes | Yes | |
| **SmolLM2-1.7B** | Yes | Yes | Yes | Yes | Yes | Official Ollama tags |

**Gap:** The 757M SLM has no production serving path. There is no chat completion endpoint, no batching, no continuous batching, and no streaming interface outside of ad-hoc scripts.

---

## 5. Production-Grade Gap Analysis

### 5.1 What the Project Has
- A working from-scratch GPT-style transformer (~757M params).
- Reproducible training and evaluation scripts.
- CPU-only forward-pass and generation benchmarks.
- An empathy fine-tune with behavioral eval (5 probes, 1 passed).
- Clean architecture documentation (`SLM_ARCHITECTURE.md`).
- Gradient checkpointing enabled in training config.

### 5.2 What the Project Needs to Reach Production Grade

#### A. Evaluation (Honest Quality Signal)
- **Run standard benchmarks.** MMLU, HellaSwag, WinoGrande, HumanEval, MATH, GSM8K, and ARC must be run via `lm-eval-harness` or equivalent. Without these, the model has no comparable quality claim.
- **Add a held-out test split.** The repo currently has only a validation split. A separate test set prevents benchmark overfitting during prompt engineering.
- **Evaluate the base model before and after fine-tuning.** Report the delta so training value is visible.

#### B. Training Data & Quality
- **Scale training data.** 757M parameters need more than ~120M code tokens + ~2.5M empathy tokens to show general capability. State-of-the-art small models use trillions of tokens or highly curated multi-domain corpora.
- **Dedup and clean.** No evidence of deduplication, PII filtering, or quality filtering in the repo's preprocessing pipeline.
- **License hygiene.** The empathy data is CC BY-NC-4.0 (non-commercial). Any deployment that uses that fine-tune is restricted. The base code corpus license is also mixed. A production artifact needs a clear commercial license.

#### C. Architecture & Context
- **Extend context window.** 256 tokens is insufficient for most real-world applications (empathetic dialogue, code generation, RAG). RoPE scaling or ALiBi extension to 2K–8K is a minimum viable change.
- **Add a chat template.** The current model is a flat text continuation model. Production assistants need a system prompt, role formatting, and stop-token control.

#### D. Quantization & Optimization
- **Implement INT4/INT8 quantization.** At 757M params, fp16 is ~1.5 GB; INT4 is ~400 MB. Quantization is required for edge and consumer GPU deployment.
- **Add GGUF export.** This enables llama.cpp and Ollama deployment, which are the lowest-friction serving paths for a 757M model.

#### E. Serving & API
- **Build an OpenAI-compatible server.** vLLM, TGI, or a lightweight FastAPI wrapper with streaming is required for any integration test.
- **Add continuous batching and prefix caching.** These are standard in production inference servers and matter more at higher QPS.

#### F. Safety & Alignment
- **Expand behavioral eval.** 5 probes is not a safety suite. Add red-teaming for jailbreaks, toxicity, PII leakage, and medical/legal advice.
- **Add refusal and redirect behavior.** The stage-2 steering set in `slm-ft/adapter_stage2/` is a start, but it is hand-written and small (30 dialogs). A production model needs systematic safety training.

#### G. Reproducibility & Release Hygiene
- **Publish a model card with known failure modes.** The repo should state explicitly: "This model has not been evaluated on MMLU, HellaSwag, HumanEval, MATH, GSM8K, or ARC. Quality is unvalidated against public benchmarks."
- **Release weights in safetensors.** The current checkpoint is a PyTorch zip bundle. Safetensors is the community standard.
- **Pin the tokenizer and training code version.** The repo uses `tiktoken`; version drift in tokenizer implementation changes byte-pair merges.

---

## 6. Speed/Quality Tradeoff Summary

| Model | Approx. Size | Gen. Quality | Gen. Speed (CPU) | VRAM (INT4) | Production Readiness |
|-------|-------------|--------------|------------------|-------------|---------------------|
| **This SLM** | 757M | **Unvalidated** (empathy eval: weak) | **42.9 tok/s** | N/A (no quant) | Research prototype |
| **Phi-4-mini** | 3.8B | Excellent (67.3 MMLU, 64.0 MATH) | ~2–5 tok/s | ~2.5 GB | Production-ready |
| **Qwen2.5-7B** | 7B | Excellent (74.2 MMLU, 57.9 HumanEval) | ~2–5 tok/s | ~4.5 GB | Production-ready |
| **Llama-3.2-1B** | 1.2B | Weak (32.2 MMLU) | Fast | ~0.7 GB | Edge-ready |
| **Llama-3.2-3B** | 3.2B | Moderate (58.0 MMLU) | Moderate | ~1.9 GB | Edge-ready |
| **Ministral-3B** | 3B | Moderate (60.9 MMLU) | Moderate | ~1.8 GB | Edge-ready |
| **Ministral-8B** | 8.4B | Strong (65.0 MMLU) | Moderate | ~4.5 GB | Production-ready |
| **Gemma-2-2B** | 2B | Weak (52.4 MMLU) | Moderate | ~1.2 GB | Edge-ready |
| **Gemma-2-9B** | 9.2B | Strong (71.3 MMLU) | Slow on CPU | ~5.0 GB | Production-ready |
| **SmolLM2-1.7B** | 1.7B | Weak (N/R MMLU, 19.4 MMLU-Pro) | Fast | ~1.0 GB | Edge-ready |

**Key takeaway:** The 757M SLM's only verified advantage is raw CPU inference speed. It has no verified quality advantage over any model in this table, and its 256-token context window is the shortest by an order of magnitude. If the goal is empathetic dialogue, Phi-4-mini or Llama-3.2-3B fine-tuned on the same data will outperform this model on every quality axis. If the goal is CPU-only edge speed with acceptable quality, SmolLM2-1.7B or Llama-3.2-1B are better-documented, better-served alternatives.

---

## 7. Recommended Path Forward

1. **Week 1:** Run `lm-eval-harness` on the current 757M checkpoint (base and fine-tuned) to get real MMLU / HellaSwag / HumanEval / MATH / GSM8K / ARC numbers.
2. **Week 2:** Extend context to 2K–4K tokens using RoPE scaling or positional interpolation. Re-run benchmarks.
3. **Week 3:** Export to GGUF (Q4_K_M, Q5_K_M) and validate perplexity parity with fp16.
4. **Week 4:** Wrap in a FastAPI streaming server with an OpenAI-compatible `/v1/chat/completions` endpoint.
5. **Week 5:** Add a behavioral eval suite of ≥50 probes and a held-out test split.

Do not skip step 1. Publishing a production comparison without MMLU / HumanEval / MATH numbers for your own model is worse than publishing no comparison at all — it signals that the numbers would not help the story.

---

## 8. Sources & Provenance

- Phi-4-mini benchmarks: [unsloth/Phi-4-mini-instruct-GGUF](https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF) and [aiwiki.ai](https://aiwiki.ai/wiki/phi_4_mini).
- Qwen2.5-7B benchmarks: [Qwen2.5 LLM blog](https://qwen.ai/blog?id=qwen2.5-llm) and [Emergent Mind](https://www.emergentmind.com/topics/qwen2-5-7b-base-model).
- Llama-3.2-1B/3B benchmarks: [meta-llama/Llama-3.2-1B README](https://huggingface.co/meta-llama/Llama-3.2-1B).
- Ministral-3B/8B benchmarks: [ai.azure.com model catalog](https://ai.azure.com/catalog/models/Ministral-3B) and [Hugging Face model card](https://huggingface.co/Columbidae/Ministral-8B-Instruct-2410).
- Gemma-2-2B/9B benchmarks: [Google AI model card](https://ai.google.dev/gemma/docs/core/model_card_2) and [arxiv 2408.00118](https://arxiv.org/html/2408.00118v1).
- SmolLM2-1.7B benchmarks: [HuggingFaceTB/SmolLM2-1.7B README](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) and [arxiv 2502.02737](https://arxiv.org/pdf/2502.02737).
- 757M SLM specs: `SLM_ARCHITECTURE.md`, `MODEL_COMPARISON.md`, `benchmarks/slm_forward_bench.json`, `benchmarks/benchmark_report.md`.
