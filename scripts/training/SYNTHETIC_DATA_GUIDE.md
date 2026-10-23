# High-Quality Synthetic Data Generation Guide

## Overview

The improved `generate_synthetic_data_v2.py` produces training data that aligns with your base model's style and format.

Key improvements:
- **Tone alignment scoring**: Validates supportive tone (validating, understanding) vs. generic/dismissive patterns
- **Degenerate response filtering**: Rejects low-quality LLM outputs
- **Model-aware formatting**: Outputs exact chat format for training
- **Quality blending**: Combines LLM's self-assessment with alignment scoring

## Quick Start

### 1. Install dependencies
```bash
pip install -r scripts/training/requirements-training.txt
```

### 2. Start your local LLM

Using vLLM (recommended):
```bash
# For Mistral 7B
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --gpu-memory-utilization 0.8

# For Qwen
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.8
```

Using Ollama:
```bash
# Start Ollama server
ollama serve

# In another terminal, pull model
ollama pull mistral
# Then set OLLAMA_MODELS if needed
```

### 3. Generate data
```bash
python scripts/training/generate_synthetic_data_v2.py \
  --local_llm_url http://localhost:8000/v1 \
  --model_name mistral-7b-instruct \
  --base_model_name mistral-7b-instruct \
  --output_file data/synthetic_sft.jsonl \
  --num_samples 1000 \
  --quality_threshold 0.65
```

### 4. Inspect output
```bash
# Check first few examples
head -3 data/synthetic_sft.jsonl | jq .

# Count examples per emotion
python -c "
import json
from collections import Counter
emotions = Counter()
with open('data/synthetic_sft.jsonl') as f:
    for line in f:
        data = json.loads(line)
        emotions[data['emotion']] += 1
print(dict(emotions))
"
```

## Understanding Quality Scores

Each generated example has two scores:

- **quality_score** (0-1): Overall quality (40% LLM self-assessment + 60% your alignment feedback)
- **alignment_score** (0-1): How well the response matches supportive tone

### Tone Alignment Markers

✓ **Good tone** (indicative):
- "understand", "valid", "difficult", "take", "step", "support", "hear", "feel"

✗ **Bad tone** (automatic reject):
- "you should", "just", "simply", "obviously", "clearly", "toxic", "eat more"

## Quality Control

Filter levels (adjust `--quality_threshold`):
- **0.80+**: Only top-tier, highly aligned responses (strictest)
- **0.70-0.79**: High quality, good for fine-tuning
- **0.60-0.69**: Acceptable, mixed quality
- **0.50-0.59**: Permissive (not recommended)

For production training on a 3090:
```bash
python scripts/training/generate_synthetic_data_v2.py \
  --local_llm_url http://localhost:8000/v1 \
  --model_name mistral-7b-instruct \
  --output_file data/synthetic_sft_production.jsonl \
  --num_samples 2000 \
  --quality_threshold 0.70 \
  --max_retries_per_combo 3
```

## Troubleshooting

**Problem**: Responses are generic/preachy
- Increase `--quality_threshold` to 0.75+
- Check that your generator LLM has good instruction-following capability
- Verify system prompt in script matches your target tone

**Problem**: Too few examples generated
- Lower `--quality_threshold` to 0.60-0.65
- Increase `--max_retries_per_combo` to 3-5
- Check your local LLM is running and accessible

**Problem**: LLM API timeout
- Increase timeout in `call_local_llm()` function
- Check local LLM GPU memory/load
- Reduce concurrent requests if running multiple scripts

## Next Steps

Once you have clean synthetic data:

1. **Split into train/val/test**:
   ```bash
   python -c "
   import json
   import random
   examples = [json.loads(line) for line in open('data/synthetic_sft.jsonl')]
   random.shuffle(examples)
   n = len(examples)
   train_n = int(n * 0.85)
   val_n = int(n * 0.10)
   
   with open('data/train.jsonl', 'w') as f:
       for ex in examples[:train_n]:
           f.write(json.dumps(ex) + '\n')
   with open('data/val.jsonl', 'w') as f:
       for ex in examples[train_n:train_n+val_n]:
           f.write(json.dumps(ex) + '\n')
   with open('data/test.jsonl', 'w') as f:
       for ex in examples[train_n+val_n:]:
           f.write(json.dumps(ex) + '\n')
   "
   ```

2. **Mix with real data** (if available):
   - Combine synthetic (50-70%) + real curated data (30-50%)
   - Use same format (messages list with role/content)

3. **Train with SFT**:
   - Use `scripts/training/train_sft_qlora.py` (when created)
   - Target: 1-2 epochs, low LR (2e-4), QLoRA rank 64

4. **Evaluate on test set**:
   - Check tone alignment manually (10-20 random samples)
   - Validate emotion classification accuracy
   - Check safety rejection patterns
