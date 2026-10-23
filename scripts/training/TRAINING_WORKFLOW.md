# Complete Model Upgrade Workflow: From Data to Decent Responses

This guide walks you through upgrading your Emotional SLM from stub to production-ready model with decent responses.

**Timeline**: ~6-8 hours end-to-end (including model training).

## Architecture Overview

```
1. Generate Synthetic Data (1.5 hours)
   ↓
2. Mix with Real Data (30 min)
   ↓
3. Train SFT Model (3-4 hours on 3090)
   ↓
4. Evaluate Quality (30 min)
   ↓
5. Serve with vLLM (30 min)
   ↓
6. Integrate into Backend (15 min)
```

---

## Step 1: Generate Synthetic Supportive Data (1.5 hours)

### Start your local LLM

Using vLLM (recommended for speed):
```bash
# Install vLLM if needed
pip install vllm

# Start vLLM on port 8000
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16
```

Or using Ollama:
```bash
ollama serve
# (in another terminal)
ollama pull mistral
```

### Generate high-quality synthetic data

```bash
# Install dependencies first
pip install -r scripts/training/requirements-training.txt

# Generate ~1000 high-quality examples (takes ~1-1.5 hours)
python scripts/training/generate_synthetic_data_v2.py \
  --local_llm_url http://localhost:8000/v1 \
  --model_name mistral-7b-instruct \
  --output_file data/synthetic_sft.jsonl \
  --num_samples 1000 \
  --quality_threshold 0.70 \
  --max_retries_per_combo 2
```

### Inspect output

```bash
# Check first few examples
head -3 data/synthetic_sft.jsonl | python -m json.tool

# Stats
python -c "
import json
from collections import Counter
from pathlib import Path
examples = [json.loads(line) for line in Path('data/synthetic_sft.jsonl').read_text().split('\n') if line]
print(f'Total: {len(examples)}')
print(f'Emotions: {dict(Counter(e[\"emotion\"] for e in examples))}')
print(f'Avg quality: {sum(e[\"quality_score\"] for e in examples) / len(examples):.3f}')
"
```

---

## Step 2: Optional - Mix with Real Data (30 min)

If you have real supportive conversations or curated examples, mix them 50/50:

```bash
# Combine synthetic + real
python -c "
import json
from pathlib import Path
import random

synthetic = [json.loads(line) for line in Path('data/synthetic_sft.jsonl').read_text().split('\n') if line]
real = [json.loads(line) for line in Path('data/real_examples.jsonl').read_text().split('\n') if line] if Path('data/real_examples.jsonl').exists() else []

combined = synthetic + real
random.shuffle(combined)

# 80/10/10 split
n = len(combined)
train_n = int(n * 0.80)
val_n = int(n * 0.10)

with open('data/train.jsonl', 'w') as f:
    for ex in combined[:train_n]:
        f.write(json.dumps(ex) + '\n')

with open('data/val.jsonl', 'w') as f:
    for ex in combined[train_n:train_n+val_n]:
        f.write(json.dumps(ex) + '\n')

with open('data/test.jsonl', 'w') as f:
    for ex in combined[train_n+val_n:]:
        f.write(json.dumps(ex) + '\n')

print(f'Train: {len(list(open(\"data/train.jsonl\")))} examples')
print(f'Val: {len(list(open(\"data/val.jsonl\")))} examples')
print(f'Test: {len(list(open(\"data/test.jsonl\")))} examples')
"
```

---

## Step 3: Train the Model (3-4 hours on RTX 3090)

### Start training

```bash
python scripts/training/train_sft_qlora.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --train_file data/train.jsonl \
  --val_file data/val.jsonl \
  --output_dir checkpoints/emotional-slm-v1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --max_seq_length 2048 \
  --save_steps 50 \
  --eval_steps 50 \
  --logging_steps 5
```

**What happens:**
- Model loads with 4-bit quantization (fits in 24GB VRAM)
- LoRA adapters applied (~100M params to train vs 7B base)
- Trains for 1 epoch (~100-200 steps depending on data size)
- Saves checkpoints every 50 steps
- Evaluates on validation set

**Monitor training:**
- Watch `/checkpoints/emotional-slm-v1/` for best model
- Check loss curve: should steadily decrease
- Typical: initial loss ~2.5, final loss ~0.8-1.2

---

## Step 4: Optional - Merge Adapters for Serving

For easier vLLM serving, merge LoRA adapters into base model:

```bash
python scripts/training/merge_adapters.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --adapter_dir checkpoints/emotional-slm-v1 \
  --output_dir checkpoints/emotional-slm-v1-merged
```

This creates a standard Hugging Face model (~13GB) ready for vLLM.

---

## Step 5: Serve the Trained Model

### Option A: Direct vLLM (if merged)
```bash
vllm serve checkpoints/emotional-slm-v1-merged \
  --port 8001 \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16
```

### Option B: vLLM with LoRA adapters
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8001 \
  --enable-lora \
  --lora-modules emotional-slm=checkpoints/emotional-slm-v1 \
  --gpu-memory-utilization 0.8
```

### Test the model
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [
      {"role": "user", "content": "I feel really anxious about work"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

---

## Step 6: Integrate into Backend

Update `.env`:
```bash
HF_API_TOKEN=local-model-token
HF_MODEL_URL=http://localhost:8001/v1
INFERENCE_MODEL_NAME=mistral-7b-instruct
INFERENCE_PROVIDER=huggingface
```

Restart the backend:
```bash
docker-compose up -d --build
```

Test the chat endpoint:
```bash
# Sign up
curl -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "test-password-123",
    "accept_terms": true,
    "confirm_age_18": true
  }'

# Extract access_token from response
export TOKEN="<access_token>"

# Create conversation
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test"}'

# Extract conversation_id
export CONV_ID="<conversation_id>"

# Send message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "'$CONV_ID'",
    "content": "I feel overwhelmed at work"
  }'
```

---

## Step 7: Evaluate Quality

### Manual spot-check
```bash
# Review 20-30 model responses for:
# ✓ Warm, validating tone
# ✓ Shows understanding (not advice-giving)
# ✓ Natural length (1-3 sentences)
# ✓ No generic/preachy responses
# ✗ No medical claims
# ✗ No dismissive language
```

### Automated evaluation (optional)
```bash
python -c "
import json
from pathlib import Path

# Load test set
test_examples = [json.loads(line) for line in Path('data/test.jsonl').read_text().split('\n') if line]

print(f'Test set: {len(test_examples)} examples')
print(f'Emotions: {set(e[\"emotion\"] for e in test_examples)}')
print(f'Sample contexts: {list({e[\"context\"] for e in test_examples})[:3]}')
"
```

---

## Troubleshooting

### Model gives generic/unhelpful responses
- **Issue**: Not enough training or low data quality
- **Fix**: 
  - Increase `--num_train_epochs` to 1.5-2
  - Raise `--quality_threshold` to 0.75+ in data generation
  - Check training loss is decreasing (initial ~2.5, final <1.0)

### Out of memory during training
- **Issue**: Too large batch size or sequence length
- **Fix**:
  - Reduce `--per_device_train_batch_size` to 1
  - Increase `--gradient_accumulation_steps` to 32
  - Reduce `--max_seq_length` to 1024

### vLLM serving slow
- **Issue**: Model not using CUDA efficiently
- **Fix**:
  - Check GPU utilization: `nvidia-smi`
  - Increase `--gpu-memory-utilization` to 0.9
  - Use merged model instead of LoRA for faster inference

### Backend not using new model
- **Issue**: Config not reloaded
- **Fix**:
  - Restart backend containers: `docker-compose restart api`
  - Verify `.env` has correct `HF_MODEL_URL`
  - Check logs: `docker-compose logs api`

---

## What "Decent Responses" Means

After this workflow, your model should:
- ✓ Respond to emotional contexts with warmth, not clinical tone
- ✓ Show understanding ("I hear you", "That sounds difficult")
- ✓ Offer grounded support (listening, validation, simple coping ideas)
- ✓ Avoid generic platitudes or dismissive language
- ✓ Keep responses 1-3 sentences (not long-winded)
- ✓ Never give medical/psychiatric advice
- ✓ Handle crisis inputs safely (routed to policy, not model)

---

## Next Steps (After v1)

1. **Collect real user feedback**
   - Tag responses as helpful/unhelpful
   - Curate hard examples for v2 retraining

2. **Add preference tuning**
   - Collect prefer/disprefer pairs
   - Run DPO (Direct Preference Optimization)
   - Further align model to your values

3. **Monitor in production**
   - Track mood classification accuracy
   - Monitor response quality metrics
   - Collect edge cases for retraining

4. **Iterate monthly**
   - Retrain with new data
   - Re-evaluate on test set
   - Deploy improved versions with rollback plan
