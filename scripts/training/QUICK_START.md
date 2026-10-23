# Quick Start Checklist - Upgrade Your SLM to Decent Responses

✓ = Done | ⏳ = In progress | ✗ = Not done

## Hardware Check
- [ ] RTX 3090 with 24GB VRAM available
- [ ] 50GB+ free disk space for models + data
- [ ] 1-2 hours for synthetic data generation
- [ ] 3-4 hours for model training

## Prerequisites
- [ ] Install vLLM: `pip install vllm`
- [ ] Install training deps: `pip install -r scripts/training/requirements-training.txt`
- [ ] Pull base model: `huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2` (optional, auto-downloaded)

## Workflow

### Phase 1: Data Generation (1.5 hours)
- [ ] Start vLLM: `vllm serve mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --gpu-memory-utilization 0.8`
- [ ] Generate synthetic data:
  ```bash
  python scripts/training/generate_synthetic_data_v2.py \
    --local_llm_url http://localhost:8000/v1 \
    --model_name mistral-7b-instruct \
    --output_file data/synthetic_sft.jsonl \
    --num_samples 1000 \
    --quality_threshold 0.70
  ```
- [ ] Check output: `head -5 data/synthetic_sft.jsonl | jq`
- [ ] Run stats:
  ```bash
  python -c "
  import json
  from pathlib import Path
  examples = [json.loads(line) for line in Path('data/synthetic_sft.jsonl').read_text().split('\n') if line]
  print(f'Total: {len(examples)}')
  print(f'Avg quality: {sum(e[\"quality_score\"] for e in examples) / len(examples):.3f}')
  "
  ```

### Phase 2: Data Split (30 min)
- [ ] Create train/val/test splits:
  ```bash
  python -c "
  import json, random
  from pathlib import Path
  examples = [json.loads(line) for line in Path('data/synthetic_sft.jsonl').read_text().split('\n') if line]
  random.shuffle(examples)
  n = len(examples)
  train_n = int(n * 0.80)
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
  print(f'Train: {train_n}, Val: {val_n}, Test: {n-train_n-val_n}')
  "
  ```
- [ ] Verify files exist: `ls -lh data/*.jsonl`

### Phase 3: Train Model (3-4 hours, unattended)
- [ ] **IMPORTANT**: Move vLLM to different GPU or close it to free 24GB VRAM
- [ ] Start training:
  ```bash
  python scripts/training/train_sft_qlora.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --train_file data/train.jsonl \
    --val_file data/val.jsonl \
    --output_dir checkpoints/emotional-slm-v1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1
  ```
- [ ] Monitor: 
  - Watch `checkpoints/emotional-slm-v1/` for checkpoint files
  - Check loss is decreasing (initial ~2.5 → final ~0.8-1.2)
  - GPU util should be 70-90%

### Phase 4: Merge & Serve (30 min)
- [ ] Merge adapters:
  ```bash
  python scripts/training/merge_adapters.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter_dir checkpoints/emotional-slm-v1 \
    --output_dir checkpoints/emotional-slm-v1-merged
  ```
- [ ] Start vLLM with trained model:
  ```bash
  vllm serve checkpoints/emotional-slm-v1-merged \
    --port 8001 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16
  ```
- [ ] Test model:
  ```bash
  curl -X POST http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mistral-7b-instruct",
      "messages": [{"role": "user", "content": "I feel really anxious"}],
      "temperature": 0.7,
      "max_tokens": 150
    }' | jq .choices[0].message.content
  ```

### Phase 5: Integrate into Backend (15 min)
- [ ] Update `.env`:
  ```
  HF_API_TOKEN=local-model
  HF_MODEL_URL=http://localhost:8001/v1
  INFERENCE_MODEL_NAME=mistral-7b-instruct
  INFERENCE_PROVIDER=huggingface
  ```
- [ ] Restart backend: `docker-compose up -d --build`
- [ ] Test chat endpoint (see TRAINING_WORKFLOW.md for curl commands)

### Phase 6: Validate Quality (30 min)
- [ ] Test 5-10 real conversations
- [ ] Manually check for:
  - [ ] Warm, validating tone
  - [ ] Shows understanding (not advice-giving)
  - [ ] Natural length
  - [ ] No generic responses
  - [ ] No medical claims
- [ ] Check emotion classification works
- [ ] Test crisis routing (type "I'm going to kill myself")

## Quality Checklist (After Training)

Does your model now:
- [ ] Respond warmly to emotional contexts?
- [ ] Show understanding ("I hear you")?
- [ ] Avoid generic advice?
- [ ] Keep responses 1-3 sentences?
- [ ] Never give medical advice?
- [ ] Route crisis inputs safely?

If **all checked**: ✅ Model is at "decent response" level!

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| OOM during training | Reduce batch to 1, increase grad accumulation to 32 |
| Generic responses | Increase training epochs to 1.5-2 or data quality threshold |
| Slow inference | Use merged model, increase GPU util to 0.9 |
| Backend not using new model | Restart API: `docker-compose restart api` |
| vLLM can't load model | Check path exists, `nvidia-smi` shows memory |

## Time Estimate
- Data generation: 1-1.5 hours ⏳
- Data split: 5 min
- Training: 3-4 hours ⏳ (can go unattended)
- Merge & serve: 15 min
- Integration: 15 min
- **Total: ~5-6 hours** (mostly waiting for training)

---

**Ready to go?** Start with Phase 1:
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --gpu-memory-utilization 0.8
```
