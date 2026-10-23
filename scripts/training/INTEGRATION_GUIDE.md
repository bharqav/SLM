"""Integration guide and setup for using your trained SLM in production.

After training with train_sft_qlora.py, use your model in the backend.
"""

# STEP 1: Train your model
# python scripts/training/train_sft_qlora.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.2 \
#   --train_file data/train.jsonl \
#   --val_file data/val.jsonl \
#   --output_dir checkpoints/emotional-slm-v1 \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --num_train_epochs 1

# STEP 2: Merge adapters into base model (optional, for easier serving)
# python scripts/training/merge_adapters.py \
#   --base_model mistralai/Mistral-7B-Instruct-v0.2 \
#   --adapter_dir checkpoints/emotional-slm-v1 \
#   --output_dir checkpoints/emotional-slm-v1-merged

# STEP 3: Serve the model with vLLM
# vllm serve checkpoints/emotional-slm-v1-merged \
#   --port 8001 \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.8

# STEP 4: Update backend config to use the new model
# .env:
#   HF_MODEL_URL=http://localhost:8001/v1
#   HF_API_TOKEN=fake-token-for-local-model
#   INFERENCE_MODEL_NAME=emotional-slm-v1

# STEP 5: Restart backend
# docker-compose up --build

print("SFT Training Integration Guide")
print("See comments above for step-by-step instructions")
