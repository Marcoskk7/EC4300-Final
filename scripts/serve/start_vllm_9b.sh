#!/bin/bash
set -e
source /opt/data/private/isaaclab/ec4300/venvs/vllm/bin/activate
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve /opt/data/private/isaaclab/ec4300-9b/hf-cache/Qwen3.5-9B-bf16 \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --attention-backend TRITON_ATTN
