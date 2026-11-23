#!/bin/bash
set -e
source /opt/data/private/isaaclab/ec4300/venvs/vllm/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_V1=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve /opt/data/private/isaaclab/ec4300/hf-cache/Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --language-model-only \
  --reasoning-parser qwen3 \
  --attention-backend TRITON_ATTN \
  --additional-config '{"gdn_prefill_backend":"triton"}'
