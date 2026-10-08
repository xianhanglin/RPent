#!/usr/bin/env bash
# Serve Qwen3.6-27B with vLLM on this machine, thinking disabled by default.
# Run inside tmux; ready when the log prints "Application startup complete".
set -euo pipefail
source "$(dirname "$0")/env.sh"

# Activating (not just calling the env's binary) runs the env's libstdc++ preload hook.
set +u
source /cfsdata/agent/miniconda3/etc/profile.d/conda.sh
conda activate vllm
set -u

mkdir -p "$EXP_ROOT/services"
export CUDA_VISIBLE_DEVICES="$QWEN_GPUS"
vllm serve "$QWEN_WEIGHTS" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$(awk -F, '{print NF}' <<< "$QWEN_GPUS")" \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --host 127.0.0.1 --port "$QWEN_PORT" \
  2>&1 | tee "$EXP_ROOT/services/qwen_vllm.log"
