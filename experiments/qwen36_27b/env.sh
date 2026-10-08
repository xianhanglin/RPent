# Shared configuration for the Qwen3.6-27B LIBERO-Pro validation runs. Source it; don't run it.
# Reproduces the leaderboard row "Qwen3.6 27B / no-reasoning" (70.63%, 565/800).

export RPENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RPENT_BIN=/cfsdata/agent/miniconda3/envs/rpent/bin
export PATH="$RPENT_BIN:$PATH"
export EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QWEN_WEIGHTS=/cfsdata/agent/models/Qwen3.6-27B
export QWEN_GPUS="${QWEN_GPUS:-2,3}"
export QWEN_PORT="${QWEN_PORT:-8000}"

export MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
# vLLM's Anthropic endpoint ignores the request's `thinking` field, so "none" only takes
# effect because serve_qwen.sh also sets enable_thinking=false as the server default.
export REASONING_EFFORT="${REASONING_EFFORT:-none}"
export RUN_TAG="${RUN_TAG:-qwen3.6-27b_${REASONING_EFFORT}_vllm}"

export ANTHROPIC_BASE_URL="http://127.0.0.1:$QWEN_PORT"
export ANTHROPIC_API_KEY=EMPTY
unset ANTHROPIC_AUTH_TOKEN
# Claude Code routes background/subagent calls to tier aliases; keep every tier on the model under test.
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

export LIBERO_TYPE=pro
export PI05_CHECKPOINT_PATH="$RPENT_ROOT/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
export SAM3_CHECKPOINT_PATH="$RPENT_ROOT/checkpoints/sam3/sam3.pt"
export MEMORY_DIR="$RPENT_ROOT/memory_pinned/libero"

# On VM-0-14 this kit starts its own Pi0.5 and SAM3 (see start_services.sh).
# Do not point these at VM-0-12; that node is reserved for the Kimi eval.
export VLA_GPU="${VLA_GPU:-0}"
export SAM3_GPU="${SAM3_GPU:-1}"
export VLA_PORT="${VLA_PORT:-8220}"
export SAM3_PORT="${SAM3_PORT:-8114}"
# GPUs that per-episode env_servers (MuJoCo + EGL rendering) are spread over; 2,3 serve Qwen.
export ENV_GPUS="${ENV_GPUS:-4 5 6 7}"
