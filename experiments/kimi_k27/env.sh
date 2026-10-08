# Shared configuration for the Kimi K2.7 LIBERO-Pro experiments. Source it; don't run it.

KIMI_ENV_FILE="${KIMI_ENV_FILE:-$HOME/.config/rpent/kimi.env}"
if [[ -f "$KIMI_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$KIMI_ENV_FILE"
  set +a
else
  echo "note: $KIMI_ENV_FILE not found; planner runs will fail until it exists" >&2
fi

export RPENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RPENT_BIN=/cfsdata/agent/miniconda3/envs/rpent/bin
export PATH="$RPENT_BIN:$PATH"
export EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-kimi-k2.7-code}"
# kimi-k2.7-code cannot run with thinking disabled, so "none" is not a valid setting here.
export REASONING_EFFORT="${REASONING_EFFORT:-high}"
export RUN_TAG="${RUN_TAG:-${MODEL}_${REASONING_EFFORT}_${KIMI_BACKEND:-api}}"

export ANTHROPIC_BASE_URL="${MOONSHOT_ANTHROPIC_BASE_URL:-}"
MOONSHOT_API_KEY="${MOONSHOT_API_KEY:-}"
unset ANTHROPIC_AUTH_TOKEN
if [[ "${MOONSHOT_AUTH_MODE:-x-api-key}" == "bearer" ]]; then
  export ANTHROPIC_AUTH_TOKEN="$MOONSHOT_API_KEY"
  export ANTHROPIC_API_KEY="$MOONSHOT_API_KEY"  # rpent-check-llm only looks for this name
else
  export ANTHROPIC_API_KEY="$MOONSHOT_API_KEY"
fi
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

# This machine has NVIDIA's EGL driver but not the public libEGL.so.1 dispatch
# library. Unpack libegl1 into this directory (no root required) so MuJoCo can
# render offscreen for LIBERO camera images.
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${RPENT_ROOT}/third_party/egl/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export VLA_GPU="${VLA_GPU:-0}"
export SAM3_GPU="${SAM3_GPU:-1}"
export VLA_PORT="${VLA_PORT:-8220}"
export SAM3_PORT="${SAM3_PORT:-8114}"
# GPUs that per-episode env_servers (MuJoCo + EGL rendering) are spread over.
export ENV_GPUS="${ENV_GPUS:-2 3 4 5 6 7}"
