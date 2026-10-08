#!/usr/bin/env bash
# Verify the local Qwen endpoint before spending time on simulator startup.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "base url: $ANTHROPIC_BASE_URL  model: $MODEL  reasoning: $REASONING_EFFORT"
cd "$RPENT_ROOT"
rpent-check-llm --planner claude_code --model "$MODEL" --json
