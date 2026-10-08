#!/usr/bin/env bash
# Verify the Kimi endpoint and key before spending time on simulator startup.
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ -n "$MOONSHOT_API_KEY" && "$MOONSHOT_API_KEY" != sk-REPLACE_ME ]] || { echo "set MOONSHOT_API_KEY in $KIMI_ENV_FILE" >&2; exit 1; }

echo "backend: ${KIMI_BACKEND:-api}  base url: $ANTHROPIC_BASE_URL  model: $MODEL  auth: ${MOONSHOT_AUTH_MODE:-x-api-key}"
cd "$RPENT_ROOT"
rpent-check-llm --planner claude_code --model "$MODEL" --json
