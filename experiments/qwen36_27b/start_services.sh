#!/usr/bin/env bash
# Start Pi0.5 and SAM3 on this machine. Run on VM-0-14, not on the Kimi eval node.
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$EXP_ROOT/services"
cd "$RPENT_ROOT"

start() {
  local name=$1 port=$2; shift 2
  if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$port"; then
    echo "$name already answering on :$port"
    return
  fi
  nohup "$@" > "$EXP_ROOT/services/$name.log" 2>&1 &
  echo $! > "$EXP_ROOT/services/$name.pid"
  echo "started $name (pid $!) -> $EXP_ROOT/services/$name.log"
}

start vla "$VLA_PORT" python -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --cuda-device "$VLA_GPU" \
  --transport http --host 127.0.0.1 --port "$VLA_PORT"

start sam3 "$SAM3_PORT" python -m rpent.robots.components.sam3_server \
  --cuda-device "$SAM3_GPU" \
  --transport http --host 127.0.0.1 --port "$SAM3_PORT"

for name in vla sam3; do
  log="$EXP_ROOT/services/$name.log"
  for _ in $(seq 1 180); do
    grep -q "RPC server listening" "$log" 2>/dev/null && { echo "$name ready"; break; }
    sleep 5
  done
  grep -q "RPC server listening" "$log" || { echo "$name did not come up; see $log" >&2; exit 1; }
done
