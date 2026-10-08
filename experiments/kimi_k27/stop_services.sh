#!/usr/bin/env bash
set -uo pipefail
EXP_ROOT="$(cd "$(dirname "$0")" && pwd)"
for name in vla sam3; do
  pidfile="$EXP_ROOT/services/$name.pid"
  [[ -f "$pidfile" ]] || continue
  kill "$(cat "$pidfile")" 2>/dev/null && echo "stopped $name"
  rm -f "$pidfile"
done
