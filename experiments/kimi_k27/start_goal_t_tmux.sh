#!/usr/bin/env bash
# Detached Goal-T sweep. Safe to re-run: finished episodes are skipped.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
SESSION=rpent-goal-t
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "source ./env.sh && python run_episodes.py --suite libero_goal_task --task 0 1 2 3 4 5 6 7 8 9 --seeds 0-9 --jobs 5 2>&1 | tee -a runs/sweep_libero_goal_task.log; echo EXIT=\$?; sleep 2"
echo "started tmux session $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log:    $PWD/runs/sweep_libero_goal_task.log"
