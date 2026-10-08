# Qwen3.6-27B LIBERO-Pro validation

Compares a local Harness VLA run against the leaderboard row **Qwen3.6 27B / no-reasoning** (70.63%, 565/800).

Run the **whole** stack on **VM-0-14** (`10.9.0.14`). VM-0-12 is the Kimi eval node; VM-0-8 serves Kimi.

Weights: `/cfsdata/agent/models/Qwen3.6-27B` (revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`).

```bash
# On VM-0-14
cd experiments/qwen36_27b   # from the rpent repo root

# 1. Pi0.5 (GPU 0) and SAM3 (GPU 1)
bash start_services.sh

# 2. Qwen vLLM (GPUs 2,3), thinking off
tmux new -s qwen36
bash serve_qwen.sh
# wait for: Application startup complete.

# 3. Planner check, then resume Goal Task (skips finished cells)
tmux new -s qwen36eval
source env.sh
bash check.sh
python run_episodes.py --suite libero_goal_task --task 0 1 2 3 4 5 6 7 8 9 --seeds 0-9 --jobs 3
python aggregate.py runs/qwen3.6-27b_none_vllm
```
