#!/usr/bin/env python3
"""Run LIBERO-Pro episodes with the Qwen3.6-27B planner against the shared VLA/SAM3 services.

Each (suite, task, seed) episode gets its own output directory. Finished episodes are
skipped, so the script can be re-run after a crash. Episodes that died because of the
LLM API or infrastructure (rate limits, 5xx, connection errors) are retried; episodes
where the planner simply failed the task are never retried.

Usage (after `source env.sh`):
    python run_episodes.py --suite libero_goal_task --task 1 --seeds 0-9 --jobs 5
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INFRA_ERROR = re.compile(
    r"rate.?limit|\b429\b|overloaded|\b5\d\d\b.*(error|status)|"
    r"APIConnectionError|ConnectionError|ConnectError|Connection refused|"
    r"ReadTimeout|RemoteProtocolError|authentication|invalid.?api.?key|401|"
    r"insufficient.?(balance|quota)|RPC.*(unreachable|failed)",
    re.IGNORECASE,
)


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def episode_dir(root: Path, suite: str, task: int, seed: int) -> Path:
    return root / suite / f"t{task}" / f"s{seed}"


def is_finished(out: Path) -> bool:
    status = out / "episode_status.txt"
    return status.exists() and status.read_text().strip() in {"ok", "planner_error"}


def classify(returncode: int, out: Path) -> str:
    if returncode == 0:
        return "ok"
    log = out / "run.log"
    text = log.read_text(errors="ignore")[-200_000:] if log.exists() else ""
    if not text or "elapsed:" not in text:
        return "infra_error"
    error_lines = "\n".join(
        line for line in text.splitlines() if "ERROR" in line or "EXCEPTION" in line
    )
    return "infra_error" if INFRA_ERROR.search(error_lines) else "planner_error"


def build_command(args: argparse.Namespace, suite: str, task: int, seed: int,
                  gpu: str, out: Path) -> list[str]:
    env = os.environ
    return [
        "rpent", "--robot", "libero", "--libero-type", "pro",
        "--suite", suite, "--task", str(task), "--seed", str(seed),
        "--planner", "claude_code", "--model", env["MODEL"],
        "--reasoning-effort", env["REASONING_EFFORT"],
        "--max-turns", str(args.max_turns),
        "--planner-timeout-s", str(args.planner_timeout_s),
        "--max-episode-steps", "10000",
        # Claude Code prices unknown model ids with Claude rates; keep its budget cap out of the way.
        "--claude-code-max-budget-usd", "1000",
        "--memory-profile", "local", "--memory-dir", env["MEMORY_DIR"],
        "--vla-endpoint", f"http://127.0.0.1:{env['VLA_PORT']}",
        "--sam3-endpoint", f"http://127.0.0.1:{env['SAM3_PORT']}",
        "--cuda-device", gpu,
        "--output-dir", str(out),
    ]


def run_one(args, suite, task, seed, gpu_pool, lock) -> tuple[str, str]:
    root = Path(os.environ["EXP_ROOT"]) / "runs" / os.environ["RUN_TAG"]
    out = episode_dir(root, suite, task, seed)
    name = f"{suite} t{task} s{seed}"
    if is_finished(out):
        return name, "skipped (done)"
    for attempt in range(1, args.infra_retries + 2):
        if out.exists():
            archive = out.with_name(f"{out.name}.failed{attempt - 1}")
            shutil.rmtree(archive, ignore_errors=True)
            out.rename(archive)
        out.mkdir(parents=True)
        with lock:
            gpu = next(gpu_pool)
        cmd = build_command(args, suite, task, seed, gpu, out)
        (out / "command.txt").write_text(" ".join(cmd) + "\n")
        with open(out / "stdout.log", "w") as stdout:
            proc = subprocess.run(cmd, cwd=os.environ["RPENT_ROOT"], stdout=stdout,
                                  stderr=subprocess.STDOUT)
        status = classify(proc.returncode, out)
        (out / "episode_status.txt").write_text(status + "\n")
        if status != "infra_error":
            return name, status
    return name, f"infra_error after {args.infra_retries + 1} attempts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, nargs="+")
    ap.add_argument("--task", required=True, nargs="+", type=int)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--planner-timeout-s", type=int, default=5000)
    ap.add_argument("--infra-retries", type=int, default=2)
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_BASE_URL"):
        sys.exit("ANTHROPIC_BASE_URL is not set; `source env.sh` first")

    gpu_pool = itertools.cycle(os.environ["ENV_GPUS"].split())
    lock = threading.Lock()
    episodes = [(s, t, seed) for s in args.suite for t in args.task
                for seed in parse_seeds(args.seeds)]
    print(f"{len(episodes)} episodes, {args.jobs} in parallel, tag {os.environ['RUN_TAG']}")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, args, *ep, gpu_pool, lock) for ep in episodes]
        for fut in as_completed(futures):
            name, status = fut.result()
            print(f"{name}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
