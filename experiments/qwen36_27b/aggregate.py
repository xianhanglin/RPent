#!/usr/bin/env python3
"""Summarise LIBERO-Pro episodes from each run.log.

Success is the simulator's verdict (RPent logs `recipe: <path>` only when LIBERO
reported task completion), not the planner's own `finish` status.

Usage:
    python aggregate.py runs/qwen3.6-27b_none_vllm
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# Leaderboard row "Qwen3.6 27B / no-reasoning" (RPent docs, leaderboard/performance.rst).
LEADERBOARD = {
    "libero_spatial_task": 82.0, "libero_spatial_swap": 78.0,
    "libero_object_task": 83.0, "libero_object_swap": 84.0,
    "libero_goal_task": 68.0, "libero_goal_swap": 68.0,
    "libero_10_task": 61.0, "libero_10_swap": 41.0,
}
LEADERBOARD_OVERALL = 70.63
LEADERBOARD_SEC_PER_EPISODE = 626.3
LEADERBOARD_OUT_TOKENS_PER_EPISODE = 3_270_754 / 800

RECIPE = re.compile(r"recipe: (.+)$", re.M)
ELAPSED = re.compile(r"elapsed: ([\d.]+)s")
USAGE = re.compile(r"usage: in=(\S+) out=(\S+) tool_calls=(\S+)")


def num(text: str) -> int | None:
    return int(text) if text.isdigit() else None


def parse(ep: Path) -> dict | None:
    status_file = ep / "episode_status.txt"
    log = ep / "run.log"
    if not status_file.exists() or not log.exists():
        return None
    text = log.read_text(errors="ignore")
    recipe = RECIPE.findall(text)
    usage = USAGE.findall(text)
    elapsed = ELAPSED.findall(text)
    return {
        "status": status_file.read_text().strip(),
        "success": bool(recipe) and "not written" not in recipe[-1],
        "elapsed_s": float(elapsed[-1]) if elapsed else None,
        "in_tok": num(usage[-1][0]) if usage else None,
        "out_tok": num(usage[-1][1]) if usage else None,
        "tool_calls": num(usage[-1][2]) if usage else None,
    }


def main() -> int:
    root = Path(sys.argv[1])
    rows = []
    for ep in sorted(root.glob("*/t*/s*")):
        if not re.fullmatch(r"s\d+", ep.name):
            continue
        rec = parse(ep)
        if rec is None:
            continue
        rec.update(suite=ep.parent.parent.name, task=int(ep.parent.name[1:]),
                   seed=int(ep.name[1:]))
        rows.append(rec)
    if not rows:
        print(f"no finished episodes under {root}")
        return 1

    print(f"{'suite':<22}{'task':>5}{'seed':>5}  {'result':<14}{'min':>7}"
          f"{'in_tok':>11}{'out_tok':>10}{'tools':>7}")
    for r in sorted(rows, key=lambda r: (r["suite"], r["task"], r["seed"])):
        result = "SUCCESS" if r["success"] else ("infra_error" if r["status"].startswith("infra")
                                                 else "fail")
        mins = f"{r['elapsed_s'] / 60:.1f}" if r["elapsed_s"] else "-"
        print(f"{r['suite']:<22}{r['task']:>5}{r['seed']:>5}  {result:<14}{mins:>7}"
              f"{r['in_tok'] or '-':>11}{r['out_tok'] or '-':>10}{r['tool_calls'] or '-':>7}")

    valid = [r for r in rows if not r["status"].startswith("infra")]
    print(f"\nepisodes: {len(rows)}  valid: {len(valid)}  infra errors: {len(rows) - len(valid)}")

    groups = defaultdict(list)
    for r in valid:
        groups[r["suite"]].append(r)
    for suite, rs in sorted(groups.items()):
        held_out = [r for r in rs if r["seed"] != 0]
        wins = sum(r["success"] for r in rs)
        line = f"{suite}: {wins}/{len(rs)} = {100 * wins / len(rs):.1f}%"
        if suite in LEADERBOARD:
            line += f"  (leaderboard {LEADERBOARD[suite]:.0f}% over 100)"
        if held_out and len(held_out) != len(rs):
            line += (f"  (seeds != 0: {sum(r['success'] for r in held_out)}/{len(held_out)};"
                     f" seed 0 is the memory-source seed)")
        print(line)
    wins = sum(r["success"] for r in valid)
    print(f"overall: {wins}/{len(valid)} = {100 * wins / len(valid):.2f}%"
          f"  (leaderboard {LEADERBOARD_OVERALL}% over 800)")

    timed = [r for r in valid if r["elapsed_s"]]
    tokened = [r for r in valid if r["out_tok"] is not None]
    if timed:
        print(f"mean time/episode: {sum(r['elapsed_s'] for r in timed) / len(timed):.1f} s"
              f"  (leaderboard {LEADERBOARD_SEC_PER_EPISODE} s)")
    if tokened:
        print(f"mean output tokens/episode: {sum(r['out_tok'] for r in tokened) / len(tokened):,.0f}"
              f"  (leaderboard {LEADERBOARD_OUT_TOKENS_PER_EPISODE:,.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
