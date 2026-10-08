"""LoRA-tune Qwen3.6-27B with a pointer head on next-tool rows.

    cd experiments/planner_head   # from the rpent repo root
    python rows.py --runs ../qwen36_27b/runs --out data
    PYTHONPATH=/cfsdata/agent/kev uv run --project /cfsdata/agent/kev python train.py \
        --data data/train.jsonl --base /cfsdata/agent/models/Qwen3.6-27B --out runs/planner-head

The backbone stays frozen except the rank-16 adapter. The head is trained from scratch.
Loss is cross-entropy on the tool name. Run this after data/train.jsonl exists.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ToolDecisionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", default="/cfsdata/agent/models/Qwen3.6-27B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lora", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} has no rows; collect solved episodes and run rows.py first")
    return rows


def main():
    args = parse_args()
    rows = load_rows(args.data)
    model = ToolDecisionModel(args.base, args.device, lora_rank=args.lora, head_dim=args.head_dim)
    head_ids = {id(p) for p in model.head.parameters()}
    groups = [
        {"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": args.lr},
        {"params": list(model.head.parameters()), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(rows) / args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr], total_steps=max(steps, 1), pct_start=0.1)
    model.train()
    args.out.mkdir(parents=True, exist_ok=True)
    seen = 0
    opt.zero_grad()
    started = time.time()
    for epoch in range(args.epochs):
        order = torch.randperm(len(rows)).tolist()
        for begin in range(0, len(rows), args.accum):
            batch = [rows[i] for i in order[begin:begin + args.accum]]
            total = 0.0
            for row in batch:
                encoded = model.encode_row(row)
                with torch.autocast(args.device if args.device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    logits = model.logits(encoded)
                loss = F.cross_entropy(logits[None], torch.tensor([encoded["label"]], device=args.device))
                (loss / len(batch)).backward()
                total += float(loss.detach())
                seen += 1
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            print(f"epoch {epoch} seen {seen}/{args.epochs * len(rows)} loss {total / len(batch):.3f}", flush=True)
    model.lm.save_pretrained(args.out / "adapter")
    torch.save({
        "head": model.head.state_dict(),
        "temperature": model.head.temperature,
        "hidden_size": model.lm.config.hidden_size,
        "head_dim": args.head_dim,
        "lora": args.lora,
        "base": args.base,
    }, args.out / "head.pt")
    (args.out / "train_config.json").write_text(json.dumps({
        "data": str(args.data), "base": args.base, "lora": args.lora, "lr": args.lr,
        "epochs": args.epochs, "accum": args.accum, "rows": len(rows), "seconds": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
