"""Turn solved Harness VLA episodes into Kev labelled-request rows.

    python rows.py --runs ../qwen36_27b/runs --out data

One row is one tool decision. The state is the task, the pose before the call, and the
calls already made. The label is the next tool name. Tasks 8 and 9 are written to
heldout.jsonl in every suite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Non-readonly LIBERO primitives that advance the episode, plus segment, which the
# recipe writer records from its artifact. reset is a boundary, not a training label.
TOOLS = {
    "segment": "Segment an object in the camera image.",
    "pi0_pick": "Run Pi0.5 to pick up an object.",
    "pi0_doubled": "Run Pi0.5 for a contact skill other than a pick.",
    "move_to": "Servo the end effector to a world-frame position.",
    "move_pose": "Servo the end effector to a position and orientation.",
    "rotate_wrist": "Set the wrist yaw.",
    "rotate_pitch": "Set the wrist pitch.",
    "set_gripper": "Open or close the gripper.",
    "release": "Open the gripper and release the object.",
}
PRIMITIVES = tuple(name for name in TOOLS if name != "segment")
HOLDOUT_TASKS = (8, 9)


def episode_solved(episode: Path) -> bool:
    status = episode / "episode_status.txt"
    if not status.exists():
        return False
    return status.read_text(encoding="utf-8").splitlines()[0].strip() == "ok"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def commands_of(episode: Path) -> list[tuple[int, dict]]:
    """(step_idx, command) in recipe order. Empty when the environment did not solve it."""
    steps = _load_json(episode / "states.json")["steps"]
    last_reset = max(
        (
            step["step_idx"]
            for step in steps
            if (step.get("command") or {}).get("action") == "reset"
            and not (isinstance(step.get("result"), dict) and step["result"].get("error"))
        ),
        default=-1,
    )
    events: list[tuple[tuple[int, int], int, dict]] = []
    for step in steps:
        if step["step_idx"] <= last_reset:
            continue
        command = step.get("command")
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if command and command.get("action") in PRIMITIVES and not result.get("error"):
            events.append(((step["step_idx"], -1), step["step_idx"], dict(command)))
        for name in sorted(step.get("artifacts") or []):
            if not (name.startswith("segment_") and name.endswith(".json")):
                continue
            path = episode / name / f"{step['step_idx']:02d}.json"
            if not path.exists():
                continue
            segment = _load_json(path)
            if segment.get("error"):
                continue
            if segment.get("mode") == "text":
                seg = {"action": "segment", "prompt": segment["prompt"], "camera": segment["camera"]}
            else:
                seg = {"action": "segment", "point": segment["point"], "camera": segment["camera"]}
            events.append(((step["step_idx"], int(segment["segment_index"])), step["step_idx"], seg))
    solved = any(step.get("terminated") and step["step_idx"] > last_reset for step in steps)
    if not solved:
        return []
    events.sort(key=lambda event: event[0])
    unknown = sorted({cmd["action"] for _, _, cmd in events if cmd["action"] not in TOOLS})
    if unknown:
        raise ValueError(f"{episode}: actions outside the tool list: {unknown}")
    return [(step_idx, cmd) for _, step_idx, cmd in events]


def _pose_record(steps: list[dict], event_step: int) -> dict:
    earlier = [step for step in steps if step["step_idx"] < event_step]
    return earlier[-1] if earlier else next(step for step in steps if step["step_idx"] == event_step)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return "[" + ", ".join(_fmt(item) for item in value) + "]"
    return str(value)


def _call_text(command: dict) -> str:
    fields = " ".join(f"{key}={_fmt(command[key])}" for key in command if key != "action")
    return command["action"] if not fields else f"{command['action']} {fields}"


def _state_text(task: str, pose: dict, history: list[dict]) -> str:
    pos = pose["state"]["robot0_eef_pos"]
    grip = pose["state"]["robot0_gripper_qpos"][0]
    names = ", ".join(pose["state"].get("object_names") or [])
    calls = "\n".join(f"{i}. {_call_text(cmd)}" for i, cmd in enumerate(history, start=1)) or "(none)"
    return (
        f"Task: {task}\n"
        f"Objects: {names}\n"
        f"End effector xyz: {_fmt(pos[0])}, {_fmt(pos[1])}, {_fmt(pos[2])}\n"
        f"Gripper opening: {_fmt(grip)}\n"
        f"Calls so far:\n{calls}"
    )


def rows_of_episode(episode: Path, holdout_tasks: tuple[int, ...] = HOLDOUT_TASKS) -> list[dict]:
    if not episode_solved(episode):
        return []
    steps = _load_json(episode / "states.json")["steps"]
    task = next(
        (step.get("extras") or {}).get("task_language")
        for step in steps
        if (step.get("extras") or {}).get("task_language")
    )
    suite = episode.parent.parent.name
    task_id = int(episode.parent.name.removeprefix("t"))
    seed = int(episode.name.removeprefix("s"))
    split = "heldout" if task_id in holdout_tasks else "train"
    history: list[dict] = []
    rows = []
    for index, (step_idx, command) in enumerate(commands_of(episode)):
        rows.append({
            "state": _state_text(task, _pose_record(steps, step_idx), history),
            "questions": {
                "next_tool": {
                    "type": "choice",
                    "instructions": "Which tool runs next?",
                    "criteria": dict(TOOLS),
                    "label": command["action"],
                }
            },
            "_meta": {
                "suite": suite,
                "task": task_id,
                "seed": seed,
                "decision": index,
                "split": split,
                "episode": f"{suite}/t{task_id}/s{seed}",
                "source": "harness",
            },
        })
        history.append(command)
    return rows


def collect(run_roots: list[Path], holdout_tasks: tuple[int, ...] = HOLDOUT_TASKS) -> tuple[list[dict], list[dict]]:
    train, heldout = [], []
    for root in run_roots:
        for status in sorted(root.rglob("episode_status.txt")):
            for row in rows_of_episode(status.parent, holdout_tasks):
                (heldout if row["_meta"]["split"] == "heldout" else train).append(row)
    return train, heldout


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-tasks", default="8,9")
    args = parser.parse_args()
    holdout = tuple(int(part) for part in args.holdout_tasks.split(",") if part)
    train, heldout = collect(args.runs, holdout)
    write_split(train, args.out / "train.jsonl")
    write_split(heldout, args.out / "heldout.jsonl")
    print(f"train {len(train)} rows, heldout {len(heldout)} rows -> {args.out}")


if __name__ == "__main__":
    main()
