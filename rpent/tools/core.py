# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared argument and result conventions for the core robot tools.

Robots that publish ``move_to``, ``rotate_wrist``, ``set_gripper``, ``release``,
or ``view_env_state`` use the argument types and result builders defined here,
so the same call means the same thing on every robot:

- positions are world-frame meters, angles are radians, and quaternions are
  ordered ``[w, x, y, z]``;
- yaw is the heading of the end-effector X axis in the world XY plane, so
  ``delta_yaw`` is comparable across robots even where absolute yaw is not;
- a gripper opening is a fraction: ``0`` is closed and ``1`` is fully open;
- a motion's gripper argument is ``"hold"``, ``"open"``, or ``"close"`` and
  defaults to ``"hold"``, which keeps a grasped object grasped;
- ``arm`` is published only by robots with more than one arm.

Robots may add optional arguments and result keys for their own controllers.
:func:`migrate_legacy_call` translates calls recorded before these conventions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from rpent.tools.base import ToolResult

if TYPE_CHECKING:
    from rpent.session import EnvState

Arm = Literal["left", "right"]
GripperMode = Literal["hold", "open", "close"]
GripperTarget = Literal["open", "close"]
WorldXYZ = Annotated[list[float], Field(min_length=3, max_length=3)]
StopReason = Literal["reached", "not_reached", "plan_failed", "terminated", "truncated"]

MOTION_RESULT_KEYS = (
    "ok",
    "stop_reason",
    "steps_used",
    "eef_xyz",
    "gripper_opening",
    "terminated",
    "truncated",
)


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to ``[-pi, pi)``."""
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


def stop_reason(*, reached: bool, terminated: bool, truncated: bool) -> StopReason:
    """Report why a motion stopped; episode endings take precedence."""
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "reached" if reached else "not_reached"


def motion_result(
    *,
    ok: bool,
    stop_reason: StopReason,
    steps_used: int,
    eef_xyz: Sequence[float],
    gripper_opening: float,
    terminated: bool,
    truncated: bool,
    dist_m: float | None = None,
    yaw_error: float | None = None,
    **extra: Any,
) -> ToolResult:
    """Build the result shared by motion and gripper tools.

    Args:
        ok: Whether the commanded goal was reached.
        stop_reason: Why the motion stopped.
        steps_used: Native environment steps executed.
        eef_xyz: Final world-frame end-effector position, meters.
        gripper_opening: Final opening fraction, clipped to ``[0, 1]``.
        terminated: Native episode termination flag.
        truncated: Native episode truncation flag.
        dist_m: Final distance to a position target, meters.
        yaw_error: Final wrapped yaw error to a yaw target, radians.
        **extra: Robot-specific keys.
    """
    data: dict[str, Any] = {
        "ok": bool(ok),
        "stop_reason": stop_reason,
        "steps_used": int(steps_used),
        "eef_xyz": [round(float(value), 4) for value in eef_xyz],
        "gripper_opening": round(min(max(float(gripper_opening), 0.0), 1.0), 3),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if dist_m is not None:
        data["dist_m"] = round(float(dist_m), 4)
    if yaw_error is not None:
        data["yaw_error"] = round(float(yaw_error), 4)
    data.update(extra)
    return ToolResult(data=data)


@dataclass(frozen=True)
class CameraView:
    """One image that ``view_env_state`` embeds, with its first present artifact."""

    camera: str
    role: str
    artifacts: tuple[str, ...]


def observation_result(
    state: EnvState,
    step: int,
    views: Sequence[CameraView],
    *,
    extra_keys: Sequence[str] = (),
) -> ToolResult:
    """Read one recorded step in the shared ``view_env_state`` shape.

    Images follow ``views`` order, and ``images`` names the camera of each one.

    Args:
        state: Recorded environment states for this run.
        step: Step number; ``-1`` selects the latest entry.
        views: Cameras to embed, in order.
        extra_keys: Robot-specific ``record.extras`` keys to include.
    """
    try:
        record = state.get(step)
    except Exception as exc:
        return ToolResult(data={"error": f"state step not available: {exc}"})
    extras = record.extras
    data: dict[str, Any] = {
        "step": record.step_idx,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "success": extras.get("success"),
        "task_language": extras.get("task_language"),
        "state": record.state,
        "artifacts": sorted(record.artifacts),
        "log": {
            "command": record.command,
            "result": record.result,
            "elapsed_s": record.elapsed_s,
        },
        "images": [],
    }
    images: list[bytes] = []
    for view in views:
        for name in view.artifacts:
            if name not in record.artifacts:
                continue
            try:
                images.append(state.load_bytes(name, step=record.step_idx))
            except FileNotFoundError:
                continue
            data["images"].append(
                {"camera": view.camera, "role": view.role, "artifact": name}
            )
            break
    for key in extra_keys:
        data[key] = extras.get(key)
    return ToolResult(data=data, images=images)


# ---------------------------------------------------------------------------
# Legacy calls
# ---------------------------------------------------------------------------

LEGACY_ROBOTS = ("libero", "robotwin", "robocasa")
#: Written into Flash plans generated with the shared core conventions.
TOOL_SPEC_VERSION = 1

# Gripper value each legacy tool used when the call omitted it.
_LEGACY_GRIPPER_DEFAULTS = {
    "libero": {
        "move_to": -1.0,
        "move_pose": -1.0,
        "rotate_wrist": 1.0,
        "rotate_pitch": 1.0,
    },
    "robocasa": {
        "move_to": "hold",
        "move_delta": "hold",
        "move_base": "hold",
        "navigate_to": "hold",
        "rotate_pitch": 1.0,
    },
}


def _signed_gripper(value: Any) -> str:
    """Translate a legacy ``-1 open / +1 close / 0 hold`` command."""
    if isinstance(value, str):
        return value
    value = float(value)
    if value > 0:
        return "close"
    if value < 0:
        return "open"
    return "hold"


def _robotwin_gripper(value: Any) -> str:
    """Translate a legacy RoboTwin target where ``0`` is closed and ``1`` open."""
    if value is None:
        return "hold"
    if isinstance(value, str):
        return value
    return "open" if float(value) >= 0.5 else "close"


def migrate_legacy_call(
    robot: str, name: str, arguments: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Translate one tool call recorded before the shared core conventions.

    The input must be a legacy call: an omitted gripper is read as that tool's
    legacy default. Tools outside the core set pass through unchanged, apart
    from their gripper argument.

    Args:
        robot: ``"libero"``, ``"robotwin"``, or ``"robocasa"``.
        name: Recorded tool name.
        arguments: Recorded tool arguments.

    Returns:
        The tool name and arguments to call with the current tools.
    """
    if robot not in LEGACY_ROBOTS:
        raise ValueError(f"no legacy tool conventions for robot {robot!r}")
    args = dict(arguments)
    if robot == "robotwin":
        if "gripper" in args or name in {"move_to", "rotate_wrist"}:
            args["gripper"] = _robotwin_gripper(args.get("gripper"))
        if name == "rotate_wrist" and "delta_yaw_deg" in args:
            args["delta_yaw"] = math.radians(float(args.pop("delta_yaw_deg")))
        if name == "set_gripper" and "val" in args:
            value = float(args.pop("val"))
            args["command"] = (
                "open" if value == 1.0 else "close" if value == 0.0 else value
            )
        if name == "release" and "val" in args:
            value = float(args.pop("val"))
            if value != 1.0:
                return "set_gripper", {**args, "command": value}
        return name, args

    defaults = _LEGACY_GRIPPER_DEFAULTS[robot]
    if name in defaults:
        args["gripper"] = _signed_gripper(args.get("gripper", defaults[name]))
    elif name == "set_gripper":
        legacy_default = -1.0 if robot == "libero" else 1.0
        args["command"] = _signed_gripper(args.pop("gripper", legacy_default))
        if args["command"] == "hold":
            raise ValueError("legacy set_gripper with gripper=0 has no equivalent")
    elif name == "release" and robot == "libero" and "max_steps" in args:
        args["steps"] = args.pop("max_steps")
    elif name == "view_env_state" and args.get("step", -1) is None:
        args["step"] = -1
    return name, args


def migrate_legacy_command(robot: str, command: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a recipe line, ``{"action": name, **arguments}``."""
    arguments = {key: value for key, value in command.items() if key != "action"}
    name, arguments = migrate_legacy_call(robot, str(command["action"]), arguments)
    return {"action": name, **arguments}


def migrate_legacy_plan(robot: str, plan: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Translate Flash plan entries, ``{"action": name, "arguments": {...}, ...}``."""
    migrated = []
    for entry in plan:
        name, arguments = migrate_legacy_call(
            robot, str(entry["action"]), entry.get("arguments", {})
        )
        migrated.append({**entry, "action": name, "arguments": arguments})
    return migrated


def _migrate_file(robot: str, source: Path, destination: Path) -> None:
    if source.suffix == ".jsonl":
        lines = [
            json.dumps(migrate_legacy_command(robot, json.loads(line)))
            for line in source.read_text().splitlines()
            if line.strip()
        ]
        destination.write_text("".join(line + "\n" for line in lines))
        return
    document = json.loads(source.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("plan"), list):
        raise ValueError(f"{source} is neither a recipe JSONL nor a Flash plan")
    if document.get("tool_spec") != TOOL_SPEC_VERSION:
        document["plan"] = migrate_legacy_plan(robot, document["plan"])
        document["tool_spec"] = TOOL_SPEC_VERSION
    destination.write_text(json.dumps(document, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite legacy recipe JSONL files and Flash plans with current tool calls."""
    parser = argparse.ArgumentParser(
        prog="python -m rpent.tools.core",
        description=(
            "Translate recipe JSONL files and Flash *_plan.json files recorded "
            "before the shared core tool conventions."
        ),
    )
    parser.add_argument("--robot", required=True, choices=LEGACY_ROBOTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write translated copies here instead of replacing the inputs.",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.paths:
        destination = path if args.output_dir is None else args.output_dir / path.name
        _migrate_file(args.robot, path, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
