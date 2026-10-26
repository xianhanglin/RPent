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

"""Shared core tool results and translation of legacy tool calls."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rpent.session import EnvState
from rpent.tools.core import (
    TOOL_SPEC_VERSION,
    CameraView,
    main,
    migrate_legacy_call,
    motion_result,
    observation_result,
    stop_reason,
)


def _motion(**overrides):
    fields = {
        "ok": True,
        "stop_reason": "reached",
        "steps_used": 3,
        "eef_xyz": [0.1, 0.2, 0.3],
        "gripper_opening": 0.5,
        "terminated": False,
        "truncated": False,
    }
    return motion_result(**{**fields, **overrides})


def test_motion_result_clips_opening_and_keeps_robot_keys():
    result = _motion(gripper_opening=1.4, dist_m=0.01234, plan_status="Success")

    assert result.data["gripper_opening"] == 1.0
    assert result.data["dist_m"] == 0.0123
    assert result.data["plan_status"] == "Success"
    assert "yaw_error" not in result.data


@pytest.mark.parametrize(
    ("reached", "terminated", "truncated", "expected"),
    [
        (True, False, False, "reached"),
        (False, False, False, "not_reached"),
        (True, True, False, "terminated"),
        (False, False, True, "truncated"),
    ],
)
def test_episode_endings_take_precedence(reached, terminated, truncated, expected):
    assert (
        stop_reason(reached=reached, terminated=terminated, truncated=truncated)
        == expected
    )


def test_observation_result_names_each_embedded_image(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={"q": 1},
        terminated=True,
        truncated=False,
        command={"action": "release"},
        result={"ok": True},
        elapsed_s=0.5,
        extras={"task_language": "put it away", "success": True, "vla_desync": True},
    ) as step:
        state.save("wrist.png", np.zeros((2, 2, 3), dtype=np.uint8), step=step)

    result = observation_result(
        state,
        -1,
        (
            CameraView("agentview", "calibration_frame", ("agentview.png",)),
            CameraView("wrist", "calibration_frame", ("wrist_high.png", "wrist.png")),
        ),
        extra_keys=("vla_desync",),
    )

    assert result.images == [state.load_bytes("wrist.png", step=0)]
    assert result.data["images"] == [
        {"camera": "wrist", "role": "calibration_frame", "artifact": "wrist.png"}
    ]
    assert result.data["terminated"] is True
    assert result.data["success"] is True
    assert result.data["vla_desync"] is True
    assert result.data["log"]["command"] == {"action": "release"}


def test_observation_result_reports_a_missing_step(tmp_path: Path):
    result = observation_result(EnvState(tmp_path), 4, ())
    assert "state step not available" in result.data["error"]


@pytest.mark.parametrize(
    ("robot", "name", "arguments", "expected"),
    [
        # Omitted grippers take the legacy default of each tool.
        (
            "libero",
            "move_to",
            {"xyz": [0, 0, 1]},
            ("move_to", {"xyz": [0, 0, 1], "gripper": "open"}),
        ),
        (
            "libero",
            "move_to",
            {"xyz": [0, 0, 1], "gripper": 1},
            ("move_to", {"xyz": [0, 0, 1], "gripper": "close"}),
        ),
        (
            "libero",
            "rotate_wrist",
            {"delta_yaw": 0.3},
            ("rotate_wrist", {"delta_yaw": 0.3, "gripper": "close"}),
        ),
        (
            "libero",
            "set_gripper",
            {"steps": 5},
            ("set_gripper", {"steps": 5, "command": "open"}),
        ),
        ("libero", "release", {"max_steps": 20}, ("release", {"steps": 20})),
        (
            "libero",
            "pi0_pick",
            {"prompt": "pick up the mug"},
            ("pi0_pick", {"prompt": "pick up the mug"}),
        ),
        (
            "robocasa",
            "move_to",
            {"xyz": [0, 0, 1]},
            ("move_to", {"xyz": [0, 0, 1], "gripper": "hold"}),
        ),
        (
            "robocasa",
            "move_delta",
            {"dxyz": [0, 0, 1], "gripper": -1},
            ("move_delta", {"dxyz": [0, 0, 1], "gripper": "open"}),
        ),
        ("robocasa", "set_gripper", {}, ("set_gripper", {"command": "close"})),
        (
            "robocasa",
            "view_env_state",
            {"step": None},
            ("view_env_state", {"step": -1}),
        ),
        (
            "robotwin",
            "move_to",
            {"arm": "left", "xyz": [0, 0, 1]},
            ("move_to", {"arm": "left", "xyz": [0, 0, 1], "gripper": "hold"}),
        ),
        (
            "robotwin",
            "move_to",
            {"arm": "left", "xyz": [0, 0, 1], "gripper": 0.0},
            ("move_to", {"arm": "left", "xyz": [0, 0, 1], "gripper": "close"}),
        ),
        (
            "robotwin",
            "set_gripper",
            {"arm": "right", "val": 1.0},
            ("set_gripper", {"arm": "right", "command": "open"}),
        ),
        (
            "robotwin",
            "set_gripper",
            {"arm": "right", "val": 0.4},
            ("set_gripper", {"arm": "right", "command": 0.4}),
        ),
        (
            "robotwin",
            "release",
            {"arm": "right", "val": 1.0, "steps": 5},
            ("release", {"arm": "right", "steps": 5}),
        ),
        (
            "robotwin",
            "release",
            {"arm": "right", "val": 0.6},
            ("set_gripper", {"arm": "right", "command": 0.6}),
        ),
    ],
)
def test_legacy_calls_keep_their_recorded_behavior(robot, name, arguments, expected):
    assert migrate_legacy_call(robot, name, arguments) == expected


def test_legacy_robotwin_yaw_degrees_become_radians():
    name, arguments = migrate_legacy_call(
        "robotwin", "rotate_wrist", {"arm": "left", "delta_yaw_deg": 90}
    )
    assert name == "rotate_wrist"
    assert arguments["delta_yaw"] == pytest.approx(math.pi / 2)
    assert "delta_yaw_deg" not in arguments
    assert arguments["gripper"] == "hold"


def test_legacy_set_gripper_hold_has_no_translation():
    with pytest.raises(ValueError, match="no equivalent"):
        migrate_legacy_call("libero", "set_gripper", {"gripper": 0})


def test_unknown_robot_is_rejected():
    with pytest.raises(ValueError, match="franka"):
        migrate_legacy_call("franka", "move_to", {})


def test_cli_rewrites_recipes_and_translates_flash_plans_once(tmp_path: Path):
    recipe = tmp_path / "t0_recipe.jsonl"
    recipe.write_text(
        json.dumps({"action": "move_to", "xyz": [0, 0, 1]})
        + "\n"
        + json.dumps({"action": "release", "max_steps": 20})
        + "\n"
    )
    plan = tmp_path / "t0_plan.json"
    plan.write_text(
        json.dumps({"plan": [{"action": "set_gripper", "arguments": {"gripper": 1}}]})
    )
    output = tmp_path / "out"

    assert (
        main(["--robot", "libero", "--output-dir", str(output), str(recipe), str(plan)])
        == 0
    )

    lines = [
        json.loads(line) for line in (output / recipe.name).read_text().splitlines()
    ]
    assert lines == [
        {"action": "move_to", "xyz": [0, 0, 1], "gripper": "open"},
        {"action": "release", "steps": 20},
    ]
    migrated = json.loads((output / plan.name).read_text())
    assert migrated["tool_spec"] == TOOL_SPEC_VERSION
    assert migrated["plan"][0]["arguments"] == {"command": "close"}
    assert json.loads(recipe.read_text().splitlines()[1]) == {
        "action": "release",
        "max_steps": 20,
    }

    main(["--robot", "libero", str(output / plan.name)])
    assert json.loads((output / plan.name).read_text()) == migrated
