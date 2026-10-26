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

"""The core tools mean the same thing on LIBERO, RoboTwin, and RoboCasa."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.libero import tools as libero_tools
from robots.libero.tools import LiberoPrimitives
from robots.robocasa import tools as robocasa_tools
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robotwin import tools as robotwin_tools
from robots.robotwin.primitives import RoboTwinPrimitives
from rpent.tools import iter_tools
from rpent.tools.core import MOTION_RESULT_KEYS

DECLARATIONS = {
    "libero": {t.name: t for t in iter_tools(libero_tools, LiberoPrimitives)},
    "robotwin": {t.name: t for t in iter_tools(robotwin_tools, RoboTwinPrimitives)},
    "robocasa": {t.name: t for t in iter_tools(robocasa_tools, RoboCasaPrimitives)},
}


def _properties(robot: str, name: str) -> dict:
    return DECLARATIONS[robot][name].input_schema["properties"]


@pytest.mark.parametrize("robot", sorted(DECLARATIONS))
def test_core_arguments_share_names_types_and_defaults(robot):
    move = _properties(robot, "move_to")
    assert move["xyz"]["minItems"] == move["xyz"]["maxItems"] == 3
    assert move["gripper"]["enum"] == ["hold", "open", "close"]
    assert (
        DECLARATIONS[robot]["move_to"].args_schema.model_fields["gripper"].default
        == "hold"
    )

    set_gripper = DECLARATIONS[robot]["set_gripper"].input_schema
    assert "command" in set_gripper["required"]
    release = _properties(robot, "release")
    assert set(release) - {"arm"} == {"steps"}

    assert set(_properties(robot, "view_env_state")) == {"step"}
    assert _properties(robot, "view_env_state")["step"]["type"] == "integer"

    if "rotate_wrist" in DECLARATIONS[robot]:
        rotate = _properties(robot, "rotate_wrist")
        assert {"delta_yaw", "target_yaw"} <= set(rotate)
        assert rotate["gripper"]["enum"] == ["hold", "open", "close"]


@pytest.mark.parametrize("robot", sorted(DECLARATIONS))
def test_only_multi_arm_robots_publish_arm(robot):
    has_arm = "arm" in _properties(robot, "move_to")
    assert has_arm is (robot == "robotwin")


@pytest.mark.parametrize("robot", sorted(DECLARATIONS))
def test_legacy_gripper_values_are_rejected(robot):
    schema = DECLARATIONS[robot]["move_to"].args_schema
    arguments = {"xyz": [0.0, 0.0, 0.0], "gripper": 1}
    if robot == "robotwin":
        arguments["arm"] = "left"
    with pytest.raises(ValueError, match="hold") as error:
        schema.model_validate(arguments, strict=True)
    assert error.value.error_count() == 1


# ---------------------------------------------------------------------------
# LIBERO: native -1 open, +1 close, 0 keeps the current command.
# ---------------------------------------------------------------------------


class LiberoEnv:
    def __init__(self):
        self.xyz = np.zeros(3)
        self.yaw = 0.0
        self.fingers = 0.04
        self.actions: list[np.ndarray] = []
        self.terminated = False
        self.truncated = False

    def observation(self):
        states = np.zeros(8)
        states[:3] = self.xyz
        states[6:8] = (self.fingers / 2, -self.fingers / 2)
        return {"states": states}

    def raw_obs(self):
        rotation = Rotation.from_euler("z", self.yaw) * Rotation.from_euler("x", np.pi)
        return {"robot0_eef_quat": rotation.as_quat()}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.xyz += action[:3] * 0.05
        self.yaw += float(action[5]) * 0.5
        self.fingers = float(np.clip(self.fingers - action[6] * 0.01, 0.0, 0.08))
        return self.observation(), 0, self.terminated, self.truncated, {}


def _libero(env: LiberoEnv) -> LiberoPrimitives:
    driver = LiberoPrimitives(
        env=env, model=None, sam3_client=None, check_cancelled=lambda: None
    )
    driver.set_obs(env.observation())
    return driver


@pytest.mark.parametrize(
    ("gripper", "native"), [("hold", 0.0), ("open", -1.0), ("close", 1.0)]
)
def test_libero_motion_gripper_modes(gripper, native):
    env = LiberoEnv()
    result = _libero(env).move_to([0.05, 0, 0], gripper=gripper).data

    assert {action[6] for action in env.actions} == {native}
    assert set(MOTION_RESULT_KEYS) <= set(result)
    assert result["ok"] is True
    assert result["stop_reason"] == "reached"
    assert result["steps_used"] == len(env.actions)


def test_libero_move_to_reports_an_unreached_target():
    env = LiberoEnv()
    result = _libero(env).move_to([0.5, 0, 0], max_steps=2).data

    assert result["ok"] is False
    assert result["stop_reason"] == "not_reached"
    assert result["dist_m"] > 0.4


def test_libero_gripper_tools_report_opening_fraction():
    env = LiberoEnv()
    driver = _libero(env)

    closed = driver.set_gripper(command="close", steps=4).data
    opened = driver.release(steps=8).data

    assert [action[6] for action in env.actions] == [1.0] * 4 + [-1.0] * 8
    assert closed["ok"] is True
    assert closed["gripper_opening"] == pytest.approx(0.0)
    assert opened["gripper_opening"] == pytest.approx(1.0)


def test_libero_rotate_wrist_holds_the_gripper_by_default():
    env = LiberoEnv()
    result = _libero(env).rotate_wrist(delta_yaw=0.2).data

    assert {action[6] for action in env.actions} == {0.0}
    assert result["ok"] is True
    assert abs(result["yaw_error"]) < 0.02


# ---------------------------------------------------------------------------
# RoboTwin: native gripper target 0 closed, 1 open.
# ---------------------------------------------------------------------------


class RoboTwinEnv:
    def __init__(self):
        down = [0.0, 1.0, 0.0, 0.0]  # wxyz, gripper pointing down
        self.planned: dict[str, np.ndarray] = {}
        self.terminated = False
        self.truncated = False
        self.qpos_actions: list[np.ndarray] = []
        self.plan_status = "Success"
        self.reach = True
        self.last_info = {
            "robot_state": {
                "left_eef_pose": np.asarray([0.1, 0.2, 0.3, *down]),
                "right_eef_pose": np.asarray([0.1, -0.2, 0.3, *down]),
                "left_gripper": 0.5,
                "right_gripper": 0.5,
                "qpos_target14": np.zeros(14),
            },
            "episode_status": {
                "eval_success": False,
                "take_action_cnt": 0,
                "step_lim": 1000,
            },
        }

    def plan_arm_path(self, arm, target):
        self.planned[arm] = np.asarray(target, dtype=np.float64)
        return {"status": self.plan_status, "position": [[0.0] * 6] * 3}

    def step(self, action, action_type):
        assert action_type == "qpos"
        self.qpos_actions.append(np.asarray(action).copy())
        state = self.last_info["robot_state"]
        state["left_gripper"] = float(action[6])
        state["right_gripper"] = float(action[13])
        for arm, target in self.planned.items():
            if self.reach:
                state[f"{arm}_eef_pose"] = target.copy()
        return (
            {},
            0,
            False,
            False,
            {
                "executed_actions": 1,
                "episode_status": self.last_info["episode_status"],
            },
        )


def _robotwin(env: RoboTwinEnv) -> RoboTwinPrimitives:
    return RoboTwinPrimitives(env=env, model=None, seed=0, check_cancelled=lambda: None)


@pytest.mark.parametrize(
    ("gripper", "native"), [("hold", 0.5), ("open", 1.0), ("close", 0.0)]
)
def test_robotwin_motion_gripper_modes(gripper, native):
    env = RoboTwinEnv()
    result = (
        _robotwin(env).move_to(arm="left", xyz=[0.2, 0.2, 0.3], gripper=gripper).data
    )

    # qpos_target14 is all zeros, so "hold" leaves the composed target untouched
    # and the fake reads it back as the gripper value.
    expected = 0.0 if gripper == "hold" else native
    assert {float(action[6]) for action in env.qpos_actions} == {expected}
    assert set(MOTION_RESULT_KEYS) <= set(result)
    assert result["ok"] is True
    assert result["eef_xyz"] == [0.2, 0.2, 0.3]


def test_robotwin_move_to_is_not_ok_when_the_arm_stops_short():
    env = RoboTwinEnv()
    env.reach = False
    result = _robotwin(env).move_to(arm="left", xyz=[0.4, 0.2, 0.3]).data

    assert result["ok"] is False
    assert result["stop_reason"] == "not_reached"
    assert result["dist_m"] == pytest.approx(0.3)


def test_robotwin_plan_failure_is_a_stop_reason():
    env = RoboTwinEnv()
    env.plan_status = "Fail"
    result = _robotwin(env).move_to(arm="right", xyz=[0.4, 0.2, 0.3]).data

    assert result["ok"] is False
    assert result["stop_reason"] == "plan_failed"
    assert result["steps_used"] == 0
    assert not env.qpos_actions


def test_robotwin_rotate_wrist_takes_radians():
    env = RoboTwinEnv()
    driver = _robotwin(env)
    start = RoboTwinPrimitives._arm_pose(driver, "left")

    result = driver.rotate_wrist(arm="left", delta_yaw=np.pi / 2).data

    w, x, y, z = env.planned["left"][3:]
    end = Rotation.from_quat([x, y, z, w])
    sw, sx, sy, sz = start[3:]
    begin = Rotation.from_quat([sx, sy, sz, sw])
    heading = lambda r: np.arctan2(r.as_matrix()[1, 0], r.as_matrix()[0, 0])  # noqa: E731
    assert heading(end) - heading(begin) == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(env.planned["left"][:3], start[:3])
    assert result["ok"] is True
    assert abs(result["yaw_error"]) < 1e-6


def test_robotwin_set_gripper_accepts_commands_and_fractions():
    env = RoboTwinEnv()
    driver = _robotwin(env)

    closed = driver.set_gripper(arm="right", command="close", steps=2).data
    partial = driver.set_gripper(arm="right", command=0.25, steps=1).data
    released = driver.release(arm="right", steps=1).data

    assert closed["gripper_opening"] == 0.0
    assert partial["gripper_opening"] == 0.25
    assert released["gripper_opening"] == 1.0
    assert closed["ok"] is partial["ok"] is released["ok"] is True


# ---------------------------------------------------------------------------
# RoboCasa: native -1 open, +1 close; "hold" servos to the entry finger width.
# ---------------------------------------------------------------------------


class RoboCasaEnv:
    def __init__(self):
        self.eef_pos = np.zeros(3)
        self.gripper_qpos = np.asarray([0.02, -0.02])
        self.actions: list[np.ndarray] = []
        self.terminated = False

    def reset(self):
        return {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.eef_pos = self.eef_pos + np.asarray(action[:3]) * 0.05
        self.gripper_qpos = np.clip(self.gripper_qpos - action[6] * 0.004, -0.04, 0.04)
        return {}, 0, False, False, {}


def _robocasa(env: RoboCasaEnv, tmp_path) -> RoboCasaPrimitives:
    return RoboCasaPrimitives(
        env_client=env,
        workdir=str(tmp_path),
        hi_res=False,
        vla_client=SimpleNamespace(),
        allow_reset=False,
    )


def test_robocasa_move_to_holds_finger_width_by_default(tmp_path):
    env = RoboCasaEnv()
    result = _robocasa(env, tmp_path).move_to([0.05, 0.0, 0.0]).data

    assert {float(action[6]) for action in env.actions} == {0.0}
    assert set(MOTION_RESULT_KEYS) <= set(result)
    assert result["ok"] is True
    assert result["gripper_opening"] == pytest.approx(0.5)


@pytest.mark.parametrize(("command", "native"), [("open", -1.0), ("close", 1.0)])
def test_robocasa_set_gripper_commands(tmp_path, command, native):
    env = RoboCasaEnv()
    result = _robocasa(env, tmp_path).set_gripper(command, steps=3).data

    assert [float(action[6]) for action in env.actions] == [native] * 3
    assert result["ok"] is True
    assert result["steps_used"] == 3
