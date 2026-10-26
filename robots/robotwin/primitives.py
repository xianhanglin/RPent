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

"""RoboTwin primitives built on RLinf environment APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
from pydantic import Field

from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.robot_spec import MODEL_SPEC, ROBOTWIN_CAMERA_NAMES
from robots.robotwin.vla_client import LingBotVLAClient
from rpent.tools import ToolResult, tool
from rpent.tools.core import (
    Arm,
    GripperMode,
    GripperTarget,
    WorldXYZ,
    motion_result,
    stop_reason,
    wrap_angle,
)

# Orientation tolerance for reporting a yaw target as reached, radians.
_YAW_TOL = 0.05


def _qmult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _yaw_of(quat_wxyz: np.ndarray) -> float:
    """World heading of the EEF X axis for a wxyz quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def _rotate_about_world_z(quat_wxyz: np.ndarray, angle: float) -> np.ndarray:
    """Compose a world-Z rotation before an EEF orientation."""
    world_z = np.asarray([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    return _qmult(world_z, np.asarray(quat_wxyz, dtype=np.float64))


class RoboTwinPrimitives:
    """Compose RoboTwin operations from the RLinf environment API."""

    def __init__(
        self,
        *,
        env: RoboTwinEnvClient,
        model: LingBotVLAClient,
        seed: int,
        check_cancelled: Callable[[], None],
        seed_mode: str = "exact",
    ):
        if seed_mode != "exact":
            raise ValueError("standard RoboTwin integration requires seed_mode='exact'")
        self.env = env
        self.model = model
        self.seed = int(seed)
        self._check_cancelled = check_cancelled
        self.policy_actions = 0
        self.native_actions = 0
        self._recording = False
        self._frames: list[np.ndarray] = []

    def start_recording(self) -> None:
        self._recording = True
        self._frames = []

    def record_frame(self, rgb: Any) -> None:
        self._frames.append(np.ascontiguousarray(np.asarray(rgb)))

    def recorded_frame_count(self) -> int:
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        return list(self._frames[int(start) :])

    def reset(
        self,
        *,
        instruction: str | None = None,
        feasibility_precheck: bool = True,
    ) -> dict[str, Any]:
        """Reset the RoboTwin episode and return the native info plus success."""
        del instruction, feasibility_precheck
        _, info = self.env.reset()
        return {**info, "success": True}

    @staticmethod
    def _completion(
        *, requested: int, executed: int, status: dict[str, Any]
    ) -> dict[str, Any]:
        step_lim = status.get("step_lim")
        budget_exhausted = step_lim is not None and int(
            status.get("take_action_cnt", 0)
        ) >= int(step_lim)
        completed = executed == requested
        if status.get("eval_success") is True:
            stop_reason = "native_success"
        elif budget_exhausted:
            stop_reason = "budget_exhausted"
        elif completed:
            stop_reason = "completed"
        else:
            stop_reason = "runtime_failure"
        return {
            "completed": completed,
            "requested_steps": requested,
            "executed_steps": executed,
            "stop_reason": stop_reason,
        }

    def _build_lingbot_observation(self) -> dict[str, Any]:
        """Assemble the rgb-only observation the LingBot policy infers on."""
        views = {}
        for camera_name in ROBOTWIN_CAMERA_NAMES:
            views[camera_name] = {
                "rgb": np.asarray(self.env.render_camera(camera_name))
            }
        return {
            "views": views,
            "robot_state": self.env.last_info["robot_state"],
            "task_language": self.env.get_task_language(),
        }

    @staticmethod
    def _validate_qpos_updates_request(updates: Any) -> list[dict[str, Any]]:
        if not isinstance(updates, list) or not updates:
            raise ValueError("qpos updates must contain at least one update")
        normalized = []
        for update in updates:
            if not isinstance(update, dict):
                raise TypeError("qpos update must be a mapping")
            arm = update.get("arm")
            if arm not in ("left", "right"):
                raise ValueError("arm must be 'left' or 'right'")
            if update.get("arm_qpos") is None and update.get("gripper") is None:
                raise ValueError("qpos update must set arm_qpos and/or gripper")
            item: dict[str, Any] = {"arm": arm}
            if update.get("arm_qpos") is not None:
                arm_qpos = np.asarray(update["arm_qpos"], dtype=np.float64)
                if arm_qpos.shape != (6,) or not np.isfinite(arm_qpos).all():
                    raise ValueError("arm_qpos must be finite and have shape (6,)")
                item["arm_qpos"] = arm_qpos
            if update.get("gripper") is not None:
                gripper = float(update["gripper"])
                if not np.isfinite(gripper):
                    raise ValueError("gripper must be finite")
                item["gripper"] = gripper
            normalized.append(item)
        return normalized

    def apply_qpos_updates(
        self,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compose qpos waypoint actions from the latest robot state."""
        if self.env.terminated or self.env.truncated:
            raise RuntimeError("RoboTwin common episode is terminal; reset is required")
        updates = self._validate_qpos_updates_request(updates)
        executed = 0
        episode_status: dict[str, Any] | None = None
        for update in updates:
            state = self.env.last_info["robot_state"]
            qpos_target14 = np.asarray(state.get("qpos_target14"), dtype=np.float64)
            if qpos_target14.shape != (14,) or not np.isfinite(qpos_target14).all():
                raise ValueError(
                    "RoboTwin robot_state.qpos_target14 must be finite and have "
                    "shape (14,)"
                )
            action = qpos_target14.copy()
            offset = 0 if update["arm"] == "left" else 7
            if "arm_qpos" in update:
                action[offset : offset + 6] = update["arm_qpos"]
            if "gripper" in update:
                action[offset + 6] = update["gripper"]
            obs, _, _, _, info = self.env.step(action, action_type="qpos")
            if self._recording and isinstance(obs, dict) and "main_images" in obs:
                self.record_frame(obs["main_images"])
            executed += int(info.get("executed_actions", 0))
            episode_status = info["episode_status"]
            if self.env.terminated or self.env.truncated:
                break
        if episode_status is None:
            raise RuntimeError("RoboTwin qpos composition executed no updates")
        return {
            "action_type": "qpos",
            "requested_actions": len(updates),
            "executed_actions": executed,
            "episode_status": episode_status,
        }

    def execute_action(
        self, values: list[float], action_type: str | None = None
    ) -> ToolResult:
        """Execute one native qpos or end-effector waypoint."""
        if action_type is None:
            action_type = next(iter(self.env.action_specs))
        action = self.env.validate_action(values, action_type=action_type)
        self._check_cancelled()
        obs, _, _, _, info = self.env.step(action, action_type=action_type)
        if self._recording and isinstance(obs, dict) and "main_images" in obs:
            self.record_frame(obs["main_images"])
        executed = int(info.get("executed_actions", 0))
        self.native_actions += executed
        return ToolResult(
            data={
                **self._completion(
                    requested=1, executed=executed, status=info["episode_status"]
                ),
                "action_type": action_type,
                "episode_status": info["episode_status"],
            }
        )

    @tool
    def lingbot_act(
        self,
        *,
        chunks: Annotated[int, Field(ge=1, json_schema_extra={"default": 4})] = 4,
        use_length: Annotated[
            int, Field(json_schema_extra={"default": 50, "const": 50})
        ] = 50,
        prompt: str | None = None,
    ) -> ToolResult:
        """Run LingBot-VLA eef16 actions using the native task instruction. The optional prompt is recorded but never sent to the policy."""
        if int(chunks) < 1:
            raise ValueError("chunks must be at least 1")
        if int(use_length) != MODEL_SPEC.use_length:
            raise ValueError(
                f"RoboTwin LingBot requires use_length={MODEL_SPEC.use_length}"
            )
        executed = 0
        requested = int(chunks) * MODEL_SPEC.use_length
        native_prompt = None
        for _ in range(int(chunks)):
            self._check_cancelled()
            status = self.env.last_info["episode_status"]
            step_lim = status.get("step_lim")
            budget_exhausted = step_lim is not None and int(
                status.get("take_action_cnt", 0)
            ) >= int(step_lim)
            if status.get("eval_success") is True or budget_exhausted:
                break
            observation = self._build_lingbot_observation()
            native_prompt = observation["task_language"]
            actions = self.model.infer(observation)[: MODEL_SPEC.use_length]
            self._check_cancelled()
            payload, _, _, _, info = self.env.chunk_step(
                actions,
                action_type="ee",
                return_all_frames=self._recording
                and self.env.execution_capabilities.get("chunk_step_all_frames")
                is True,
            )
            if self._recording and isinstance(payload, dict) and "frames" in payload:
                for frame in payload["frames"]:
                    self.record_frame(frame)
            count = int(info.get("executed_actions", 0))
            executed += count
            self.policy_actions += count
            self.native_actions += count
            self._check_cancelled()
        status = self.env.last_info["episode_status"]
        return ToolResult(
            data={
                **self._completion(
                    requested=requested, executed=executed, status=status
                ),
                "success": True,
                "prompt": native_prompt,
                "agent_prompt_ignored": prompt is not None,
                "ignored_agent_prompt": prompt,
                "episode_status": status,
            }
        )

    @staticmethod
    def _gripper_target(gripper: GripperMode) -> float | None:
        """Native gripper target: ``0`` closed, ``1`` open, ``None`` keeps it."""
        return {"hold": None, "open": 1.0, "close": 0.0}[gripper]

    def _arm_pose(self, arm: Arm) -> np.ndarray:
        return np.asarray(
            self.env.last_info["robot_state"][f"{arm}_eef_pose"], dtype=np.float64
        )

    def _result(
        self,
        *,
        arm: Arm,
        reached: bool,
        executed: int,
        native_stop_reason: str,
        **extra: Any,
    ) -> ToolResult:
        """Report a RoboTwin action in the shared motion result shape."""
        terminated = bool(self.env.terminated) or native_stop_reason == "native_success"
        truncated = bool(self.env.truncated) or native_stop_reason == "budget_exhausted"
        state = self.env.last_info["robot_state"]
        return motion_result(
            ok=reached,
            stop_reason=stop_reason(
                reached=reached, terminated=terminated, truncated=truncated
            ),
            steps_used=executed,
            eef_xyz=self._arm_pose(arm)[:3],
            gripper_opening=float(state[f"{arm}_gripper"]),
            terminated=terminated,
            truncated=truncated,
            native_stop_reason=native_stop_reason,
            **extra,
        )

    def _move_pose(
        self,
        *,
        arm: Arm,
        xyz: list[float],
        quat: np.ndarray,
        gripper: GripperMode,
        substeps: int,
        tol: float,
        yaw_target: float | None = None,
    ) -> ToolResult:
        """Plan and execute a qpos path to a target end-effector pose."""
        if int(substeps) < 0:
            raise ValueError("substeps must be non-negative")
        self._check_cancelled()
        target = np.asarray([*xyz, *quat], dtype=np.float64)
        planned = self.env.plan_arm_path(arm, target)
        self._check_cancelled()
        if planned["status"] != "Success" or planned.get("position") is None:
            pose = self._arm_pose(arm)
            return motion_result(
                ok=False,
                stop_reason="plan_failed",
                steps_used=0,
                eef_xyz=pose[:3],
                gripper_opening=float(
                    self.env.last_info["robot_state"][f"{arm}_gripper"]
                ),
                terminated=bool(self.env.terminated),
                truncated=bool(self.env.truncated),
                dist_m=float(np.linalg.norm(pose[:3] - np.asarray(xyz))),
                plan_status=planned["status"],
                hint="target may be unreachable or in collision",
            )
        path = np.asarray(planned["position"], dtype=np.float64)
        if substeps == 1:
            path = path[-1:]
        elif substeps >= 2 and len(path) > substeps:
            indices = np.linspace(0, len(path) - 1, substeps).astype(int)
            path = path[indices]
        target_gripper = self._gripper_target(gripper)
        updates = [
            {"arm": arm, "arm_qpos": waypoint, "gripper": target_gripper}
            for waypoint in path
        ]
        execution = self.apply_qpos_updates(updates)
        executed = int(execution.get("executed_actions", 0))
        self.native_actions += executed
        self._check_cancelled()
        completion = self._completion(
            requested=len(updates),
            executed=executed,
            status=execution["episode_status"],
        )
        pose = self._arm_pose(arm)
        dist = float(np.linalg.norm(pose[:3] - np.asarray(xyz, dtype=np.float64)))
        yaw_error = (
            None if yaw_target is None else wrap_angle(yaw_target - _yaw_of(pose[3:]))
        )
        return self._result(
            arm=arm,
            reached=completion["completed"]
            and dist <= tol
            and (yaw_error is None or abs(yaw_error) <= _YAW_TOL),
            executed=executed,
            native_stop_reason=completion["stop_reason"],
            dist_m=dist,
            yaw_error=yaw_error,
            plan_status=planned["status"],
            waypoints=len(path),
            episode_status=execution["episode_status"],
        )

    @tool
    def move_to(
        self,
        *,
        arm: Arm,
        xyz: WorldXYZ,
        gripper: GripperMode = "hold",
        target_yaw: float | None = None,
        quat: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None,
        substeps: Annotated[int, Field(ge=0, json_schema_extra={"default": 25})] = 25,
        tol: float = 0.01,
    ) -> ToolResult:
        """Plan and move one arm to a world-frame xyz. Keeps the current orientation unless target_yaw or a full wxyz quat is given. The native planner returns qpos waypoints executed with fresh state.

        Args:
            arm: Arm to move
            xyz: World-frame target [x, y, z] in meters
            gripper: 'hold' keeps the current gripper target (default), 'open' opens, 'close' closes
            target_yaw: Optional world-frame yaw target in radians; keeps the current tilt
            quat: Optional full world-frame orientation, wxyz; excludes target_yaw
            substeps: Maximum executed waypoints sampled from the planned path (default 25)
            tol: Position tolerance for ok, m (default 0.01)
        """
        if target_yaw is not None and quat is not None:
            raise ValueError("pass target_yaw or quat, not both")
        current = self._arm_pose(arm)[3:]
        if quat is not None:
            orientation = np.asarray(quat, dtype=np.float64)
        elif target_yaw is not None:
            orientation = _rotate_about_world_z(
                current, wrap_angle(target_yaw - _yaw_of(current))
            )
        else:
            orientation = current
        return self._move_pose(
            arm=arm,
            xyz=list(xyz),
            quat=orientation,
            gripper=gripper,
            substeps=substeps,
            tol=tol,
            yaw_target=target_yaw,
        )

    @tool
    def rotate_wrist(
        self,
        *,
        arm: Arm,
        delta_yaw: float | None = None,
        target_yaw: float | None = None,
        gripper: GripperMode = "hold",
        substeps: Annotated[int, Field(ge=0, json_schema_extra={"default": 25})] = 25,
    ) -> ToolResult:
        """Rotate one EEF about world Z in place. Provide either delta_yaw (relative) or target_yaw (absolute).

        Args:
            arm: Arm to rotate
            delta_yaw: Relative yaw delta, rad
            target_yaw: Absolute world-frame yaw target, rad
            gripper: 'hold' keeps the current gripper target (default), 'open' opens, 'close' closes
            substeps: Maximum executed waypoints sampled from the planned path (default 25)
        """
        if (delta_yaw is None) == (target_yaw is None):
            raise ValueError("pass exactly one of delta_yaw or target_yaw")
        pose = self._arm_pose(arm)
        start_yaw = _yaw_of(pose[3:])
        goal = start_yaw + float(delta_yaw) if target_yaw is None else target_yaw
        result = self._move_pose(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=_rotate_about_world_z(pose[3:], wrap_angle(goal - start_yaw)),
            gripper=gripper,
            substeps=substeps,
            tol=0.01,
            yaw_target=goal,
        )
        result.data["start_yaw"] = round(start_yaw, 4)
        return result

    @tool
    def set_gripper(
        self,
        *,
        arm: Arm,
        command: GripperTarget | Annotated[float, Field(ge=0, le=1)],
        steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 10})] = 10,
    ) -> ToolResult:
        """Move one gripper open, closed, or to an opening fraction (0 closed, 1 open) over `steps` native actions.

        Args:
            arm: Gripper's arm
            command: 'open', 'close', or an opening fraction from 0 (closed) to 1 (open)
            steps: Number of native actions (default 10)
        """
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        self._check_cancelled()
        current = float(self.env.last_info["robot_state"][f"{arm}_gripper"])
        target = (
            self._gripper_target(command)
            if isinstance(command, str)
            else float(command)
        )
        step_count = int(steps)
        values = [
            current + (target - current) * index / step_count
            for index in range(1, step_count + 1)
        ]
        updates = [{"arm": arm, "gripper": float(value)} for value in values]
        execution = self.apply_qpos_updates(updates)
        executed = int(execution.get("executed_actions", 0))
        self.native_actions += executed
        self._check_cancelled()
        completion = self._completion(
            requested=len(updates),
            executed=executed,
            status=execution["episode_status"],
        )
        return self._result(
            arm=arm,
            reached=completion["completed"],
            executed=executed,
            native_stop_reason=completion["stop_reason"],
            episode_status=execution["episode_status"],
        )

    @tool
    def release(
        self,
        *,
        arm: Arm,
        steps: Annotated[int, Field(ge=1, json_schema_extra={"default": 10})] = 10,
    ) -> ToolResult:
        """Open one gripper fully over `steps` native actions.

        Args:
            arm: Gripper's arm
            steps: Number of native actions (default 10)
        """
        return self.set_gripper(arm=arm, command="open", steps=steps)

    def status(self) -> dict[str, Any]:
        """Return the native episode status plus action counters."""
        return {
            **self.env.last_info["episode_status"],
            "policy_actions": self.policy_actions,
            "native_actions": self.native_actions,
        }

    @tool(readonly=True)
    def finish(self, *, status: str, summary: str) -> ToolResult:
        """Stop the run. A fresh native status query is authoritative; requesting success cannot override TASK_ENV.eval_success."""
        requested_success = status.lower() == "success"
        try:
            native = self.status()
        except Exception as error:  # The terminal tool must still stop the Planner.
            return ToolResult(
                data={
                    "_finish": True,
                    "status": "error",
                    "summary": summary,
                    "requested_status": status,
                    "requested_success": requested_success,
                    "runtime_error": f"{type(error).__name__}: {error}",
                }
            )
        verified_success = native.get("eval_success") is True
        reported_status = (
            "success"
            if verified_success
            else ("failure" if requested_success else status)
        )
        return ToolResult(
            data={
                "_finish": True,
                "status": reported_status,
                "summary": summary,
                "requested_success": requested_success,
                "success": verified_success,
                "episode_status": native,
            }
        )
