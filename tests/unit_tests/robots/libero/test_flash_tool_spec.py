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

"""Flash plans recorded with legacy tool arguments replay with current tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robots.libero.flash import replay
from rpent.tools.core import TOOL_SPEC_VERSION

ANCHORS = {
    "anchors": [{"phrase": "the mug", "median_xy": [0.1, 0.2], "locator": "segment"}]
}


def _write(root: Path, plan: list[dict], **document) -> None:
    (root / "t0_plan.json").write_text(json.dumps({"plan": plan, **document}))
    (root / "t0_anchors.json").write_text(json.dumps(ANCHORS))


def test_legacy_plans_are_translated_on_load(tmp_path: Path):
    _write(
        tmp_path,
        [
            {"action": "move_to", "arguments": {"xyz": [0, 0, 1]}, "anchor": "the mug"},
            {"action": "set_gripper", "arguments": {"gripper": 1, "steps": 5}},
            {"action": "release", "arguments": {"max_steps": 20}},
        ],
    )

    plan = replay.load(tmp_path, "t0")["plan"]

    assert plan == [
        {
            "action": "move_to",
            "arguments": {"xyz": [0, 0, 1], "gripper": "open"},
            "anchor": "the mug",
        },
        {"action": "set_gripper", "arguments": {"command": "close", "steps": 5}},
        {"action": "release", "arguments": {"steps": 20}},
    ]


def test_current_plans_load_unchanged(tmp_path: Path):
    plan = [{"action": "move_to", "arguments": {"xyz": [0, 0, 1]}}]
    _write(tmp_path, plan, tool_spec=TOOL_SPEC_VERSION)

    assert replay.load(tmp_path, "t0")["plan"] == plan


@pytest.mark.parametrize("legacy", [True, False])
def test_generated_plans_record_the_tool_spec(tmp_path: Path, legacy: bool):
    from robots.libero.flash.generate import generate_flash_plan

    tag = "goal_task_t0_s0"
    audit = tmp_path / f"{tag}.json"
    audit.write_text(
        json.dumps(
            {"libero_terminated": True, "task_language": "put the bowl on the plate"}
        )
    )
    recipe = tmp_path / f"{tag}_recipe.jsonl"
    gripper = {"gripper": 1} if legacy else {"gripper": "close"}
    recipe.write_text(
        json.dumps({"action": "move_to", "xyz": [0.1, 0.2, 0.3], **gripper}) + "\n"
    )

    generate_flash_plan(audit, recipe, tmp_path / "out", legacy_tool_args=legacy)

    (document_path,) = (tmp_path / "out").glob("*_plan.json")
    document = json.loads(document_path.read_text())
    assert document["tool_spec"] == TOOL_SPEC_VERSION
    assert document["plan"][0]["arguments"]["gripper"] == "close"
