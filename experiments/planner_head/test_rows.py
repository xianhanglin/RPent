"""Check one solved episode becomes next-tool rows without future calls in the state."""
import json
import unittest
from pathlib import Path

from rows import TOOLS, rows_of_episode

EPISODE = (
    Path(__file__).resolve().parent.parent
    / "qwen36_27b/runs/qwen3.6-27b_none_vllm/libero_goal_swap/t1/s2"
)


class RowsTest(unittest.TestCase):
    def test_recipe_order_and_no_future_leak(self):
        rows = rows_of_episode(EPISODE)
        recipe = [
            json.loads(line)["action"]
            for line in (EPISODE / "goal_swap_t1_s2_recipe.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual([row["questions"]["next_tool"]["label"] for row in rows], recipe)
        self.assertEqual(set(rows[0]["questions"]["next_tool"]["criteria"]), set(TOOLS))
        self.assertIn("Calls so far:\n(none)", rows[0]["state"])
        self.assertNotIn("pick up the bowl", rows[0]["state"])
        self.assertIn("segment", rows[1]["state"])
        self.assertNotIn("pi0_pick", rows[1]["state"])
        self.assertTrue(all(row["_meta"]["split"] == "train" for row in rows))
        self.assertTrue(all(row["_meta"]["task"] == 1 for row in rows))

    def test_heldout_tasks(self):
        rows = rows_of_episode(EPISODE, holdout_tasks=(1,))
        self.assertTrue(rows)
        self.assertTrue(all(row["_meta"]["split"] == "heldout" for row in rows))


if __name__ == "__main__":
    unittest.main()
