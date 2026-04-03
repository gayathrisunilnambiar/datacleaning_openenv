"""Tests for the stateful DataCleaningEnv loop."""

from __future__ import annotations

import unittest

from environment.env import DataCleaningEnv


class EnvironmentTests(unittest.TestCase):
    """Integration tests for reset, step, reward shaping, and submit."""

    def test_reset_returns_expected_observation_shape(self) -> None:
        env = DataCleaningEnv("easy")
        observation = env.reset()
        self.assertEqual(observation.task_id, "easy")
        self.assertEqual(observation.step_number, 0)
        self.assertEqual(observation.max_steps, 20)
        self.assertFalse(observation.done)
        self.assertTrue(len(observation.current_df) > 0)

    def test_invalid_action_becomes_safe_no_op(self) -> None:
        env = DataCleaningEnv("easy")
        result = env.step(
            {
                "action_type": "fill_nulls",
                "column": "missing_column",
                "params": {"strategy": "median"},
            }
        )
        self.assertFalse(result.done)
        self.assertEqual(result.reward, -0.05)
        self.assertIsNotNone(result.info.error)
        self.assertFalse(result.info.action_applied)

    def test_scripted_easy_episode_finishes(self) -> None:
        env = DataCleaningEnv("easy")
        env.step({"action_type": "drop_duplicates"})
        env.step({"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}})
        env.step(
            {
                "action_type": "fill_nulls",
                "column": "salary",
                "params": {"strategy": "median"},
            }
        )
        result = env.step({"action_type": "submit"})
        self.assertTrue(result.done)
        self.assertIsNotNone(result.info.grader_score)
        self.assertGreaterEqual(result.info.grader_score or 0.0, 0.0)
        self.assertLessEqual(result.info.grader_score or 1.0, 1.0)

    def test_max_steps_forces_episode_end(self) -> None:
        env = DataCleaningEnv("easy")
        for _ in range(env.max_steps):
            result = env.step({"action_type": "drop_duplicates"})
        self.assertTrue(result.done)
        self.assertTrue(result.info.max_steps_reached)
        self.assertIsNotNone(result.info.grader_score)


if __name__ == "__main__":
    unittest.main()

