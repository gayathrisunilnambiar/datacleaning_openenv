"""Tests for Pydantic models and action validation."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from environment.models import Action, Observation


class ActionModelTests(unittest.TestCase):
    """Validation tests for action payloads."""

    def test_fill_nulls_median_is_valid(self) -> None:
        action = Action(
            action_type="fill_nulls",
            column=" age ",
            params={"strategy": "median"},
        )
        self.assertEqual(action.column, "age")
        self.assertEqual(action.params, {"strategy": "median"})

    def test_fill_nulls_constant_requires_value(self) -> None:
        with self.assertRaises(ValidationError):
            Action(
                action_type="fill_nulls",
                column="salary",
                params={"strategy": "constant"},
            )

    def test_cast_column_accepts_dtype_alias(self) -> None:
        action = Action(
            action_type="cast_column",
            column="quantity",
            params={"target_dtype": "float"},
        )
        self.assertEqual(action.action_type.value, "cast_column")

    def test_submit_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            Action(action_type="submit", column="age")


class ObservationModelTests(unittest.TestCase):
    """Shape tests for the observation contract."""

    def test_observation_requires_task_metadata_fields(self) -> None:
        observation = Observation(
            task_id="easy",
            step_number=0,
            current_df=[{"id": 1, "age": None}],
            dirty_columns=["age"],
            columns_meta={"id": "int64", "age": "float64"},
            episode_reward_so_far=0.0,
            done=False,
            max_steps=20,
            task_description="Example task",
        )
        self.assertEqual(observation.max_steps, 20)
        self.assertEqual(observation.task_description, "Example task")


if __name__ == "__main__":
    unittest.main()

