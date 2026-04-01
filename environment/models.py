"""
Pydantic v2 models for the Data Cleaning OpenEnv environment.

Defines the core data structures used across the environment:
ActionType enum, Action, Observation, StepResult, and RewardBreakdown.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# ActionType enum
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Enumeration of all valid cleaning actions an agent can issue."""

    drop_duplicates = "drop_duplicates"
    fill_nulls = "fill_nulls"
    cast_column = "cast_column"
    remove_outliers = "remove_outliers"
    rename_column = "rename_column"
    normalize_values = "normalize_values"
    submit = "submit"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

# Actions that operate on exactly one column and therefore *require* the
# ``column`` field to be set.
_COLUMN_REQUIRED_ACTIONS: set[ActionType] = {
    ActionType.fill_nulls,
    ActionType.cast_column,
    ActionType.remove_outliers,
    ActionType.rename_column,
    ActionType.normalize_values,
}

# Actions that need extra parameters via the ``params`` dict.
_PARAMS_REQUIRED_ACTIONS: dict[ActionType, list[str]] = {
    ActionType.fill_nulls: ["strategy"],        # e.g. "mean", "median", "mode", "constant"
    ActionType.cast_column: ["target_dtype"],    # e.g. "int64", "float64", "str"
    ActionType.rename_column: ["new_name"],      # the new column name
}


class Action(BaseModel):
    """A single cleaning action issued by the agent.

    Attributes:
        action_type: The type of cleaning operation to perform.
        column: Target column name (required for column-specific actions).
        params: Extra parameters needed by some action types.
    """

    action_type: ActionType
    column: Optional[str] = Field(default=None, description="Target column name")
    params: Optional[dict] = Field(default=None, description="Extra parameters for the action")

    # ----- field-level validators ------------------------------------------

    @field_validator("column", mode="before")
    @classmethod
    def _strip_column_name(cls, v: Optional[str]) -> Optional[str]:
        """Strip leading/trailing whitespace from column names."""
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
        return v

    # ----- model-level validators ------------------------------------------

    @model_validator(mode="after")
    def _validate_action_constraints(self) -> "Action":
        """Ensure column and params are provided where required."""
        # Column requirement check
        if self.action_type in _COLUMN_REQUIRED_ACTIONS and self.column is None:
            raise ValueError(
                f"Action '{self.action_type.value}' requires a 'column' to be specified."
            )

        # Params requirement check
        if self.action_type in _PARAMS_REQUIRED_ACTIONS:
            required_keys = _PARAMS_REQUIRED_ACTIONS[self.action_type]
            if self.params is None:
                raise ValueError(
                    f"Action '{self.action_type.value}' requires params with keys: {required_keys}"
                )
            missing = [k for k in required_keys if k not in self.params]
            if missing:
                raise ValueError(
                    f"Action '{self.action_type.value}' is missing required param keys: {missing}"
                )

        # Submit should not carry column or params
        if self.action_type == ActionType.submit:
            if self.column is not None or self.params is not None:
                raise ValueError(
                    "'submit' action should not include 'column' or 'params'."
                )

        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Observable state returned to the agent after each step.

    Attributes:
        task_id: Unique identifier for the cleaning task.
        step_number: How many steps have been taken in the current episode.
        current_df: The current DataFrame serialized as a list of row-dicts.
        dirty_columns: Column names that still differ from the ground truth.
        columns_meta: Mapping of column name → dtype string.
        episode_reward_so_far: Cumulative reward collected in this episode.
        done: Whether the episode has ended.
    """

    task_id: str
    step_number: int = Field(ge=0)
    current_df: list[dict]
    dirty_columns: list[str]
    columns_meta: dict[str, str]
    episode_reward_so_far: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Itemised breakdown of the reward for a single step.

    Attributes:
        column_improvement: Partial credit for columns that moved closer to ground truth.
        submit_bonus: Bonus awarded on successful submit (similarity > 0.80).
        redundancy_penalty: Penalty applied when an action causes zero change.
        total: Net reward for this step.
    """

    column_improvement: float = 0.0
    submit_bonus: float = 0.0
    redundancy_penalty: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Composite result returned by ``DataCleaningEnv.step()``.

    Attributes:
        observation: The new observation after applying the action.
        reward: Scalar reward for the step.
        done: Whether the episode has ended.
        info: Auxiliary info dict (contains the RewardBreakdown, etc.).
    """

    observation: Observation
    reward: float
    done: bool
    info: dict
