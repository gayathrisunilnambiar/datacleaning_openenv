# Enriched Step Info for DataCleaningEnv

## Summary

Extended the `info` dict returned by `DataCleaningEnv.step()` with richer diagnostics for agent observability and logging.

## Changes Made

### [models.py](file:///c:/Gayathri/DataClean_OpenEnv/environment/models.py) — Added optional fields to `StepInfo`

```diff:models.py
"""Pydantic models shared across the environment, API, and baseline agent."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_FILL_NULL_STRATEGIES = {"mean", "median", "mode", "constant"}
_CAST_DTYPES = {"int", "float", "str", "bool", "datetime"}
_REMOVE_OUTLIER_METHODS = {"iqr", "zscore"}


class StrictBaseModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


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

class Action(StrictBaseModel):
    """A single cleaning action issued by the agent."""

    action_type: ActionType
    column: str | None = Field(default=None, description="Target column name")
    params: dict[str, object] | None = Field(
        default=None,
        description="Extra action parameters. Contents vary by action_type.",
    )

    @field_validator("column", mode="before")
    @classmethod
    def _strip_column_name(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value

    @model_validator(mode="after")
    def _validate_action_constraints(self) -> "Action":
        """Validate action-specific requirements without over-restricting callers."""
        if self.action_type in _COLUMN_REQUIRED_ACTIONS and self.column is None:
            raise ValueError(
                f"Action '{self.action_type.value}' requires a non-empty 'column'."
            )

        params = dict(self.params or {})

        if self.action_type == ActionType.fill_nulls:
            strategy = str(params.get("strategy", "")).strip().lower()
            if strategy not in _FILL_NULL_STRATEGIES:
                raise ValueError(
                    f"'fill_nulls' requires strategy in {sorted(_FILL_NULL_STRATEGIES)}."
                )
            if strategy == "constant" and "value" not in params:
                raise ValueError("'fill_nulls' with strategy='constant' requires params['value'].")

        if self.action_type == ActionType.cast_column:
            dtype_value = params.get("dtype", params.get("target_dtype"))
            dtype_name = str(dtype_value or "").strip().lower()
            if dtype_name not in _CAST_DTYPES:
                raise ValueError(f"'cast_column' requires dtype in {sorted(_CAST_DTYPES)}.")

        if self.action_type == ActionType.remove_outliers:
            method = str(params.get("method", "")).strip().lower()
            if method not in _REMOVE_OUTLIER_METHODS:
                raise ValueError(
                    f"'remove_outliers' requires method in {sorted(_REMOVE_OUTLIER_METHODS)}."
                )
            if "threshold" in params:
                try:
                    float(params["threshold"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("'remove_outliers' threshold must be numeric.") from exc

        if self.action_type == ActionType.rename_column:
            new_name = str(params.get("new_name", "")).strip()
            if not new_name:
                raise ValueError("'rename_column' requires a non-empty params['new_name'].")

        if self.action_type == ActionType.normalize_values:
            mapping = params.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("'normalize_values' requires a non-empty params['mapping'].")

        if self.action_type == ActionType.submit and (self.column is not None or self.params):
            raise ValueError("'submit' must not include 'column' or 'params'.")

        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(StrictBaseModel):
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
    current_df: list[dict[str, object | None]]
    dirty_columns: list[str]
    columns_meta: dict[str, str]
    episode_reward_so_far: float = 0.0
    done: bool = False
    max_steps: int = Field(ge=1)
    task_description: str


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(StrictBaseModel):
    """Itemized breakdown of the reward for a single step.

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

class StepInfo(StrictBaseModel):
    """Auxiliary diagnostics returned alongside each step result."""

    action_applied: bool = False
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    column_deltas: dict[str, float] = Field(default_factory=dict)
    grader_score: float | None = None
    error: str | None = None
    steps_taken: int | None = None
    max_steps_reached: bool = False


class StepResult(StrictBaseModel):
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
    info: StepInfo


class ResetRequest(StrictBaseModel):
    """Request model for starting a new episode."""

    task_id: str = "easy"
    session_id: str | None = None


class ResetResponse(StrictBaseModel):
    """Response model returned by ``POST /reset``."""

    session_id: str
    observation: Observation


class StepRequest(StrictBaseModel):
    """Request model for applying one action."""

    session_id: str
    action: Action


class StateResponse(StrictBaseModel):
    """Response model returned by ``GET /state``."""

    session_id: str
    observation: Observation


class TaskInfoModel(StrictBaseModel):
    """Task metadata surfaced via the HTTP API."""

    task_id: str
    difficulty: str
    description: str
    max_steps: int


class HealthResponse(StrictBaseModel):
    """Simple liveness payload used by local and deployed health checks."""

    status: Literal["healthy"] = "healthy"
    version: str = "1.0.0"


class MetadataResponse(StrictBaseModel):
    """High-level environment metadata for validator and UI discovery."""

    name: str
    description: str
    version: str
    tags: list[str] = Field(default_factory=list)


class SchemaResponse(StrictBaseModel):
    """JSON schema bundle for the main action and state contracts."""

    action: dict[str, object]
    observation: dict[str, object]
    state: dict[str, object]


class JsonRpcResponse(StrictBaseModel):
    """Minimal JSON-RPC response envelope for MCP reachability checks."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, object] = Field(default_factory=dict)


class ValidateChecks(StrictBaseModel):
    """Detailed results for the ``/validate`` self-check endpoint."""

    reset: bool = False
    step: bool = False
    state: bool = False
    graders: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ValidateResponse(StrictBaseModel):
    """Top-level response for ``POST /validate``."""

    passed: bool
    checks: ValidateChecks
===
"""Pydantic models shared across the environment, API, and baseline agent."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_FILL_NULL_STRATEGIES = {"mean", "median", "mode", "constant"}
_CAST_DTYPES = {"int", "float", "str", "bool", "datetime"}
_REMOVE_OUTLIER_METHODS = {"iqr", "zscore"}


class StrictBaseModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


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

class Action(StrictBaseModel):
    """A single cleaning action issued by the agent."""

    action_type: ActionType
    column: str | None = Field(default=None, description="Target column name")
    params: dict[str, object] | None = Field(
        default=None,
        description="Extra action parameters. Contents vary by action_type.",
    )

    @field_validator("column", mode="before")
    @classmethod
    def _strip_column_name(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value

    @model_validator(mode="after")
    def _validate_action_constraints(self) -> "Action":
        """Validate action-specific requirements without over-restricting callers."""
        if self.action_type in _COLUMN_REQUIRED_ACTIONS and self.column is None:
            raise ValueError(
                f"Action '{self.action_type.value}' requires a non-empty 'column'."
            )

        params = dict(self.params or {})

        if self.action_type == ActionType.fill_nulls:
            strategy = str(params.get("strategy", "")).strip().lower()
            if strategy not in _FILL_NULL_STRATEGIES:
                raise ValueError(
                    f"'fill_nulls' requires strategy in {sorted(_FILL_NULL_STRATEGIES)}."
                )
            if strategy == "constant" and "value" not in params:
                raise ValueError("'fill_nulls' with strategy='constant' requires params['value'].")

        if self.action_type == ActionType.cast_column:
            dtype_value = params.get("dtype", params.get("target_dtype"))
            dtype_name = str(dtype_value or "").strip().lower()
            if dtype_name not in _CAST_DTYPES:
                raise ValueError(f"'cast_column' requires dtype in {sorted(_CAST_DTYPES)}.")

        if self.action_type == ActionType.remove_outliers:
            method = str(params.get("method", "")).strip().lower()
            if method not in _REMOVE_OUTLIER_METHODS:
                raise ValueError(
                    f"'remove_outliers' requires method in {sorted(_REMOVE_OUTLIER_METHODS)}."
                )
            if "threshold" in params:
                try:
                    float(params["threshold"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("'remove_outliers' threshold must be numeric.") from exc

        if self.action_type == ActionType.rename_column:
            new_name = str(params.get("new_name", "")).strip()
            if not new_name:
                raise ValueError("'rename_column' requires a non-empty params['new_name'].")

        if self.action_type == ActionType.normalize_values:
            mapping = params.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("'normalize_values' requires a non-empty params['mapping'].")

        if self.action_type == ActionType.submit and (self.column is not None or self.params):
            raise ValueError("'submit' must not include 'column' or 'params'.")

        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(StrictBaseModel):
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
    current_df: list[dict[str, object | None]]
    dirty_columns: list[str]
    columns_meta: dict[str, str]
    episode_reward_so_far: float = 0.0
    done: bool = False
    max_steps: int = Field(ge=1)
    task_description: str


# ---------------------------------------------------------------------------
# RewardBreakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(StrictBaseModel):
    """Itemized breakdown of the reward for a single step.

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

class StepInfo(StrictBaseModel):
    """Auxiliary diagnostics returned alongside each step result."""

    action_applied: bool = False
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    column_deltas: dict[str, float] = Field(default_factory=dict)
    grader_score: float | None = None
    error: str | None = None
    steps_taken: int | None = None
    max_steps_reached: bool = False

    # ── Enriched diagnostics (every step) ──
    partial_score: float | None = None
    dirty_columns_remaining: int | None = None

    # ── Enriched diagnostics (submit only) ──
    final_score: float | None = None
    grader_breakdown: dict[str, float] | None = None
    steps_used: int | None = None
    steps_budget: int | None = None
    improvement_from_start: float | None = None


class StepResult(StrictBaseModel):
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
    info: StepInfo


class ResetRequest(StrictBaseModel):
    """Request model for starting a new episode."""

    task_id: str = "easy"
    session_id: str | None = None


class ResetResponse(StrictBaseModel):
    """Response model returned by ``POST /reset``."""

    session_id: str
    observation: Observation


class StepRequest(StrictBaseModel):
    """Request model for applying one action."""

    session_id: str
    action: Action


class StateResponse(StrictBaseModel):
    """Response model returned by ``GET /state``."""

    session_id: str
    observation: Observation


class TaskInfoModel(StrictBaseModel):
    """Task metadata surfaced via the HTTP API."""

    task_id: str
    difficulty: str
    description: str
    max_steps: int


class HealthResponse(StrictBaseModel):
    """Simple liveness payload used by local and deployed health checks."""

    status: Literal["healthy"] = "healthy"
    version: str = "1.0.0"


class MetadataResponse(StrictBaseModel):
    """High-level environment metadata for validator and UI discovery."""

    name: str
    description: str
    version: str
    tags: list[str] = Field(default_factory=list)


class SchemaResponse(StrictBaseModel):
    """JSON schema bundle for the main action and state contracts."""

    action: dict[str, object]
    observation: dict[str, object]
    state: dict[str, object]


class JsonRpcResponse(StrictBaseModel):
    """Minimal JSON-RPC response envelope for MCP reachability checks."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, object] = Field(default_factory=dict)


class ValidateChecks(StrictBaseModel):
    """Detailed results for the ``/validate`` self-check endpoint."""

    reset: bool = False
    step: bool = False
    state: bool = False
    graders: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ValidateResponse(StrictBaseModel):
    """Top-level response for ``POST /validate``."""

    passed: bool
    checks: ValidateChecks
```

**New fields (every step):**
| Field | Type | Description |
|---|---|---|
| `partial_score` | `float` | Grader's `partial_score()` — average column similarity |
| `dirty_columns_remaining` | `int` | Count of columns still differing from ground truth |

**New fields (submit / max-steps reached):**
| Field | Type | Description |
|---|---|---|
| `final_score` | `float` | Final grader score |
| `grader_breakdown` | `dict[str, float]` | Per-column similarity scores |
| `steps_used` | `int` | Total steps taken |
| `steps_budget` | `int` | Task's `max_steps` |
| `improvement_from_start` | `float` | `partial_score_now − initial_similarity` |

---

### [env.py](file:///c:/Gayathri/DataClean_OpenEnv/environment/env.py) — Computed and populated enriched fields

```diff:env.py
"""Core stateful environment for the DataCleaning OpenEnv benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import ValidationError

from environment.graders import GRADER_REGISTRY
from environment.models import Action, ActionType, Observation, RewardBreakdown, StepInfo, StepResult
from environment.tasks import TASK_REGISTRY
from environment.tasks.base_task import BaseTask


@dataclass
class EnvironmentConfig:
    """Configurable runtime knobs for one environment instance."""

    submit_bonus_threshold: float = 0.80
    submit_bonus_value: float = 0.30
    no_op_penalty: float = 0.05
    min_step_reward: float = -0.05
    max_step_reward: float = 1.30


class DataCleaningEnv:
    """Interactive single-episode environment for one data-cleaning task."""

    def __init__(self, task_id: str, config: EnvironmentConfig | None = None) -> None:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {sorted(TASK_REGISTRY)}")

        self.task: BaseTask = TASK_REGISTRY[task_id]()
        self.grader = GRADER_REGISTRY[task_id]()
        self.config = config or EnvironmentConfig()

        self.current_df = pd.DataFrame()
        self.step_number = 0
        self.episode_reward = 0.0
        self.done = False
        self.reset()

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def max_steps(self) -> int:
        return int(self.task.max_steps)

    def reset(self) -> Observation:
        """Start a fresh episode for the configured task."""
        self.current_df = self.task.get_dirty_df().copy()
        self.step_number = 0
        self.episode_reward = 0.0
        self.done = False
        return self.state()

    def state(self) -> Observation:
        """Return the current observable state."""
        return Observation(
            task_id=self.task_id,
            step_number=self.step_number,
            current_df=self._serialize_df(self.current_df),
            dirty_columns=self.grader.dirty_columns(self.current_df),
            columns_meta={column: str(dtype) for column, dtype in self.current_df.dtypes.items()},
            episode_reward_so_far=round(self.episode_reward, 6),
            done=self.done,
            max_steps=self.max_steps,
            task_description=self.task.description,
        )

    def step(self, action: Action | dict[str, object]) -> StepResult:
        """Apply one cleaning action and return the resulting transition."""
        if self.done:
            info = StepInfo(
                error="Episode already completed.",
                steps_taken=self.step_number,
                grader_score=self.grader.score(self.current_df),
                max_steps_reached=self.step_number >= self.max_steps,
            )
            observation = self.state()
            return StepResult(observation=observation, reward=0.0, done=True, info=info)

        try:
            action_obj = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError as exc:
            return self._finalize_step(
                previous_scores=self.grader.column_scores(self.current_df),
                changed=False,
                updated_df=self.current_df.copy(deep=True),
                error=str(exc),
                grader_score=None,
            )

        previous_scores = self.grader.column_scores(self.current_df)

        if action_obj.action_type == ActionType.submit:
            submit_score = self.grader.score(self.current_df)
            reward_breakdown = RewardBreakdown(
                submit_bonus=(
                    self.config.submit_bonus_value
                    if submit_score >= self.config.submit_bonus_threshold
                    else 0.0
                )
            )
            reward_breakdown.total = self._clamp_reward(reward_breakdown.submit_bonus)

            self.step_number += 1
            self.episode_reward += reward_breakdown.total
            self.done = True

            info = StepInfo(
                action_applied=True,
                reward_breakdown=reward_breakdown,
                grader_score=submit_score,
                steps_taken=self.step_number,
                max_steps_reached=self.step_number >= self.max_steps,
            )
            observation = self.state()
            return StepResult(
                observation=observation,
                reward=reward_breakdown.total,
                done=True,
                info=info,
            )

        try:
            updated_df = self._apply_action(self.current_df, action_obj)
            changed = self._frame_changed(self.current_df, updated_df)
            error = None
        except Exception as exc:  # noqa: BLE001 - invalid actions should never crash the API
            updated_df = self.current_df.copy(deep=True)
            changed = False
            error = str(exc)

        return self._finalize_step(
            previous_scores=previous_scores,
            changed=changed,
            updated_df=updated_df,
            error=error,
            grader_score=None,
        )

    def _finalize_step(
        self,
        previous_scores: dict[str, float],
        changed: bool,
        updated_df: pd.DataFrame,
        error: str | None,
        grader_score: float | None,
    ) -> StepResult:
        """Commit the step result and compute reward shaping diagnostics."""
        self.current_df = updated_df
        new_scores = self.grader.column_scores(self.current_df)
        column_deltas = {
            column: round(new_scores.get(column, 0.0) - previous_scores.get(column, 0.0), 6)
            for column in previous_scores
        }
        reward_breakdown = RewardBreakdown(
            column_improvement=sum(column_deltas.values()),
            redundancy_penalty=0.0 if changed else self.config.no_op_penalty,
        )
        reward_breakdown.total = self._clamp_reward(
            reward_breakdown.column_improvement - reward_breakdown.redundancy_penalty
        )

        self.step_number += 1
        self.episode_reward += reward_breakdown.total

        max_steps_reached = self.step_number >= self.max_steps
        if max_steps_reached:
            self.done = True
            grader_score = self.grader.score(self.current_df)

        info = StepInfo(
            action_applied=changed,
            reward_breakdown=reward_breakdown,
            column_deltas=column_deltas,
            grader_score=grader_score,
            error=error,
            steps_taken=self.step_number if self.done else None,
            max_steps_reached=max_steps_reached,
        )

        observation = self.state()
        return StepResult(
            observation=observation,
            reward=reward_breakdown.total,
            done=observation.done,
            info=info,
        )

    def _apply_action(self, df: pd.DataFrame, action: Action) -> pd.DataFrame:
        """Return a new DataFrame with the action applied."""
        updated = df.copy(deep=True)
        params = dict(action.params or {})

        if action.action_type == ActionType.drop_duplicates:
            return updated.drop_duplicates(ignore_index=True)

        if action.column is None:
            raise ValueError(f"Action '{action.action_type.value}' requires a column.")

        if action.column not in updated.columns:
            raise ValueError(f"Column '{action.column}' does not exist.")

        if action.action_type == ActionType.fill_nulls:
            strategy = str(params["strategy"]).strip().lower()
            updated[action.column] = self._fill_nulls(updated[action.column], strategy, params)
            return updated

        if action.action_type == ActionType.cast_column:
            dtype_name = str(params.get("dtype", params.get("target_dtype"))).strip().lower()
            updated[action.column] = self._cast_column(updated[action.column], dtype_name)
            return updated

        if action.action_type == ActionType.remove_outliers:
            method = str(params["method"]).strip().lower()
            threshold = float(
                params.get("threshold", 1.5 if method == "iqr" else 3.0)
            )
            updated[action.column] = self._remove_outliers(updated[action.column], method, threshold)
            return updated

        if action.action_type == ActionType.rename_column:
            new_name = str(params["new_name"]).strip()
            if new_name in updated.columns and new_name != action.column:
                raise ValueError(f"Column '{new_name}' already exists.")
            return updated.rename(columns={action.column: new_name})

        if action.action_type == ActionType.normalize_values:
            mapping = params["mapping"]
            if not isinstance(mapping, dict):
                raise ValueError("normalize_values mapping must be a dictionary.")
            updated[action.column] = self._normalize_values(updated[action.column], mapping)
            return updated

        raise ValueError(f"Unsupported action_type '{action.action_type.value}'.")

    def _fill_nulls(
        self,
        series: pd.Series,
        strategy: str,
        params: dict[str, object],
    ) -> pd.Series:
        if strategy == "constant":
            fill_value = params["value"]
        elif strategy == "mode":
            mode = series.dropna().mode()
            if mode.empty:
                raise ValueError("Cannot compute mode for an all-null column.")
            fill_value = mode.iloc[0]
        else:
            numeric = self.grader._to_numeric(series)
            if numeric.dropna().empty:
                raise ValueError(f"Cannot compute {strategy} for a non-numeric column.")
            fill_value = float(numeric.mean()) if strategy == "mean" else float(numeric.median())

        return series.where(series.notna(), fill_value)

    def _cast_column(self, series: pd.Series, dtype_name: str) -> pd.Series:
        if dtype_name == "int":
            numeric = self.grader._to_numeric(series)
            rounded = numeric.round()
            return rounded.astype("Int64") if rounded.isna().any() else rounded.astype("int64")

        if dtype_name == "float":
            return self.grader._to_numeric(series).astype("float64")

        if dtype_name == "str":
            return series.astype("string")

        if dtype_name == "bool":
            normalized = self.grader._to_bool(series)
            return normalized.astype("boolean") if normalized.isna().any() else normalized.astype(bool)

        if dtype_name == "datetime":
            return self.grader._to_datetime(series)

        raise ValueError(f"Unsupported dtype '{dtype_name}'.")

    def _remove_outliers(self, series: pd.Series, method: str, threshold: float) -> pd.Series:
        numeric = self.grader._to_numeric(series)
        valid = numeric.dropna()
        if valid.empty:
            raise ValueError("Cannot remove outliers from an all-null column.")

        if method == "iqr":
            lower, upper = self.grader._iqr_bounds(valid, multiplier=threshold)
        elif method == "zscore":
            std = float(valid.std(ddof=0))
            if std <= 1e-9:
                return numeric
            mean = float(valid.mean())
            lower = mean - (threshold * std)
            upper = mean + (threshold * std)
        else:
            raise ValueError("remove_outliers method must be 'iqr' or 'zscore'.")

        return numeric.clip(lower=lower, upper=upper)

    @staticmethod
    def _normalize_values(series: pd.Series, mapping: dict[object, object]) -> pd.Series:
        cleaned_mapping = {str(key).strip(): value for key, value in mapping.items()}
        normalized = series.copy()
        non_null = series.notna()
        normalized.loc[non_null] = series.loc[non_null].map(
            lambda value: cleaned_mapping.get(str(value).strip(), value)
        )
        return normalized

    @staticmethod
    def _frame_changed(before: pd.DataFrame, after: pd.DataFrame) -> bool:
        return (
            list(before.columns) != list(after.columns)
            or not before.dtypes.equals(after.dtypes)
            or not before.equals(after)
        )

    def _clamp_reward(self, reward: float) -> float:
        return max(self.config.min_step_reward, min(self.config.max_step_reward, float(reward)))

    @staticmethod
    def _serialize_df(df: pd.DataFrame) -> list[dict[str, object | None]]:
        """Convert a DataFrame into JSON-friendly row dicts."""
        serializable = df.copy()
        for column in serializable.columns:
            if pd.api.types.is_datetime64_any_dtype(serializable[column]):
                serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")

        serializable = serializable.astype(object).where(pd.notna(serializable), None)
        return serializable.to_dict(orient="records")

===
"""Core stateful environment for the DataCleaning OpenEnv benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import ValidationError

from environment.graders import GRADER_REGISTRY
from environment.models import Action, ActionType, Observation, RewardBreakdown, StepInfo, StepResult
from environment.tasks import TASK_REGISTRY
from environment.tasks.base_task import BaseTask


@dataclass
class EnvironmentConfig:
    """Configurable runtime knobs for one environment instance."""

    submit_bonus_threshold: float = 0.80
    submit_bonus_value: float = 0.30
    no_op_penalty: float = 0.05
    min_step_reward: float = -0.05
    max_step_reward: float = 1.30


class DataCleaningEnv:
    """Interactive single-episode environment for one data-cleaning task."""

    def __init__(self, task_id: str, config: EnvironmentConfig | None = None) -> None:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {sorted(TASK_REGISTRY)}")

        self.task: BaseTask = TASK_REGISTRY[task_id]()
        self.grader = GRADER_REGISTRY[task_id]()
        self.config = config or EnvironmentConfig()

        self.current_df = pd.DataFrame()
        self.step_number = 0
        self.episode_reward = 0.0
        self.done = False
        self.reset()

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def max_steps(self) -> int:
        return int(self.task.max_steps)

    def reset(self) -> Observation:
        """Start a fresh episode for the configured task."""
        self.current_df = self.task.get_dirty_df().copy()
        self.step_number = 0
        self.episode_reward = 0.0
        self.done = False
        self.initial_similarity = self.grader.partial_score(self.current_df)
        return self.state()

    def state(self) -> Observation:
        """Return the current observable state."""
        return Observation(
            task_id=self.task_id,
            step_number=self.step_number,
            current_df=self._serialize_df(self.current_df),
            dirty_columns=self.grader.dirty_columns(self.current_df),
            columns_meta={column: str(dtype) for column, dtype in self.current_df.dtypes.items()},
            episode_reward_so_far=round(self.episode_reward, 6),
            done=self.done,
            max_steps=self.max_steps,
            task_description=self.task.description,
        )

    def step(self, action: Action | dict[str, object]) -> StepResult:
        """Apply one cleaning action and return the resulting transition."""
        if self.done:
            current_partial = self.grader.partial_score(self.current_df)
            info = StepInfo(
                error="Episode already completed.",
                steps_taken=self.step_number,
                grader_score=self.grader.score(self.current_df),
                max_steps_reached=self.step_number >= self.max_steps,
                partial_score=current_partial,
                dirty_columns_remaining=len(self.grader.dirty_columns(self.current_df)),
            )
            observation = self.state()
            return StepResult(observation=observation, reward=0.0, done=True, info=info)

        try:
            action_obj = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError as exc:
            return self._finalize_step(
                previous_scores=self.grader.column_scores(self.current_df),
                changed=False,
                updated_df=self.current_df.copy(deep=True),
                error=str(exc),
                grader_score=None,
            )

        previous_scores = self.grader.column_scores(self.current_df)

        if action_obj.action_type == ActionType.submit:
            submit_score = self.grader.score(self.current_df)
            column_scores = self.grader.column_scores(self.current_df)
            current_partial = self.grader.partial_score(self.current_df)
            reward_breakdown = RewardBreakdown(
                submit_bonus=(
                    self.config.submit_bonus_value
                    if submit_score >= self.config.submit_bonus_threshold
                    else 0.0
                )
            )
            reward_breakdown.total = self._clamp_reward(reward_breakdown.submit_bonus)

            self.step_number += 1
            self.episode_reward += reward_breakdown.total
            self.done = True

            info = StepInfo(
                action_applied=True,
                reward_breakdown=reward_breakdown,
                grader_score=submit_score,
                steps_taken=self.step_number,
                max_steps_reached=self.step_number >= self.max_steps,
                # Enriched — every step
                column_deltas={col: 0.0 for col in column_scores},
                partial_score=current_partial,
                dirty_columns_remaining=len(self.grader.dirty_columns(self.current_df)),
                # Enriched — submit only
                final_score=submit_score,
                grader_breakdown=column_scores,
                steps_used=self.step_number,
                steps_budget=self.max_steps,
                improvement_from_start=round(current_partial - self.initial_similarity, 6),
            )
            observation = self.state()
            return StepResult(
                observation=observation,
                reward=reward_breakdown.total,
                done=True,
                info=info,
            )

        try:
            updated_df = self._apply_action(self.current_df, action_obj)
            changed = self._frame_changed(self.current_df, updated_df)
            error = None
        except Exception as exc:  # noqa: BLE001 - invalid actions should never crash the API
            updated_df = self.current_df.copy(deep=True)
            changed = False
            error = str(exc)

        return self._finalize_step(
            previous_scores=previous_scores,
            changed=changed,
            updated_df=updated_df,
            error=error,
            grader_score=None,
        )

    def _finalize_step(
        self,
        previous_scores: dict[str, float],
        changed: bool,
        updated_df: pd.DataFrame,
        error: str | None,
        grader_score: float | None,
    ) -> StepResult:
        """Commit the step result and compute reward shaping diagnostics."""
        self.current_df = updated_df
        new_scores = self.grader.column_scores(self.current_df)
        column_deltas = {
            column: round(new_scores.get(column, 0.0) - previous_scores.get(column, 0.0), 6)
            for column in previous_scores
        }
        reward_breakdown = RewardBreakdown(
            column_improvement=sum(column_deltas.values()),
            redundancy_penalty=0.0 if changed else self.config.no_op_penalty,
        )
        reward_breakdown.total = self._clamp_reward(
            reward_breakdown.column_improvement - reward_breakdown.redundancy_penalty
        )

        self.step_number += 1
        self.episode_reward += reward_breakdown.total

        max_steps_reached = self.step_number >= self.max_steps
        if max_steps_reached:
            self.done = True
            grader_score = self.grader.score(self.current_df)

        current_partial = self.grader.partial_score(self.current_df)
        dirty_remaining = len(self.grader.dirty_columns(self.current_df))

        info = StepInfo(
            action_applied=changed,
            reward_breakdown=reward_breakdown,
            column_deltas=column_deltas,
            grader_score=grader_score,
            error=error,
            steps_taken=self.step_number if self.done else None,
            max_steps_reached=max_steps_reached,
            # Enriched — every step
            partial_score=current_partial,
            dirty_columns_remaining=dirty_remaining,
        )

        # If episode ended due to max_steps, also attach submit-like diagnostics
        if max_steps_reached:
            column_scores = self.grader.column_scores(self.current_df)
            info.final_score = grader_score
            info.grader_breakdown = column_scores
            info.steps_used = self.step_number
            info.steps_budget = self.max_steps
            info.improvement_from_start = round(
                current_partial - self.initial_similarity, 6
            )

        observation = self.state()
        return StepResult(
            observation=observation,
            reward=reward_breakdown.total,
            done=observation.done,
            info=info,
        )

    def _apply_action(self, df: pd.DataFrame, action: Action) -> pd.DataFrame:
        """Return a new DataFrame with the action applied."""
        updated = df.copy(deep=True)
        params = dict(action.params or {})

        if action.action_type == ActionType.drop_duplicates:
            return updated.drop_duplicates(ignore_index=True)

        if action.column is None:
            raise ValueError(f"Action '{action.action_type.value}' requires a column.")

        if action.column not in updated.columns:
            raise ValueError(f"Column '{action.column}' does not exist.")

        if action.action_type == ActionType.fill_nulls:
            strategy = str(params["strategy"]).strip().lower()
            updated[action.column] = self._fill_nulls(updated[action.column], strategy, params)
            return updated

        if action.action_type == ActionType.cast_column:
            dtype_name = str(params.get("dtype", params.get("target_dtype"))).strip().lower()
            updated[action.column] = self._cast_column(updated[action.column], dtype_name)
            return updated

        if action.action_type == ActionType.remove_outliers:
            method = str(params["method"]).strip().lower()
            threshold = float(
                params.get("threshold", 1.5 if method == "iqr" else 3.0)
            )
            updated[action.column] = self._remove_outliers(updated[action.column], method, threshold)
            return updated

        if action.action_type == ActionType.rename_column:
            new_name = str(params["new_name"]).strip()
            if new_name in updated.columns and new_name != action.column:
                raise ValueError(f"Column '{new_name}' already exists.")
            return updated.rename(columns={action.column: new_name})

        if action.action_type == ActionType.normalize_values:
            mapping = params["mapping"]
            if not isinstance(mapping, dict):
                raise ValueError("normalize_values mapping must be a dictionary.")
            updated[action.column] = self._normalize_values(updated[action.column], mapping)
            return updated

        raise ValueError(f"Unsupported action_type '{action.action_type.value}'.")

    def _fill_nulls(
        self,
        series: pd.Series,
        strategy: str,
        params: dict[str, object],
    ) -> pd.Series:
        if strategy == "constant":
            fill_value = params["value"]
        elif strategy == "mode":
            mode = series.dropna().mode()
            if mode.empty:
                raise ValueError("Cannot compute mode for an all-null column.")
            fill_value = mode.iloc[0]
        else:
            numeric = self.grader._to_numeric(series)
            if numeric.dropna().empty:
                raise ValueError(f"Cannot compute {strategy} for a non-numeric column.")
            fill_value = float(numeric.mean()) if strategy == "mean" else float(numeric.median())

        return series.where(series.notna(), fill_value)

    def _cast_column(self, series: pd.Series, dtype_name: str) -> pd.Series:
        if dtype_name == "int":
            numeric = self.grader._to_numeric(series)
            rounded = numeric.round()
            return rounded.astype("Int64") if rounded.isna().any() else rounded.astype("int64")

        if dtype_name == "float":
            return self.grader._to_numeric(series).astype("float64")

        if dtype_name == "str":
            return series.astype("string")

        if dtype_name == "bool":
            normalized = self.grader._to_bool(series)
            return normalized.astype("boolean") if normalized.isna().any() else normalized.astype(bool)

        if dtype_name == "datetime":
            return self.grader._to_datetime(series)

        raise ValueError(f"Unsupported dtype '{dtype_name}'.")

    def _remove_outliers(self, series: pd.Series, method: str, threshold: float) -> pd.Series:
        numeric = self.grader._to_numeric(series)
        valid = numeric.dropna()
        if valid.empty:
            raise ValueError("Cannot remove outliers from an all-null column.")

        if method == "iqr":
            lower, upper = self.grader._iqr_bounds(valid, multiplier=threshold)
        elif method == "zscore":
            std = float(valid.std(ddof=0))
            if std <= 1e-9:
                return numeric
            mean = float(valid.mean())
            lower = mean - (threshold * std)
            upper = mean + (threshold * std)
        else:
            raise ValueError("remove_outliers method must be 'iqr' or 'zscore'.")

        return numeric.clip(lower=lower, upper=upper)

    @staticmethod
    def _normalize_values(series: pd.Series, mapping: dict[object, object]) -> pd.Series:
        cleaned_mapping = {str(key).strip(): value for key, value in mapping.items()}
        normalized = series.copy()
        non_null = series.notna()
        normalized.loc[non_null] = series.loc[non_null].map(
            lambda value: cleaned_mapping.get(str(value).strip(), value)
        )
        return normalized

    @staticmethod
    def _frame_changed(before: pd.DataFrame, after: pd.DataFrame) -> bool:
        return (
            list(before.columns) != list(after.columns)
            or not before.dtypes.equals(after.dtypes)
            or not before.equals(after)
        )

    def _clamp_reward(self, reward: float) -> float:
        return max(self.config.min_step_reward, min(self.config.max_step_reward, float(reward)))

    @staticmethod
    def _serialize_df(df: pd.DataFrame) -> list[dict[str, object | None]]:
        """Convert a DataFrame into JSON-friendly row dicts."""
        serializable = df.copy()
        for column in serializable.columns:
            if pd.api.types.is_datetime64_any_dtype(serializable[column]):
                serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")

        serializable = serializable.astype(object).where(pd.notna(serializable), None)
        return serializable.to_dict(orient="records")

```

Key changes:
1. **`reset()`** — stores `self.initial_similarity` via `grader.partial_score()` for later `improvement_from_start` computation.
2. **Submit branch** — populates all enriched fields including `final_score`, `grader_breakdown` (per-column scores), `steps_used`, `steps_budget`, and `improvement_from_start`.
3. **`_finalize_step()`** — populates `partial_score` and `dirty_columns_remaining` on every step; also attaches submit-like diagnostics when the episode ends due to `max_steps` being reached.
4. **Already-done guard** — also includes `partial_score` and `dirty_columns_remaining` for consistency.

## Design Decisions

- **`grader_breakdown`** uses `grader.column_scores()` (per-column similarity dict) since the grader's `score()` method returns only a scalar. This gives agents the most useful per-column diagnostic.
- **Max-steps termination** gets the same submit-like fields as an explicit submit, so agents always see full diagnostics when an episode ends.
- All new `StepInfo` fields default to `None`, so existing code that doesn't read them is unaffected.

## Verification

- ✅ Smoke test: regular step has `partial_score`, `dirty_columns_remaining` populated; submit-only fields are `None`
- ✅ Smoke test: submit step has all fields populated with correct types
- ✅ Type assertions on all enriched fields passed
- ✅ Hard grader diagnostic (`grader_hard.py __main__`) ran cleanly
- ✅ Docker image rebuilt successfully
