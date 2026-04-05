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
    seed: int | None = None


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
