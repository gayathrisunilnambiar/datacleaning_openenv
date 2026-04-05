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

