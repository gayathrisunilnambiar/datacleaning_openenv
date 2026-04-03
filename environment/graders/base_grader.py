"""Shared grading utilities for all DataCleaningEnv tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from environment.tasks.base_task import BaseTask


class BaseGrader(ABC):
    """Base grader with shared comparison helpers and fast partial scoring."""

    def __init__(self, task: BaseTask) -> None:
        self.task = task
        self.truth_df = task.get_ground_truth_df()
        self.metadata = task.get_metadata()

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _series_from_column(df: pd.DataFrame, column: str) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series(index=df.index, dtype="object")

    def _comparison_key(self) -> str:
        for column in self.truth_df.columns:
            if column == "id" or column.endswith("_id"):
                return column
        return self.truth_df.columns[0]

    def _aligned_frames(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align a candidate frame to ground truth for column-wise comparison."""
        truth = self.truth_df.copy()
        current = df.copy()
        key = self._comparison_key()

        if key in truth.columns and key in current.columns:
            truth = (
                truth.sort_values(key, kind="stable")
                .drop_duplicates(subset=[key], keep="first")
                .set_index(key)
            )
            current = (
                current.sort_values(key, kind="stable")
                .drop_duplicates(subset=[key], keep="first")
                .set_index(key)
            )
            current = current.reindex(truth.index)
        else:
            truth = truth.reset_index(drop=True)
            current = current.reset_index(drop=True).reindex(truth.index)

        return current, truth

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.strip()
        text = text.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        text = text.replace({"None": None, "nan": None, "NaN": None, "NaT": None})
        return pd.to_numeric(text, errors="coerce")

    @staticmethod
    def _to_datetime(series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors="coerce")
        text = series.astype("string").str.strip()
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

        for fmt, dayfirst in (
            ("%Y-%m-%d", False),
            ("%d/%m/%Y", True),
            ("%B %d %Y", False),
        ):
            remaining = parsed.isna() & text.notna()
            if not remaining.any():
                break
            parsed.loc[remaining] = pd.to_datetime(
                text.loc[remaining],
                format=fmt,
                errors="coerce",
                dayfirst=dayfirst,
            )

        remaining = parsed.isna() & text.notna()
        if remaining.any():
            parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce")
        return parsed

    @staticmethod
    def _to_bool(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.astype("boolean")

        text = series.astype(str).str.strip().str.lower()
        mapped = pd.Series(pd.NA, index=series.index, dtype="boolean")
        mapped.loc[text.isin(["true", "1", "yes", "y"])] = True
        mapped.loc[text.isin(["false", "0", "no", "n"])] = False
        return mapped

    @staticmethod
    def _is_datetime_series(series: pd.Series) -> bool:
        return pd.api.types.is_datetime64_any_dtype(series)

    @staticmethod
    def _is_bool_series(series: pd.Series) -> bool:
        return pd.api.types.is_bool_dtype(series)

    @staticmethod
    def _is_numeric_series(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

    def _numeric_similarity(self, current: pd.Series, truth: pd.Series) -> float:
        current_num = self._to_numeric(current)
        truth_num = self._to_numeric(truth)
        truth_valid = truth_num.notna()
        overlap = truth_valid & current_num.notna()

        if not truth_valid.any():
            return 1.0
        if not overlap.any():
            return 0.0

        value_range = float(truth_num[truth_valid].max() - truth_num[truth_valid].min())
        if value_range <= 1e-9:
            value_range = 1.0

        mae = float((current_num[overlap] - truth_num[overlap]).abs().mean())
        coverage = float(overlap.mean())
        return self._clamp(coverage * (1.0 - (mae / value_range)))

    def _datetime_similarity(self, current: pd.Series, truth: pd.Series) -> float:
        current_dt = self._to_datetime(current)
        truth_dt = self._to_datetime(truth)
        truth_valid = truth_dt.notna()

        if not truth_valid.any():
            return 1.0

        matches = pd.Series(False, index=truth.index, dtype=bool)
        overlap = truth_valid & current_dt.notna()
        matches.loc[overlap] = (
            (current_dt[overlap] - truth_dt[overlap]).abs() <= pd.Timedelta(days=1)
        )
        return self._clamp(float(matches.mean()))

    def _boolean_similarity(self, current: pd.Series, truth: pd.Series) -> float:
        current_bool = self._to_bool(current)
        truth_bool = self._to_bool(truth)
        valid = truth_bool.notna()

        if not valid.any():
            return 1.0

        matches = current_bool.eq(truth_bool) & current_bool.notna() & truth_bool.notna()
        return self._clamp(float(matches[valid].mean()))

    def _categorical_similarity(self, current: pd.Series, truth: pd.Series) -> float:
        current_text = current.where(current.notna(), "__missing__").astype(str)
        truth_text = truth.where(truth.notna(), "__missing__").astype(str)
        return self._clamp(float(current_text.eq(truth_text).mean()))

    def _series_similarity(self, current: pd.Series, truth: pd.Series) -> float:
        if self._is_bool_series(truth):
            return self._boolean_similarity(current, truth)
        # Treat datetime-cast columns as improved even when the ground truth is
        # stored as canonical ISO strings.
        if self._is_datetime_series(current) or self._is_datetime_series(truth):
            return self._datetime_similarity(current, truth)
        if self._is_numeric_series(truth):
            return self._numeric_similarity(current, truth)
        return self._categorical_similarity(current, truth)

    def column_scores(self, df: pd.DataFrame) -> dict[str, float]:
        """Return per-column similarity scores versus the ground truth."""
        current, truth = self._aligned_frames(df)
        scores: dict[str, float] = {}

        for column in truth.columns:
            if column not in current.columns:
                scores[column] = 0.0
                continue
            scores[column] = self._series_similarity(current[column], truth[column])

        return scores

    def dirty_columns(self, df: pd.DataFrame) -> list[str]:
        """Columns that still diverge materially from the hidden ground truth."""
        return [column for column, score in self.column_scores(df).items() if score < 0.999]

    def partial_score(self, df: pd.DataFrame) -> float:
        """Fast similarity score for reward shaping during the episode."""
        scores = self.column_scores(df)
        if not scores:
            return 0.0
        return self._clamp(sum(scores.values()) / len(scores))

    @staticmethod
    def _fraction_valid(values: pd.Series) -> float:
        if len(values) == 0:
            return 0.0
        return float(values.fillna(False).mean())

    @staticmethod
    def _iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        return q1 - (multiplier * iqr), q3 + (multiplier * iqr)

    @staticmethod
    def _iso_date_fraction(series: pd.Series) -> float:
        if pd.api.types.is_datetime64_any_dtype(series):
            return 1.0
        text = series.astype(str).str.strip()
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        return float(parsed.notna().mean()) if len(series) else 0.0

    @staticmethod
    def _canonical_gender(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.strip().str.lower()
        mapped = pd.Series(pd.NA, index=series.index, dtype="object")
        mapped.loc[text.isin(["m", "male", "1"])] = "Male"
        mapped.loc[text.isin(["f", "female", "0"])] = "Female"
        return mapped

    @staticmethod
    def _canonical_blood_type(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.strip().str.lower()
        mapping = {
            "a+": "A+",
            "a_pos": "A+",
            "apos": "A+",
            "a-": "A-",
            "a_neg": "A-",
            "aneg": "A-",
            "b+": "B+",
            "b_pos": "B+",
            "bpos": "B+",
            "b-": "B-",
            "b_neg": "B-",
            "bneg": "B-",
            "ab+": "AB+",
            "ab_pos": "AB+",
            "abpos": "AB+",
            "ab-": "AB-",
            "ab_neg": "AB-",
            "abneg": "AB-",
            "o+": "O+",
            "o_pos": "O+",
            "opos": "O+",
            "o-": "O-",
            "o_neg": "O-",
            "oneg": "O-",
        }
        return text.map(mapping)

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        """Return the final task score in the range [0.0, 1.0]."""
