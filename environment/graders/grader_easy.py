"""Final grader for the easy employee-records task."""

from __future__ import annotations

import pandas as pd

from environment.graders.base_grader import BaseGrader
from environment.tasks.task_easy import EasyTask


class EasyGrader(BaseGrader):
    """Grader tuned to the easy task's duplicate and missing-value issues."""

    task_id = "easy"

    def __init__(self) -> None:
        super().__init__(EasyTask())

    @staticmethod
    def _age_dtype_ready(series: pd.Series) -> float:
        if pd.api.types.is_integer_dtype(series):
            return 1.0
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and ((numeric % 1) == 0).all():
            return 1.0
        return 0.0

    @staticmethod
    def _salary_dtype_ready(series: pd.Series) -> float:
        return 1.0 if pd.api.types.is_float_dtype(series) else 0.0

    def score(self, df: pd.DataFrame) -> float:
        age = self._to_numeric(self._series_from_column(df, "age"))
        salary = self._to_numeric(self._series_from_column(df, "salary"))

        duplicate_score = 1.0 if df.duplicated().sum() == 0 else 0.0
        completeness_score = float((age.notna() & salary.notna()).mean()) if len(df) else 0.0
        dtype_score = (
            self._age_dtype_ready(self._series_from_column(df, "age"))
            + self._salary_dtype_ready(self._series_from_column(df, "salary"))
        ) / 2.0
        range_score = self._fraction_valid(
            age.between(18, 65, inclusive="both")
            & salary.between(20_000, 200_000, inclusive="both")
        )

        total = (
            (0.35 * duplicate_score)
            + (0.30 * completeness_score)
            + (0.20 * dtype_score)
            + (0.15 * range_score)
        )
        return self._clamp(total)

