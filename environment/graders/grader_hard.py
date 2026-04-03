"""Final grader for the hard hospital-admissions task."""

from __future__ import annotations

import pandas as pd

from environment.graders.base_grader import BaseGrader
from environment.tasks.task_hard import HardTask


class HardGrader(BaseGrader):
    """Grader for multi-issue healthcare cleaning with mixed units and labels."""

    task_id = "hard"

    def __init__(self) -> None:
        super().__init__(HardTask())

    @staticmethod
    def _age_dtype_ready(series: pd.Series) -> float:
        if pd.api.types.is_integer_dtype(series):
            return 1.0
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and ((numeric % 1) == 0).all():
            return 1.0
        return 0.0

    def score(self, df: pd.DataFrame) -> float:
        scores = self.column_scores(df)

        row_count_score = 1.0 if len(df) == len(self.truth_df) else 0.0

        age_raw = self._series_from_column(df, "age")
        age_alignment_score = scores.get("age", 0.0)
        age_dtype_score = self._age_dtype_ready(age_raw)
        age_score = (0.5 * age_alignment_score) + (0.5 * age_dtype_score)

        admission_series = self._series_from_column(df, "admission_date")
        admission_score = (
            1.0 if pd.api.types.is_datetime64_any_dtype(admission_series) else self._iso_date_fraction(admission_series)
        )

        gender_raw = self._series_from_column(df, "gender").astype(str).str.strip()
        gender_score = self._fraction_valid(gender_raw.isin(["Male", "Female"]))

        blood_type_raw = self._series_from_column(df, "blood_type").astype(str).str.strip()
        blood_type_score = self._fraction_valid(
            blood_type_raw.isin(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        )

        readmitted_raw = self._series_from_column(df, "readmitted")
        readmitted_score = 1.0 if pd.api.types.is_bool_dtype(readmitted_raw) else 0.0

        total_score = (
            (0.15 * row_count_score)
            + (0.10 * age_score)
            + (0.20 * admission_score)
            + (0.15 * gender_score)
            + (0.10 * scores.get("weight_kg", 0.0))
            + (0.15 * blood_type_score)
            + (0.15 * readmitted_score)
        )
        return self._clamp(total_score)
