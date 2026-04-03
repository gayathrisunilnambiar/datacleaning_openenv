"""Final grader for the medium sales-transactions task."""

from __future__ import annotations

import pandas as pd

from environment.graders.base_grader import BaseGrader
from environment.tasks.task_medium import MediumTask


class MediumGrader(BaseGrader):
    """Grader focused on type normalization, date cleanup, and outlier repair."""

    task_id = "medium"

    def __init__(self) -> None:
        super().__init__(MediumTask())

    @staticmethod
    def _float_dtype_ready(series: pd.Series) -> float:
        return 1.0 if pd.api.types.is_float_dtype(series) else 0.0

    def score(self, df: pd.DataFrame) -> float:
        scores = self.column_scores(df)

        quantity_dtype_score = self._float_dtype_ready(self._series_from_column(df, "quantity"))
        unit_price_dtype_score = self._float_dtype_ready(self._series_from_column(df, "unit_price"))

        date_series = self._series_from_column(df, "date")
        date_score = 1.0 if pd.api.types.is_datetime64_any_dtype(date_series) else self._iso_date_fraction(date_series)

        total_score = (
            (0.25 * quantity_dtype_score)
            + (0.25 * unit_price_dtype_score)
            + (0.20 * date_score)
            + (0.15 * scores.get("unit_price", 0.0))
            + (0.15 * scores.get("total", 0.0))
        )
        return self._clamp(total_score)
