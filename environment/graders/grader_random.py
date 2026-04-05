"""
Grader for the procedurally generated random task.

Uses the issue_log from RandomTask to know exactly what was injected
and scores each resolved issue equally.
"""

from __future__ import annotations

import pandas as pd

from environment.graders.base_grader import BaseGrader
from environment.tasks.task_random import RandomTask


class RandomGrader(BaseGrader):
    """
    Grades a randomly generated task using the issue_log from RandomTask
    to know exactly what was injected and what a correct fix looks like.
    """

    task_id = "random"

    def __init__(self, task: RandomTask | None = None) -> None:
        if task is None:
            task = RandomTask(seed=42)
        super().__init__(task)
        meta = task.get_metadata()
        self.issue_log: dict[str, dict] = meta.get("issue_log", {})
        self.domain_config: dict = task.domain

    # ── Main scoring ──────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> float:
        """Weight each injected issue equally: 1.0 / len(issue_log)."""
        if not self.issue_log:
            return self.partial_score(df)

        weight = 1.0 / len(self.issue_log)
        total = 0.0
        for issue_type, details in self.issue_log.items():
            total += weight * self._score_issue(df, issue_type, details)
        return self._clamp(total)

    def score_detailed(self, df: pd.DataFrame) -> tuple[float, dict]:
        """Return (overall_score, per-issue breakdown dict)."""
        if not self.issue_log:
            s = self.partial_score(df)
            return s, {}

        weight = 1.0 / len(self.issue_log)
        breakdown: dict[str, float] = {}
        for issue_type, details in self.issue_log.items():
            breakdown[issue_type] = round(self._score_issue(df, issue_type, details), 6)

        overall = self._clamp(sum(s * weight for s in breakdown.values()))
        return overall, breakdown

    # ── Per-issue scoring ─────────────────────────────────────────────

    def _score_issue(
        self, df: pd.DataFrame, issue_type: str, details: dict,
    ) -> float:
        col = details.get("column", "")

        # duplicate_rows is a row-level issue
        if issue_type == "duplicate_rows":
            truth_rows = len(self.truth_df)
            row_diff = abs(len(df) - truth_rows) / max(truth_rows, 1)
            row_score = self._clamp(1.0 - row_diff)
            no_dups = 1.0 if df.duplicated().sum() == 0 else 0.0
            return 0.5 * row_score + 0.5 * no_dups

        # Column-level issues — delegate to BaseGrader comparison helpers
        if col not in df.columns or col not in self.truth_df.columns:
            return 0.0

        current, truth = self._aligned_frames(df)
        if col not in current.columns:
            # Column may have been consumed as the alignment key.
            # Fall back to raw column comparison (length-matched).
            if col in df.columns and col in self.truth_df.columns:
                raw_cur = df[col].reset_index(drop=True)
                raw_tru = self.truth_df[col].reset_index(drop=True)
                min_len = min(len(raw_cur), len(raw_tru))
                return self._series_similarity(
                    raw_cur.iloc[:min_len], raw_tru.iloc[:min_len]
                )
            return 0.0

        return self._series_similarity(current[col], truth[col])
