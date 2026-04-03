"""
Easy Task — Employee Records (50 rows × 6 columns).

Injected issues (all reproducible via seed 42):
  1. ~15 % of rows duplicated
  2. ~20 % of 'age' values set to NaN
  3. ~20 % of 'salary' values set to NaN
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from environment.tasks.base_task import BaseTask


class EasyTask(BaseTask):
    """Employee-records cleaning task (difficulty: easy).

    The ground truth is a tidy 50-row DataFrame with columns:
    id, name, age, department, salary, join_date.
    """

    task_id: str = "easy"
    difficulty: str = "easy"
    max_steps: int = 20
    description: str = (
        "Employee records with duplicate rows and missing values "
        "in the 'age' and 'salary' columns."
    )

    # ------------------------------------------------------------------
    # Internal: deterministic clean data generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_clean_df() -> pd.DataFrame:
        """Build the canonical 50-row employee DataFrame (seed 42)."""
        rng = np.random.RandomState(42)

        first_names = [
            "Alice", "Bob", "Carlos", "Diana", "Eve", "Frank",
            "Grace", "Hector", "Irene", "Jack", "Karen", "Leo",
            "Mia", "Nate", "Olivia", "Paul", "Quinn", "Rita",
            "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
            "Yara", "Zane", "Aiden", "Beth", "Caleb", "Donna",
            "Eli", "Faye", "Gus", "Holly", "Ivan", "Jill",
            "Kyle", "Luna", "Mike", "Nora", "Oscar", "Penny",
            "Reed", "Sara", "Troy", "Ursula", "Vince", "Willa",
            "Yusuf", "Zoe",
        ]
        departments = ["Engineering", "Sales", "Marketing", "HR", "Finance"]

        n = 50
        df = pd.DataFrame(
            {
                "id": list(range(1, n + 1)),
                "name": first_names[:n],
                "age": rng.randint(22, 60, size=n).tolist(),
                "department": rng.choice(departments, size=n).tolist(),
                "salary": rng.randint(35_000, 120_000, size=n).tolist(),
                "join_date": pd.date_range("2018-01-15", periods=n, freq="15D")
                .strftime("%Y-%m-%d")
                .tolist(),
            }
        )
        # Ensure deterministic dtypes
        df["age"] = df["age"].astype(float)
        df["salary"] = df["salary"].astype(float)
        return df

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_ground_truth_df(self) -> pd.DataFrame:
        """Return the clean employee DataFrame."""
        return self._build_clean_df()

    def get_dirty_df(self) -> pd.DataFrame:
        """Return the dirty version with duplicates and NaNs.

        Modifications applied (in order):
        1. Duplicate ~15 % of rows (chosen deterministically).
        2. Set ~20 % of 'age' cells to NaN.
        3. Set ~20 % of 'salary' cells to NaN.
        """
        rng = np.random.RandomState(42)
        df = self._build_clean_df()

        # --- 1. Duplicate rows (~15 %, i.e. ~7-8 rows) ----------------
        n_dup = int(len(df) * 0.15)
        dup_indices = rng.choice(df.index, size=n_dup, replace=False)
        dup_rows = df.loc[dup_indices]
        df = pd.concat([df, dup_rows], ignore_index=True)

        # --- 2. NaN in 'age' (~20 %) -----------------------------------
        n_age_nan = int(len(df) * 0.20)
        age_nan_idx = rng.choice(df.index, size=n_age_nan, replace=False)
        df.loc[age_nan_idx, "age"] = np.nan

        # --- 3. NaN in 'salary' (~20 %) --------------------------------
        n_sal_nan = int(len(df) * 0.20)
        sal_nan_idx = rng.choice(df.index, size=n_sal_nan, replace=False)
        df.loc[sal_nan_idx, "salary"] = np.nan

        return df

    def get_metadata(self) -> dict:
        """Return task metadata including column types and issue list."""
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
            "description": self.description,
            "column_types": {
                "id": "int64",
                "name": "object",
                "age": "float64",
                "department": "object",
                "salary": "float64",
                "join_date": "object",
            },
            "num_rows": 50,
            "num_cols": 6,
            "issues": [
                "duplicate_rows",
                "missing_age",
                "missing_salary",
            ],
        }


# ------------------------------------------------------------------
# Registry entry
# ------------------------------------------------------------------
TASK_REGISTRY: dict[str, type] = {"easy": EasyTask}


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    task = EasyTask()
    dirty = task.get_dirty_df()
    clean = task.get_ground_truth_df()
    meta = task.get_metadata()

    print("=== EASY TASK — Employee Records ===")
    print(f"Clean shape : {clean.shape}")
    print(f"Dirty shape : {dirty.shape}")
    print(f"Metadata    : {meta}")
    print()
    print("--- Clean head ---")
    print(clean.head(8).to_string(index=False))
    print()
    print("--- Dirty head ---")
    print(dirty.head(8).to_string(index=False))
    print()
    print(f"NaN counts in dirty:\n{dirty.isna().sum()}")
    print(f"Duplicate rows in dirty: {dirty.duplicated().sum()}")
