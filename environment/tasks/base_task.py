"""
Abstract base class for all Data Cleaning OpenEnv tasks.

Every concrete task must subclass ``BaseTask`` and implement the three
abstract methods so the environment can load dirty data, ground truth,
and metadata in a uniform way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseTask(ABC):
    """Base contract that every cleaning task must satisfy.

    Attributes:
        task_id: Unique short identifier for the task (e.g. "easy").
        difficulty: Human-readable difficulty label ("easy" / "medium" / "hard").
        description: One-line summary of what makes this dataset dirty.
        max_steps: Maximum number of agent actions allowed in one episode.
    """

    task_id: str
    difficulty: str
    description: str
    max_steps: int

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dirty_df(self) -> pd.DataFrame:
        """Return the dirty DataFrame the agent will start with.

        The returned frame must be **deterministic** — identical across
        calls given the same seed — so that episodes are reproducible.
        """
        ...

    @abstractmethod
    def get_ground_truth_df(self) -> pd.DataFrame:
        """Return the clean, canonical DataFrame used for scoring.

        This is the target the agent's cleaning actions should converge
        toward.  It is never exposed to the agent directly.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return a metadata dictionary describing the task.

        Expected keys (at minimum):
            - task_id   (str)
            - difficulty (str)
            - description (str)
            - max_steps  (int)
            - column_types (dict[str, str]) — column name → expected dtype
            - num_rows  (int) — row count of the ground truth
            - num_cols  (int) — column count of the ground truth
            - issues    (list[str]) — short labels for each injected issue
        """
        ...


if __name__ == "__main__":
    # Quick sanity: cannot instantiate the ABC directly
    try:
        BaseTask()  # type: ignore[abstract]
    except TypeError as exc:
        print(f"[OK] BaseTask is abstract: {exc}")
