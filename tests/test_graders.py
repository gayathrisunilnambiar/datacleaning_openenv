"""Regression tests for task graders."""

from __future__ import annotations

import unittest

from environment.graders.grader_easy import EasyGrader
from environment.graders.grader_hard import HardGrader
from environment.graders.grader_medium import MediumGrader


class GraderTests(unittest.TestCase):
    """Each grader should distinguish dirty from clean data."""

    def test_easy_grader_dirty_vs_clean(self) -> None:
        grader = EasyGrader()
        dirty_score = grader.score(grader.task.get_dirty_df())
        clean_score = grader.score(grader.task.get_ground_truth_df())
        self.assertGreaterEqual(dirty_score, 0.0)
        self.assertLessEqual(dirty_score, 1.0)
        self.assertLess(dirty_score, clean_score)
        self.assertAlmostEqual(clean_score, 1.0, places=6)

    def test_medium_grader_dirty_vs_clean(self) -> None:
        grader = MediumGrader()
        dirty_score = grader.score(grader.task.get_dirty_df())
        clean_score = grader.score(grader.task.get_ground_truth_df())
        partial_dirty = grader.partial_score(grader.task.get_dirty_df())
        partial_clean = grader.partial_score(grader.task.get_ground_truth_df())
        self.assertGreaterEqual(dirty_score, 0.0)
        self.assertLessEqual(dirty_score, 1.0)
        self.assertLess(dirty_score, clean_score)
        self.assertLess(partial_dirty, partial_clean)
        self.assertAlmostEqual(clean_score, 1.0, places=6)

    def test_hard_grader_dirty_vs_clean(self) -> None:
        grader = HardGrader()
        dirty_score = grader.score(grader.task.get_dirty_df())
        clean_score = grader.score(grader.task.get_ground_truth_df())
        self.assertGreaterEqual(dirty_score, 0.0)
        self.assertLessEqual(dirty_score, 1.0)
        self.assertLess(dirty_score, clean_score)
        self.assertAlmostEqual(clean_score, 1.0, places=6)

    def test_dirty_scores_follow_expected_difficulty_curve(self) -> None:
        easy_dirty = EasyGrader().score(EasyGrader().task.get_dirty_df())
        medium_dirty = MediumGrader().score(MediumGrader().task.get_dirty_df())
        hard_dirty = HardGrader().score(HardGrader().task.get_dirty_df())

        self.assertGreater(easy_dirty, medium_dirty)
        self.assertGreater(medium_dirty, hard_dirty)


if __name__ == "__main__":
    unittest.main()
