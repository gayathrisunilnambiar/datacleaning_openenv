"""Task-specific grader registry for DataCleaningEnv."""

from environment.graders.grader_easy import EasyGrader
from environment.graders.grader_hard import HardGrader
from environment.graders.grader_medium import MediumGrader

GRADER_REGISTRY = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}

__all__ = ["GRADER_REGISTRY", "EasyGrader", "MediumGrader", "HardGrader"]

