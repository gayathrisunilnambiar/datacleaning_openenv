"""Task-specific grader registry for DataCleaningEnv."""

from environment.graders.grader_easy import EasyGrader
from environment.graders.grader_hard import HardGrader
from environment.graders.grader_medium import MediumGrader
from environment.graders.grader_random import RandomGrader

GRADER_REGISTRY = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
    "random": RandomGrader,
}

__all__ = ["GRADER_REGISTRY", "EasyGrader", "MediumGrader", "HardGrader", "RandomGrader"]

