"""Task registry for the Data Cleaning OpenEnv benchmark."""

from environment.tasks.task_easy import TASK_REGISTRY as _easy
from environment.tasks.task_medium import TASK_REGISTRY as _medium
from environment.tasks.task_hard import TASK_REGISTRY as _hard

TASK_REGISTRY: dict[str, type] = {**_easy, **_medium, **_hard}

__all__ = ["TASK_REGISTRY"]
