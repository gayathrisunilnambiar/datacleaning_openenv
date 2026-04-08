"""Structured logging configuration for the DataCleaning OpenEnv benchmark.

Uses structlog with JSON rendering to stdout. All modules should obtain
their logger via ``get_logger(__name__)``.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys

import structlog


_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id",
    default=None,
)
_task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "task_id",
    default=None,
)
_step_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "step_number",
    default=None,
)


class _StdoutProxy:
    """Delegate writes to the current ``sys.stdout`` so tests can patch it."""

    def write(self, message: str) -> int:
        return sys.stdout.write(message)

    def flush(self) -> None:
        sys.stdout.flush()


def bind_request_context(
    session_id: str | None = None,
    task_id: str | None = None,
    step_number: int | None = None,
) -> None:
    """Call at the start of any handler that has session context."""
    if session_id is not None:
        _session_id_var.set(session_id)
    if task_id is not None:
        _task_id_var.set(task_id)
    if step_number is not None:
        _step_var.set(step_number)


def _clear_request_context() -> None:
    """Reset the current request-scoped logging context."""
    _session_id_var.set(None)
    _task_id_var.set(None)
    _step_var.set(None)


def _inject_context(
    logger: object,
    method: str,
    event_dict: dict[str, object],
) -> dict[str, object]:
    """
    structlog processor - injects context vars into every log line.
    Runs automatically as part of the processor chain.
    Only adds keys that are not None so clean events stay clean.
    """
    del logger, method

    session_id = _session_id_var.get()
    task_id = _task_id_var.get()
    step_number = _step_var.get()

    if session_id is not None:
        event_dict.setdefault("session_id", session_id)
    if task_id is not None:
        event_dict.setdefault("task_id", task_id)
    if step_number is not None:
        event_dict.setdefault("step", step_number)

    return event_dict


def configure_logging() -> None:
    """One-time structlog + stdlib logging setup.

    Call once at process startup (import-time of this module handles it).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Configure structlog processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _inject_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up stdlib root logger with a structlog-powered formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
        if os.getenv("LOG_FORMAT", "json").lower() == "console"
        else structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(_StdoutProxy())
    handler.setFormatter(formatter)
    handler._openenv_structlog_handler = True  # type: ignore[attr-defined]

    root_logger = logging.getLogger()
    # Avoid duplicate handlers on repeated calls
    if not any(getattr(existing, "_openenv_structlog_handler", False) for existing in root_logger.handlers):
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


# Auto-configure on first import
configure_logging()


__all__ = ["bind_request_context", "get_logger"]


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a structlog-wrapped logger bound to the given module name."""
    return structlog.get_logger(name)
