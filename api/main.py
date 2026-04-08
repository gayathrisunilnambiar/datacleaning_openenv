"""FastAPI surface for the DataCleaning OpenEnv benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel

from environment.env import DataCleaningEnv
from environment.graders import GRADER_REGISTRY
from environment.logging_config import _clear_request_context, bind_request_context, get_logger
from environment.models import (
    HealthResponse,
    JsonRpcResponse,
    MetadataResponse,
    Observation,
    ResetRequest,
    ResetResponse,
    Action,
    SchemaResponse,
    StateResponse,
    StepRequest,
    StepResult,
    TaskInfoModel,
    ValidateChecks,
    ValidateResponse,
)
from environment.tasks import TASK_REGISTRY

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # default 1 hour
CLEANUP_INTERVAL_SECONDS = 300  # run cleanup every 5 minutes

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # empty = disabled
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")  # optional HMAC signing key
WEBHOOK_TIMEOUT_SECONDS = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "5.0"))

APP_VERSION = "1.0.0"
APP_NAME = "data-cleaning-env"
APP_DESCRIPTION = "Deterministic multi-task benchmark for iterative tabular data cleaning agents."
APP_TAGS = ["openenv", "data-cleaning", "tabular", "benchmark"]

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

# Read once at startup — None means auth is disabled
_API_KEY: str | None = os.getenv("API_KEY")


def verify_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional API key guard.

    Behaviour:
    - API_KEY env var NOT set → all requests pass through freely.
      This is the correct mode for the HF Space competition deployment.
    - API_KEY env var IS set → requests must include the matching
      X-API-Key header or receive HTTP 401.
    - Never returns 401 when _API_KEY is None.
    """
    if _API_KEY is not None and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------


@dataclass
class SessionRecord:
    """Tracked in-memory state for one active session."""

    env: DataCleaningEnv
    last_accessed: float = field(default_factory=time.time)


_SESSIONS: dict[str, SessionRecord] = {}


def _prune_sessions() -> None:
    """Remove sessions that have been inactive longer than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, record in _SESSIONS.items()
        if now - record.last_accessed > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        record = _SESSIONS.pop(session_id, None)
        if record:
            log.info(
                "session_expired",
                session_id=session_id,
                task_id=record.env.task_id,
                steps_completed=record.env.step_number,
                final_step_reward=round(record.env.episode_reward, 4),
            )


def _get_session(session_id: str) -> SessionRecord:
    """Look up a session, pruning stale ones first.

    Raises HTTPException 404 if the session does not exist or has expired.
    """
    _prune_sessions()
    record = _SESSIONS.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Episode not found or expired")
    record.last_accessed = time.time()
    return record


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------


async def _cleanup_loop() -> None:
    """Periodically prune expired sessions without blocking the event loop."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        _prune_sessions()


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    """FastAPI lifespan: start the background cleanup task on startup."""
    # Log auth mode at startup
    if _API_KEY is None:
        log.info("auth_mode", status="disabled", note="Set API_KEY env var to enable")
    else:
        log.info("auth_mode", status="enabled", protected_routes=["/admin/sessions", "/submit"])

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


async def fire_webhook(payload: dict) -> None:
    """POST episode results to the configured webhook URL.

    Runs as a background task — any failure is logged and swallowed so the
    caller's HTTP response is never affected.
    """
    if not WEBHOOK_URL:
        return  # feature disabled

    body_bytes = json.dumps(payload, default=str).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if WEBHOOK_SECRET:
        signature = hmac.new(
            WEBHOOK_SECRET.encode(),
            body_bytes,
            hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
            response = await client.post(WEBHOOK_URL, content=body_bytes, headers=headers)
            response.raise_for_status()
    except Exception:  # noqa: BLE001 — must never propagate
        log.warning(
            "webhook_failed",
            url=WEBHOOK_URL,
            episode_id=payload.get("episode_id"),
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class RootResponse(BaseModel):
    """Top-level API directory returned by GET /."""

    name: str
    version: str
    description: str
    status: str
    endpoints: dict[str, str]
    baseline_scores: dict[str, float]


app = FastAPI(title="DataCleaningEnv", version=APP_VERSION, lifespan=_lifespan)


@app.middleware("http")
async def clear_logging_context(request: Request, call_next):
    """Ensure request-scoped log context is cleared before and after each request."""
    _clear_request_context()
    try:
        response = await call_next(request)
    finally:
        _clear_request_context()
    return response


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC routes (no auth)
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/", response_model=RootResponse, summary="API Directory", tags=["info"])
def root() -> RootResponse:
    """Return a clean API directory with every available endpoint."""
    return RootResponse(
        name=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        status="ok",
        endpoints={
            "root": "GET /",
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "tasks": "GET /tasks",
            "state": "GET /state",
            "reset": "POST /reset",
            "step": "POST /step",
            "mcp": "POST /mcp",
            "validate": "POST /validate",
            "admin_sessions": "GET /admin/sessions",
        },
        baseline_scores={
            "easy": 1.0000,
            "medium": 0.9811,
            "hard": 0.8332,
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness endpoint for local and deployed health checks."""
    log.info("health_check")
    return HealthResponse(version=APP_VERSION, auth_enabled=_API_KEY is not None)


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return top-level metadata for OpenEnv validators and registry UIs."""
    return MetadataResponse(
        name=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        tags=APP_TAGS,
    )


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    """Expose JSON schemas for the core HTTP contracts."""
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state=StateResponse.model_json_schema(),
    )


@app.get("/tasks", response_model=list[TaskInfoModel])
def tasks() -> list[TaskInfoModel]:
    """List all available tasks with minimal metadata."""
    task_models: list[TaskInfoModel] = []
    for task_id in sorted(TASK_REGISTRY):
        if task_id == "random":
            # Stable listing — avoid generating data just for metadata
            task_models.append(
                TaskInfoModel(
                    task_id="random",
                    difficulty="variable",
                    description=(
                        "Procedurally generated task with random domain and "
                        "data quality issues. Unique each episode; pass a seed "
                        "for reproducibility."
                    ),
                    max_steps=25,
                )
            )
        else:
            task = TASK_REGISTRY[task_id]()
            task_models.append(
                TaskInfoModel(
                    task_id=task.task_id,
                    difficulty=task.difficulty,
                    description=task.description,
                    max_steps=task.max_steps,
                )
            )
    return task_models


# ──────────────────────────────────────────────────────────────────────────────
# Environment routes (public — competition validator hits these without a key)
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=ResetResponse)
def reset(payload: ResetRequest) -> ResetResponse:
    """Create a new environment session and return the initial observation."""
    session_id = payload.session_id or str(uuid4())
    bind_request_context(session_id=session_id, task_id=payload.task_id)
    if payload.task_id not in TASK_REGISTRY:
        log.warning("unknown_task", task_id=payload.task_id)
    env = DataCleaningEnv(payload.task_id, seed=payload.seed)
    _SESSIONS[session_id] = SessionRecord(env=env, last_accessed=time.time())
    log.info(
        "episode_started",
        max_steps=env.max_steps,
        dirty_calibration=round(env.grader.partial_score(env.task.get_dirty_df()), 4),
    )
    return ResetResponse(session_id=session_id, observation=env.state())


@app.get("/state", response_model=StateResponse)
def state(session_id: str = Query(..., description="Existing environment session id")) -> StateResponse:
    """Return the current state snapshot for an existing session."""
    record = _get_session(session_id)
    bind_request_context(
        session_id=session_id,
        task_id=record.env.task_id,
        step_number=record.env.step_number,
    )
    log.info("state_inspected")
    return StateResponse(session_id=session_id, observation=record.env.state())


@app.post("/step", response_model=StepResult)
def step(payload: StepRequest) -> StepResult:
    """Apply one action to an active environment session."""
    record = _get_session(payload.session_id)
    bind_request_context(
        session_id=payload.session_id,
        task_id=record.env.task_id,
        step_number=record.env.step_number + 1,
    )
    result = record.env.step(payload.action)
    log.info(
        "step_taken",
        action=payload.action.action_type.value,
        reward=round(result.reward, 4),
        done=result.done,
        dirty_remaining=result.info.dirty_columns_remaining,
    )
    return result


@app.post("/mcp", response_model=JsonRpcResponse)
def mcp(payload: dict[str, object] | None = None) -> JsonRpcResponse:
    """Minimal JSON-RPC shim so validators can detect MCP reachability."""
    request_id = None
    if isinstance(payload, dict):
        candidate = payload.get("id")
        if isinstance(candidate, (str, int)) or candidate is None:
            request_id = candidate

    return JsonRpcResponse(
        id=request_id,
        result={
            "name": APP_NAME,
            "transport": "http",
            "message": "MCP shim reachable. Use the HTTP environment endpoints for interaction.",
        },
    )


@app.post("/validate", response_model=ValidateResponse)
def validate() -> ValidateResponse:
    """Run an internal deterministic self-check without ever returning HTTP 500."""
    log.info("validate_requested")
    checks = ValidateChecks()

    try:
        env = DataCleaningEnv("easy")
        checks.reset = True

        scripted_actions = [
            {"action_type": "drop_duplicates"},
            {"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}},
            {"action_type": "fill_nulls", "column": "salary", "params": {"strategy": "median"}},
            {"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}},
            {"action_type": "submit"},
        ]

        for index, action in enumerate(scripted_actions, start=1):
            result = env.step(action)
            if index == 1:
                checks.state = bool(env.state().current_df)
            checks.step = isinstance(result, StepResult)
    except Exception as exc:  # noqa: BLE001 - validation must return HTTP 200 on failure
        checks.errors.append(f"episode_check_failed: {exc}")

    try:
        for task_id, grader_cls in GRADER_REGISTRY.items():
            grader = grader_cls()
            dirty_score = grader.score(grader.task.get_dirty_df())
            checks.graders[task_id] = dirty_score
            if not 0.0 <= dirty_score <= 1.0:
                checks.errors.append(f"grader_out_of_range:{task_id}:{dirty_score}")
    except Exception as exc:  # noqa: BLE001
        checks.errors.append(f"grader_check_failed: {exc}")

    passed = (
        checks.reset
        and checks.step
        and checks.state
        and all(0.0 <= score <= 1.0 for score in checks.graders.values())
        and not checks.errors
    )
    return ValidateResponse(passed=passed, checks=checks)


@app.post("/submit", response_model=StepResult, dependencies=[Depends(verify_key)])
def submit(payload: StepRequest, background_tasks: BackgroundTasks) -> StepResult:
    """Convenience endpoint that forces a submit action."""
    record = _get_session(payload.session_id)
    bind_request_context(
        session_id=payload.session_id,
        task_id=record.env.task_id,
        step_number=record.env.step_number + 1,
    )
    submit_action = Action(action_type="submit")
    result = record.env.step(submit_action)
    log.info(
        "episode_submitted",
        final_score=result.info.final_score,
        steps_used=result.info.steps_used,
        steps_budget=result.info.steps_budget,
        improvement=result.info.improvement_from_start,
    )

    # Fire webhook in the background (non-blocking)
    webhook_payload = {
        "event": "episode_complete",
        "episode_id": payload.session_id,
        "task_id": result.observation.task_id,
        "final_score": result.info.final_score,
        "steps_used": result.info.steps_used,
        "grader_breakdown": result.info.grader_breakdown,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    background_tasks.add_task(fire_webhook, webhook_payload)

    return result


@app.get("/admin/sessions", dependencies=[Depends(verify_key)])
def admin_sessions() -> dict:
    """Return a summary of active sessions (admin/debug use)."""
    _prune_sessions()
    now = time.time()
    ages = [now - rec.last_accessed for rec in _SESSIONS.values()]
    return {
        "active_sessions": len(_SESSIONS),
        "oldest_session_age_seconds": round(max(ages), 2) if ages else 0.0,
    }
