"""FastAPI surface for the DataCleaning OpenEnv benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from environment.env import DataCleaningEnv
from environment.graders import GRADER_REGISTRY
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


SESSION_TTL = timedelta(minutes=30)
APP_VERSION = "1.0.0"
APP_NAME = "data-cleaning-env"
APP_DESCRIPTION = "Deterministic multi-task benchmark for iterative tabular data cleaning agents."
APP_TAGS = ["openenv", "data-cleaning", "tabular", "benchmark"]


@dataclass
class SessionRecord:
    """Tracked in-memory state for one active session."""

    env: DataCleaningEnv
    last_accessed: datetime


class RootResponse(BaseModel):
    """Top-level API directory returned by GET /."""

    name: str
    version: str
    description: str
    status: str
    endpoints: dict[str, str]
    baseline_scores: dict[str, float]


app = FastAPI(title="DataCleaningEnv", version=APP_VERSION)
_SESSIONS: dict[str, SessionRecord] = {}


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
        },
        baseline_scores={
            "easy": 1.0000,
            "medium": 0.9811,
            "hard": 0.8332,
        },
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _prune_sessions() -> None:
    cutoff = _utc_now() - SESSION_TTL
    expired = [session_id for session_id, record in _SESSIONS.items() if record.last_accessed < cutoff]
    for session_id in expired:
        _SESSIONS.pop(session_id, None)


def _get_session(session_id: str) -> SessionRecord:
    _prune_sessions()
    record = _SESSIONS.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'.")
    record.last_accessed = _utc_now()
    return record


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness endpoint for local and deployed health checks."""
    return HealthResponse(version=APP_VERSION)


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


@app.post("/reset", response_model=ResetResponse)
def reset(payload: ResetRequest) -> ResetResponse:
    """Create a new environment session and return the initial observation."""
    env = DataCleaningEnv(payload.task_id)
    session_id = payload.session_id or str(uuid4())
    _SESSIONS[session_id] = SessionRecord(env=env, last_accessed=_utc_now())
    return ResetResponse(session_id=session_id, observation=env.state())


@app.get("/state", response_model=StateResponse)
def state(session_id: str = Query(..., description="Existing environment session id")) -> StateResponse:
    """Return the current state snapshot for an existing session."""
    record = _get_session(session_id)
    return StateResponse(session_id=session_id, observation=record.env.state())


@app.post("/step", response_model=StepResult)
def step(payload: StepRequest) -> StepResult:
    """Apply one action to an active environment session."""
    record = _get_session(payload.session_id)
    return record.env.step(payload.action)


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
