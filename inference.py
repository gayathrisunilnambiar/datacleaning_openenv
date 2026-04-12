"""Submission-style baseline runner for DataCleaningEnv.

Required environment variables:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")
    LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from typing import Any

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "data-cleaning-env"
TASK_ORDER = ["easy", "medium", "hard"]
REQUEST_TIMEOUT_SECONDS = 15
MAX_LOGICAL_STEPS = 120
SUCCESS_SCORE_THRESHOLD = 0.80

# Hard-coded schema hints per task — API Observation does not include schema_hint.
TASK_SCHEMA_HINTS: dict[str, dict[str, str]] = {
    "easy": {
        "id":         "int, range 1-50, not nullable",
        "name":       "str, not nullable",
        "age":        "int, range 18-65, not nullable",
        "department": "str, allowed: HR/Engineering/Sales/Marketing/Finance",
        "salary":     "float, range 20000-200000, not nullable",
        "join_date":  "str, ISO date YYYY-MM-DD, not nullable",
    },
    "medium": {
        "txn_id":     "int, not nullable",
        "date":       "str, ISO format YYYY-MM-DD, not nullable",
        "product":    "str, not nullable",
        "category":   "str, not nullable",
        "quantity":   "int, range 1-1000, not nullable",
        "unit_price": "float, range 0.01-10000, not nullable",
        "total":      "float, must equal quantity*unit_price, not nullable",
        "region":     "str, not nullable",
    },
    "hard": {
        "patient_id":     "int, not nullable",
        "admission_date": "datetime, ISO format, not nullable",
        "discharge_date": "datetime, ISO format, not nullable",
        "diagnosis":      "str, not nullable",
        "ward":           "str, not nullable",
        "age":            "int, range 18-90, not nullable, use median strategy",
        "gender":         "str, allowed: Male/Female ONLY, not nullable",
        "weight_kg":      "float, range 30-200 kg, not nullable — values >200 are in lbs",
        "blood_type":     "str, allowed: A+/A-/B+/B-/O+/O-/AB+/AB- ONLY, not nullable",
        "readmitted":     "bool, True/False only, not nullable",
    },
}

SYSTEM_PROMPT = """
You are a precise data cleaning agent operating inside DataCleaningEnv.
Your goal is to maximize the grader score (0.0-1.0) by issuing one
cleaning action per turn. You have a fixed step budget. Every wasted
step costs -0.05 in reward.

AVAILABLE ACTIONS (one per turn, raw JSON only):
{"action_type": "drop_duplicates"}
{"action_type": "fill_nulls",       "column": "<col>", "params": {"strategy": "mean|median|mode|constant", "value": <val>}}
{"action_type": "cast_column",      "column": "<col>", "params": {"dtype": "int|float|str|bool|datetime"}}
{"action_type": "remove_outliers",  "column": "<col>", "params": {"method": "iqr|zscore"}}
{"action_type": "rename_column",    "column": "<col>", "params": {"new_name": "<name>"}}
{"action_type": "normalize_values", "column": "<col>", "params": {"mapping": {"<old>": "<new>"}}}
{"action_type": "submit"}

REWARD SIGNALS:
  +column_improvement  each column closer to ground truth
  +0.30 submit bonus   if overall similarity > 0.80 on submit
  -0.05 no-op penalty  if action caused zero change

REASONING PROTOCOL (follow in order every turn):

STEP 1 - Check dirty_columns.
  If empty → output {"action_type": "submit"} immediately.

STEP 2 - Check budget_remaining.
  If <= 2 → output {"action_type": "submit"} immediately.

STEP 3 - Check last action.
  If action_applied == false → action was no-op.
  Never repeat same action_type + column combination.

STEP 4 - Check zero_reward_actions list.
  Never repeat any action listed there.

STEP 5 - Prioritize by column_deltas.
  Columns with positive deltas last turn are responding.
  Focus on those first.

STEP 6 - Use schema_hint to choose action per dirty column:
  dtype mismatch              → cast_column first
  nullable: false + nulls     → fill_nulls
  allowed_values violation    → normalize_values with FULL mapping
  range violation (outliers)  → remove_outliers
  weight_kg values > 200      → those are lbs values (×2.20462 to convert)
                                remove_outliers(iqr) clips them but
                                does not convert — check reward after

STEP 7 - Output ONE raw JSON object. No prose. No markdown.

NORMALIZATION REFERENCE (use these mappings verbatim):

gender:
{"M":"Male","m":"Male","male":"Male","1":"Male",
 "F":"Female","f":"Female","female":"Female","0":"Female"}

blood_type:
{"A_pos":"A+","Apos":"A+","a+":"A+",
 "A_neg":"A-","Aneg":"A-","a-":"A-",
 "B_pos":"B+","Bpos":"B+","b+":"B+",
 "B_neg":"B-","Bneg":"B-","b-":"B-",
 "AB_pos":"AB+","ABpos":"AB+","ab+":"AB+",
 "AB_neg":"AB-","ABneg":"AB-","ab-":"AB-",
 "O_pos":"O+","Opos":"O+","o+":"O+",
 "O_neg":"O-","Oneg":"O-","o-":"O-"}

readmitted:
  Use cast_column(readmitted, bool) — handles yes/no/True/False/1/0.
  Do not use normalize_values for this column.

OUTPUT RULES:
- One JSON object per turn, always
- No text before or after the JSON
- No markdown fences
- No comments inside JSON
- If uncertain → {"action_type": "submit"}
"""


def _sanitize_single_line(value: object | None) -> str:
    """Render values as one-line log-safe strings."""
    if value is None:
        return "null"
    return str(value).replace("\r", " ").replace("\n", "\\n")


def _action_to_log_string(action: dict[str, Any] | None) -> str:
    """Render an action as compact single-line JSON."""
    if action is None:
        return "null"
    return json.dumps(action, separators=(",", ":"), default=str)


def _clamp_score(value: float | None) -> float:
    """Clamp scores into the expected [0, 1] range."""
    if value is None:
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def log_start(task: str, env: str, model: str) -> None:
    """Emit the required START line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict[str, Any] | None, reward: float, done: bool, error: str | None) -> None:
    """Emit the required STEP line."""
    print(
        f"[STEP] step={step} action={_action_to_log_string(action)} "
        f"reward={reward:.2f} done={str(done).lower()} error={_sanitize_single_line(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit the required END line."""
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={_clamp_score(score):.2f} rewards={rewards_str}",
        flush=True,
    )


class DockerEnvHandle:
    """Small helper for running a local environment image during inference."""

    def __init__(self, image_name: str) -> None:
        self.image_name = image_name
        self.container_name: str | None = None
        self.base_url: str | None = None

    def start(self) -> str:
        """Start the local Docker image and wait for /health."""
        port = _free_port()
        self.container_name = f"data-cleaning-env-{uuid.uuid4().hex[:8]}"
        self.base_url = f"http://127.0.0.1:{port}"

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                self.container_name,
                "-p",
                f"{port}:7860",
                self.image_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return self.base_url
            except requests.RequestException:
                pass
            time.sleep(1)

        self.close()
        raise RuntimeError(f"Timed out waiting for local image '{self.image_name}' to become ready.")

    def close(self) -> None:
        """Stop the temporary container if one was started."""
        if not self.container_name:
            return
        subprocess.run(
            ["docker", "stop", self.container_name],
            check=False,
            capture_output=True,
            text=True,
        )
        self.container_name = None


def _free_port() -> int:
    """Find a currently free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def column_has_nulls(observation: dict[str, Any], column: str) -> bool:
    """Check whether a serialized observation still has nulls in a given column."""
    for row in observation.get("current_df", []):
        if row.get(column) is None:
            return True
    return False


def dry_run_action(task_id: str, observation: dict[str, Any]) -> dict[str, Any]:
    """Simple deterministic heuristic baseline that never calls an LLM."""
    dirty = set(observation.get("dirty_columns", []))
    columns_meta = observation.get("columns_meta", {})
    step_number = int(observation.get("step_number", 0))

    if task_id == "easy":
        if observation.get("step_number", 0) == 0:
            return {"action_type": "drop_duplicates"}
        if "age" in dirty and column_has_nulls(observation, "age"):
            return {
                "action_type": "fill_nulls",
                "column": "age",
                "params": {"strategy": "median"},
            }
        if "salary" in dirty and column_has_nulls(observation, "salary"):
            return {
                "action_type": "fill_nulls",
                "column": "salary",
                "params": {"strategy": "median"},
            }
        if "age" in dirty and columns_meta.get("age") != "int64":
            return {"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}}
        return {"action_type": "submit"}

    if task_id == "medium":
        if step_number == 0 and columns_meta.get("quantity") != "float64":
            return {"action_type": "cast_column", "column": "quantity", "params": {"dtype": "float"}}
        if step_number <= 1 and columns_meta.get("unit_price") != "float64":
            return {
                "action_type": "cast_column",
                "column": "unit_price",
                "params": {"dtype": "float"},
            }
        if step_number <= 2 and not str(columns_meta.get("date", "")).startswith("datetime64"):
            return {"action_type": "cast_column", "column": "date", "params": {"dtype": "datetime"}}
        if step_number <= 3 and "unit_price" in dirty:
            return {
                "action_type": "remove_outliers",
                "column": "unit_price",
                "params": {"method": "iqr"},
            }
        return {"action_type": "submit"}

    # Hard task: optimized sequence (no drop_duplicates — confirmed 0 dupes)
    _hard_actions = [
        {"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}},
        {"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}},
        {"action_type": "fill_nulls", "column": "weight_kg", "params": {"strategy": "median"}},
        {"action_type": "cast_column", "column": "admission_date", "params": {"dtype": "datetime"}},
        {"action_type": "remove_outliers", "column": "weight_kg", "params": {"method": "iqr"}},
        {
            "action_type": "normalize_values",
            "column": "gender",
            "params": {
                "mapping": {
                    "M": "Male", "m": "Male", "male": "Male", "1": "Male",
                    "F": "Female", "f": "Female", "female": "Female", "0": "Female",
                }
            },
        },
        {
            "action_type": "normalize_values",
            "column": "blood_type",
            "params": {
                "mapping": {
                    "A_pos": "A+", "Apos": "A+", "a+": "A+",
                    "A_neg": "A-", "Aneg": "A-", "a-": "A-",
                    "B_pos": "B+", "Bpos": "B+", "b+": "B+",
                    "B_neg": "B-", "Bneg": "B-", "b-": "B-",
                    "AB_pos": "AB+", "ABpos": "AB+", "ab+": "AB+",
                    "AB_neg": "AB-", "ABneg": "AB-", "ab-": "AB-",
                    "O_pos": "O+", "Opos": "O+", "o+": "O+",
                    "O_neg": "O-", "Oneg": "O-", "o-": "O-",
                }
            },
        },
        {"action_type": "cast_column", "column": "readmitted", "params": {"dtype": "bool"}},
    ]
    if step_number < len(_hard_actions):
        return _hard_actions[step_number]
    return {"action_type": "submit"}


def parse_json_action(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM response.

    Tries four strategies in order:
      a) Raw JSON
      b) Markdown ```json ... ``` block
      c) Markdown ``` ... ``` block (no lang tag)
      d) First { ... } substring embedded in prose
    Returns None (never crashes) if all four fail.
    """
    # --- (a) Raw JSON ---------------------------------------------------
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, ValueError):
        pass

    # --- (b) Markdown ```json ... ``` block -----------------------------
    match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass

    # --- (c) Markdown ``` ... ``` block (no lang tag) -------------------
    match = re.search(r"```\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass

    # --- (d) First { ... } embedded in prose ----------------------------
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# Seconds to wait before retrying after a 429 rate-limit response.
_RATE_LIMIT_WAIT_SECONDS = 65


def _is_rate_limited(exc: Exception) -> bool:
    """Return True if the exception signals an HTTP 429."""
    # openai SDK raises openai.RateLimitError (status_code=429).
    exc_type = type(exc).__name__
    if "RateLimitError" in exc_type:
        return True
    # Also check for a generic status_code attribute.
    status = getattr(exc, "status_code", None)
    return status == 429


def llm_action(
    client: OpenAI,
    model_name: str,
    task_id: str,
    observation: dict[str, Any],
) -> dict[str, Any] | None:
    """Ask a chat-completions-compatible model for the next action.

    If the provider returns 429 (rate limit), waits 65 s and retries once.
    If the retry also fails, returns None so the caller can fall back to
    the dry-run policy for this step.
    """
    preview_rows = observation.get("current_df", [])[:10]
    prompt = {
        "task_id": task_id,
        "task_description": observation.get("task_description"),
        "step_number": observation.get("step_number"),
        "max_steps": observation.get("max_steps"),
        "dirty_columns": observation.get("dirty_columns", []),
        "columns_meta": observation.get("columns_meta", {}),
        "data_preview": preview_rows,
        "action_schema": {
            "drop_duplicates": {"action_type": "drop_duplicates"},
            "fill_nulls": {
                "action_type": "fill_nulls",
                "column": "<column>",
                "params": {"strategy": "mean|median|mode|constant", "value": "<optional>"},
            },
            "cast_column": {
                "action_type": "cast_column",
                "column": "<column>",
                "params": {"dtype": "int|float|str|bool|datetime"},
            },
            "remove_outliers": {
                "action_type": "remove_outliers",
                "column": "<column>",
                "params": {"method": "iqr|zscore", "threshold": "<optional float>"},
            },
            "rename_column": {
                "action_type": "rename_column",
                "column": "<column>",
                "params": {"new_name": "<new_name>"},
            },
            "normalize_values": {
                "action_type": "normalize_values",
                "column": "<column>",
                "params": {"mapping": {"raw": "canonical"}},
            },
            "submit": {"action_type": "submit"},
        },
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful data-cleaning agent. "
                "Return exactly one JSON object representing the next action. "
                "No markdown, no explanation. "
                "No-op actions are penalized. 'submit' ends the episode."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(prompt, default=str),
        },
    ]

    # --- First attempt ---------------------------------------------------
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=512,  # raised from 256 to prevent truncated JSON
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        return parse_json_action(content)
    except Exception as exc:  # noqa: BLE001
        if not _is_rate_limited(exc):
            raise  # re-raise non-429 errors for the caller to handle

    # --- 429 retry: wait, then try once more ------------------------------
    time.sleep(_RATE_LIMIT_WAIT_SECONDS)

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=512,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        return parse_json_action(content)
    except Exception:  # noqa: BLE001
        # Retry also failed — return None so caller falls back to dry-run.
        return None


def build_prompt(
    obs: dict[str, Any],
    last_result: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> str:
    """Build the user message for each LLM turn from env state.

    Parameters:
        obs         — raw observation dict from env /state or /reset
        last_result — raw step result dict from last /step call
                      (None on first turn)
        history     — list of {"action": dict, "reward": float,
                      "action_applied": bool} accumulated this episode

    Returns:
        Formatted string to send as the user message.
    """
    import pandas as pd

    # Build DataFrame summary from observation
    current_df = obs.get("current_df", [])
    df = pd.DataFrame(current_df) if current_df else pd.DataFrame()
    total_rows = len(df)
    null_counts = df.isnull().sum().to_dict() if not df.empty else {}
    duplicate_count = int(df.duplicated().sum()) if not df.empty else 0
    df_head = df.head(5).to_string() if not df.empty else "(empty)"

    # Budget
    step_number = obs.get("step_number", 0)
    max_steps = obs.get("max_steps", 30)
    budget_remaining = max_steps - step_number

    # Schema hint — not in API Observation, use hard-coded fallback
    task_id = obs.get("task_id", "")
    task_hints = TASK_SCHEMA_HINTS.get(task_id, {})
    schema_str = "\n".join(
        f"  {col}: {hint}" for col, hint in task_hints.items()
    ) or "  (none)"

    # Action history and zero-reward tracking
    history = history or []
    history_lines: list[str] = []
    zero_reward_actions: list[dict[str, Any]] = []
    for i, h in enumerate(history, 1):
        r = h.get("reward", 0.0)
        applied = h.get("action_applied", True)
        reward_str = f"+{r:.4f}" if r >= 0 else f"{r:.4f}"
        history_lines.append(
            f"  {i:2d}. {json.dumps(h['action'])}"
            f" → reward={reward_str} applied={applied}"
        )
        if r == 0.0 and applied:
            zero_reward_actions.append(h["action"])

    history_str = "\n".join(history_lines) or "  (none — first turn)"
    zero_str = "\n".join(
        f"  {json.dumps(a)}" for a in zero_reward_actions
    ) or "  (none)"

    # Last step result
    if last_result is None:
        last_action_str = "none"
        last_reward_str = "n/a"
        col_deltas_str = "{}"
        dirty_remaining = "n/a"
        partial_score_str = "n/a"
        action_applied_str = "n/a"
        error_str = "none"
    else:
        info = last_result.get("info", {})
        last_action_str = json.dumps(history[-1]["action"]) if history else "n/a"
        last_reward_str = f"{last_result.get('reward', 0.0):+.4f}"
        col_deltas_str = str(info.get("column_deltas", {}))
        dirty_remaining = str(info.get("dirty_columns_remaining", "n/a"))
        partial_score_str = str(info.get("partial_score", "n/a"))
        action_applied_str = str(info.get("action_applied", True))
        error_str = str(info.get("error", "none") or "none")

    return f"""
<STATE>
task_id:              {task_id}
step:                 {step_number} / {max_steps}
budget_remaining:     {budget_remaining}
episode_reward:       {obs.get('episode_reward_so_far', 0.0):.4f}
partial_score:        {partial_score_str}

dirty_columns:        {obs.get('dirty_columns', [])}
columns_meta:         {obs.get('columns_meta', {})}

schema_hint:
{schema_str}

current_df (first 5 rows):
{df_head}

total_rows:           {total_rows}
null_counts:          {null_counts}
duplicate_row_count:  {duplicate_count}
</STATE>

<STEP_RESULT>
action_taken:         {last_action_str}
reward:               {last_reward_str}
column_deltas:        {col_deltas_str}
dirty_columns_now:    {dirty_remaining}
partial_score_now:    {partial_score_str}
action_applied:       {action_applied_str}
error:                {error_str}
</STEP_RESULT>

<HISTORY>
{history_str}
</HISTORY>

Zero-reward actions this episode (do not repeat):
{zero_str}
""".strip()


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform an HTTP request and return parsed JSON or raise a descriptive error."""
    response = session.request(method, url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def build_live_client() -> OpenAI:
    """Create the required OpenAI-compatible client using submission env vars."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set for live mode. Use --dry-run otherwise.")
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def run_task(
    session: requests.Session,
    base_url: str,
    task_id: str,
    dry_run: bool,
    client: OpenAI | None,
    model_name: str,
) -> None:
    """Execute one full task episode and emit START/STEP/END lines."""
    reset_payload = request_json(session, "POST", f"{base_url}/reset", {"task_id": task_id})
    session_id = reset_payload["session_id"]
    observation = reset_payload["observation"]
    log_start(task=task_id, env=BENCHMARK, model=model_name)

    last_score: float = 0.0
    last_step: int = 0
    rewards: list[float] = []
    # Live LLM mode: track action history and last step result
    llm_history: list[dict[str, Any]] = []
    last_result_dict: dict[str, Any] | None = None

    try:
        for _ in range(MAX_LOGICAL_STEPS):
            action: dict[str, Any] | None = None

            if dry_run:
                action = dry_run_action(task_id, observation)
            else:
                try:
                    if client is None:
                        raise RuntimeError("OpenAI client is not configured.")
                    # Build rich prompt with history and state for LLM
                    user_msg = build_prompt(observation, last_result_dict, llm_history)
                    _msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ]
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            temperature=0.0,
                            max_tokens=256,
                            messages=_msgs,
                        )
                        raw = response.choices[0].message.content or ""
                        action = parse_json_action(raw)
                    except Exception as llm_exc:  # noqa: BLE001
                        if not _is_rate_limited(llm_exc):
                            raise  # non-429 → outer except handles it
                        # 429 retry: same wait-and-retry as llm_action()
                        print(f"[RATE_LIMIT] wait_seconds={_RATE_LIMIT_WAIT_SECONDS}", flush=True)
                        print(f"rate_limited wait_seconds={_RATE_LIMIT_WAIT_SECONDS}", file=sys.stderr, flush=True)
                        time.sleep(_RATE_LIMIT_WAIT_SECONDS)
                        try:
                            response = client.chat.completions.create(
                                model=model_name,
                                temperature=0.0,
                                max_tokens=256,
                                messages=_msgs,
                            )
                            raw = response.choices[0].message.content or ""
                            action = parse_json_action(raw)
                        except Exception:  # noqa: BLE001
                            action = None  # handled by if action is None block below
                except Exception as exc:  # noqa: BLE001 - baseline must never crash
                    # Fall back to dry-run policy for this step only.
                    action = dry_run_action(task_id, observation)

            if action is None:
                # Fall back to dry-run policy for this step only.
                action = dry_run_action(task_id, observation)

            try:
                step_payload = request_json(
                    session,
                    "POST",
                    f"{base_url}/step",
                    {"session_id": session_id, "action": action},
                )
            except Exception as exc:  # noqa: BLE001
                # Last-resort: force submit to end the episode cleanly.
                try:
                    action = {"action_type": "submit"}
                    step_payload = request_json(
                        session,
                        "POST",
                        f"{base_url}/step",
                        {"session_id": session_id, "action": action},
                    )
                except Exception:  # noqa: BLE001
                    # Even the submit failed — break out, finally will emit [END].
                    break

            observation = step_payload["observation"]
            last_step = observation.get("step_number", last_step)
            info = step_payload.get("info", {}) or {}
            step_reward = float(step_payload.get("reward", 0.0) or 0.0)
            step_done = bool(step_payload.get("done", False))
            rewards.append(step_reward)

            for score_key in ("final_score", "grader_score", "partial_score"):
                if info.get(score_key) is not None:
                    last_score = _clamp_score(info.get(score_key))
                    break

            log_step(
                step=last_step,
                action=action,
                reward=step_reward,
                done=step_done,
                error=info.get("error"),
            )

            # Track history for live LLM mode
            if not dry_run:
                llm_history.append({
                    "action": action,
                    "reward": step_payload.get("reward", 0.0),
                    "action_applied": step_payload.get("info", {}).get("action_applied", True),
                })
                last_result_dict = step_payload

            if step_done:
                return

    finally:
        success = last_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=last_step, score=last_score, rewards=rewards)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Submission inference runner for DataCleaningEnv.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic actions.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_ORDER,
        default=None,
        help="Optional subset of tasks to run. Defaults to easy medium hard.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for the running DataCleaningEnv API. If omitted and LOCAL_IMAGE_NAME is set, Docker is used.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Model name used for OpenAI-compatible LLM calls.",
    )
    args = parser.parse_args()

    client: OpenAI | None = None
    effective_model = args.model
    docker_handle: DockerEnvHandle | None = None
    if not args.dry_run:
        try:
            client = build_live_client()
        except Exception as exc:  # noqa: BLE001 - CLI should fail cleanly
            print(str(exc), file=sys.stderr)
            return 1

    session = requests.Session()
    try:
        base_url = args.base_url.rstrip("/") if args.base_url else None
        if base_url is None and LOCAL_IMAGE_NAME:
            docker_handle = DockerEnvHandle(LOCAL_IMAGE_NAME)
            base_url = docker_handle.start()
        if base_url is None:
            base_url = "http://localhost:7860"

        selected_tasks = args.tasks or TASK_ORDER
        for task_id in selected_tasks:
            try:
                run_task(
                    session=session,
                    base_url=base_url,
                    task_id=task_id,
                    dry_run=args.dry_run,
                    client=client,
                    model_name=effective_model,
                )
            except Exception as exc:  # noqa: BLE001 - baseline must never crash
                print(f"task_fatal_error task_id={task_id} error={exc}", file=sys.stderr, flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
    except Exception as exc:  # noqa: BLE001 - baseline must never crash
        print(f"fatal_error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        session.close()
        if docker_handle is not None:
            docker_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
