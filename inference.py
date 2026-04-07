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

    if step_number == 0:
        return {"action_type": "drop_duplicates"}
    if step_number == 1 and "age" in dirty and column_has_nulls(observation, "age"):
        return {"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}}
    if step_number <= 2 and "age" in dirty and columns_meta.get("age") != "int64":
        return {"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}}
    if step_number <= 3 and "weight_kg" in dirty and column_has_nulls(observation, "weight_kg"):
        return {
            "action_type": "fill_nulls",
            "column": "weight_kg",
            "params": {"strategy": "median"},
        }
    if step_number <= 4 and not str(columns_meta.get("admission_date", "")).startswith("datetime64"):
        return {
            "action_type": "cast_column",
            "column": "admission_date",
            "params": {"dtype": "datetime"},
        }
    if step_number <= 5 and "weight_kg" in dirty:
        return {
            "action_type": "remove_outliers",
            "column": "weight_kg",
            "params": {"method": "iqr"},
        }
    if step_number <= 6 and "gender" in dirty:
        return {
            "action_type": "normalize_values",
            "column": "gender",
            "params": {
                "mapping": {
                    "M": "Male",
                    "male": "Male",
                    "1": "Male",
                    "F": "Female",
                    "female": "Female",
                    "0": "Female",
                }
            },
        }
    if step_number <= 7 and "blood_type" in dirty:
        return {
            "action_type": "normalize_values",
            "column": "blood_type",
            "params": {
                "mapping": {
                    "A_pos": "A+",
                    "Apos": "A+",
                    "a+": "A+",
                    "A_neg": "A-",
                    "Aneg": "A-",
                    "a-": "A-",
                    "B_pos": "B+",
                    "Bpos": "B+",
                    "b+": "B+",
                    "B_neg": "B-",
                    "Bneg": "B-",
                    "b-": "B-",
                    "AB_pos": "AB+",
                    "ABpos": "AB+",
                    "ab+": "AB+",
                    "AB_neg": "AB-",
                    "ABneg": "AB-",
                    "ab-": "AB-",
                    "O_pos": "O+",
                    "Opos": "O+",
                    "o+": "O+",
                    "O_neg": "O-",
                    "Oneg": "O-",
                    "o-": "O-",
                }
            },
        }
    if step_number <= 8 and columns_meta.get("readmitted") != "bool":
        return {"action_type": "cast_column", "column": "readmitted", "params": {"dtype": "bool"}}
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

    try:
        for _ in range(MAX_LOGICAL_STEPS):
            action: dict[str, Any] | None = None

            if dry_run:
                action = dry_run_action(task_id, observation)
            else:
                try:
                    if client is None:
                        raise RuntimeError("OpenAI client is not configured.")
                    action = llm_action(client, model_name, task_id, observation)
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

        for task_id in TASK_ORDER:
            run_task(
                session=session,
                base_url=base_url,
                task_id=task_id,
                dry_run=args.dry_run,
                client=client,
                model_name=effective_model,
            )
    except Exception as exc:  # noqa: BLE001 - baseline must never crash
        print(f"fatal_error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()
        if docker_handle is not None:
            docker_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
