"""Baseline runner for DataCleaningEnv with dry-run and live LLM modes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

TASK_ORDER = ["easy", "medium", "hard"]
REQUEST_TIMEOUT_SECONDS = 15
MAX_LOGICAL_STEPS = 120
HF_TOKEN_ENV = "HF_TOKEN"


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings for a live LLM run."""

    provider: str
    api_key_env: str
    default_model: str
    base_url: str | None = None


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    ),
    "gemini": ProviderConfig(
        provider="gemini",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
}


def emit(tag: str, payload: dict[str, Any]) -> None:
    """Emit one structured log line with the required competition token prefix."""
    print(f"{tag} {json.dumps(payload, default=str)}", flush=True)


def emit_stderr(event: str, **payload: Any) -> None:
    """Emit auxiliary diagnostics to stderr so stdout stays evaluator-friendly."""
    print(json.dumps({"event": event, **payload}, default=str), file=sys.stderr, flush=True)


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

    # --- 429 retry: wait 65 s, then try once more ------------------------
    emit("[RATE_LIMIT]", {"wait_seconds": _RATE_LIMIT_WAIT_SECONDS})
    emit_stderr("rate_limited", wait_seconds=_RATE_LIMIT_WAIT_SECONDS)
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


def configure_api_session(session: requests.Session) -> None:
    """Apply optional runtime headers used for hosted deployments."""
    session.headers.setdefault("User-Agent", "DataCleaningEnv-Baseline/1.0")
    hf_token = os.getenv(HF_TOKEN_ENV)
    if hf_token:
        session.headers.setdefault("Authorization", f"Bearer {hf_token}")


def resolve_provider_config(
    provider_name: str,
    model_name: str | None,
    llm_base_url: str | None,
) -> tuple[ProviderConfig, str]:
    """Resolve the provider config and effective model name for live mode."""
    provider_key = provider_name.strip().lower()
    if provider_key not in PROVIDER_CONFIGS:
        raise ValueError(f"Unsupported provider '{provider_name}'. Choose from {sorted(PROVIDER_CONFIGS)}.")

    provider = PROVIDER_CONFIGS[provider_key]
    effective_provider = ProviderConfig(
        provider=provider.provider,
        api_key_env=provider.api_key_env,
        default_model=provider.default_model,
        base_url=(llm_base_url or provider.base_url or os.getenv("LLM_BASE_URL")),
    )
    effective_model = model_name or os.getenv("MODEL_NAME") or provider.default_model
    return effective_provider, effective_model


def build_live_client(provider: ProviderConfig) -> OpenAI:
    """Create an OpenAI SDK client for the selected provider."""
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{provider.api_key_env} must be set for live mode. Use --dry-run otherwise."
        )
    return OpenAI(api_key=api_key, base_url=provider.base_url)


def run_task(
    session: requests.Session,
    base_url: str,
    task_id: str,
    dry_run: bool,
    client: OpenAI | None,
    model_name: str,
) -> None:
    """Execute one full task episode and emit structured logs.

    [END] is **always** emitted — even if the episode errors mid-way —
    via a try/finally guard.
    """
    session_id: str | None = None
    observation: dict[str, Any] = {}

    # Track last-known score so [END] always has *something* to report.
    last_score: float = 0.0
    last_step: int = 0
    end_emitted = False

    try:
        try:
            reset_payload = request_json(
                session,
                "POST",
                f"{base_url}/reset",
                {"task_id": task_id},
            )
            session_id = reset_payload["session_id"]
            observation = reset_payload["observation"]
            emit("[START]", {"task_id": task_id, "session_id": session_id})
            emit_stderr("episode_start", task_id=task_id, session_id=session_id, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - baseline must degrade cleanly
            emit_stderr("reset_failed", task_id=task_id, error=str(exc))
            emit("[START]", {"task_id": task_id, "session_id": None})
            emit(
                "[END]",
                {
                    "task_id": task_id,
                    "score": 0.0,
                    "steps": 0,
                    "error": str(exc),
                },
            )
            end_emitted = True
            return

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
                    emit(
                        "[STEP]",
                        {
                            "step": observation.get("step_number", 0) + 1,
                            "action": None,
                            "reward": 0.0,
                            "done": False,
                            "error": str(exc),
                        },
                    )
                    # Fall back to dry-run policy for this step only.
                    emit_stderr(
                        "llm_fallback",
                        step=observation.get("step_number", 0) + 1,
                        error=str(exc),
                    )
                    action = dry_run_action(task_id, observation)

            if action is None:
                # All 4 JSON extraction strategies failed.
                emit(
                    "[STEP]",
                    {
                        "step": observation.get("step_number", 0) + 1,
                        "action": None,
                        "reward": 0.0,
                        "done": False,
                        "error": "unparseable_llm_json",
                    },
                )
                # Fall back to dry-run policy for this step only.
                emit_stderr("json_parse_fallback", step=observation.get("step_number", 0) + 1)
                action = dry_run_action(task_id, observation)

            try:
                step_payload = request_json(
                    session,
                    "POST",
                    f"{base_url}/step",
                    {"session_id": session_id, "action": action},
                )
            except Exception as exc:  # noqa: BLE001
                emit(
                    "[STEP]",
                    {
                        "step": observation.get("step_number", 0) + 1,
                        "action": action,
                        "reward": 0.0,
                        "done": False,
                        "error": str(exc),
                    },
                )
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
            grader_score = step_payload.get("info", {}).get("grader_score")
            if grader_score is not None:
                last_score = grader_score

            emit(
                "[STEP]",
                {
                    "step": last_step,
                    "action": action,
                    "reward": step_payload.get("reward", 0.0),
                    "done": step_payload.get("done", False),
                },
            )

            if step_payload.get("done"):
                emit(
                    "[END]",
                    {
                        "task_id": task_id,
                        "score": last_score,
                        "steps": last_step,
                    },
                )
                end_emitted = True
                return

    finally:
        # Guarantee [END] even on unexpected errors or max-step exhaustion.
        if not end_emitted:
            emit("[END]", {"task_id": task_id, "score": last_score, "steps": last_step})


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Baseline runner for DataCleaningEnv.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic actions.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://localhost:7860"),
        help="Base URL for the running DataCleaningEnv API.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_CONFIGS),
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="Live LLM provider used when not running in --dry-run mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name used when not running in --dry-run mode.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional chat-completions base URL override for live mode.",
    )
    args = parser.parse_args()

    client: OpenAI | None = None
    effective_model = args.model or os.getenv("MODEL_NAME") or PROVIDER_CONFIGS[args.provider].default_model
    if not args.dry_run:
        try:
            provider, effective_model = resolve_provider_config(
                provider_name=args.provider,
                model_name=args.model,
                llm_base_url=args.llm_base_url,
            )
            client = build_live_client(provider)
        except Exception as exc:  # noqa: BLE001 - CLI should fail cleanly
            emit_stderr("provider_init_failed", error=str(exc))
            print(str(exc), file=sys.stderr)
            return 1

    session = requests.Session()
    configure_api_session(session)
    try:
        for task_id in TASK_ORDER:
            try:
                run_task(
                    session=session,
                    base_url=args.base_url.rstrip("/"),
                    task_id=task_id,
                    dry_run=args.dry_run,
                    client=client,
                    model_name=effective_model,
                )
            except Exception as exc:  # noqa: BLE001 - baseline must never crash
                emit_stderr("task_fatal_error", task_id=task_id, error=str(exc))
                emit(
                    "[END]",
                    {"task_id": task_id, "score": 0.0, "steps": 0, "error": str(exc)},
                )
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
