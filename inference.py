"""Baseline runner for DataCleaningEnv with dry-run and live LLM modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI


TASK_ORDER = ["easy", "medium", "hard"]
REQUEST_TIMEOUT_SECONDS = 15
MAX_LOGICAL_STEPS = 120


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
    """Best-effort JSON extraction from an LLM response."""
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def llm_action(
    client: OpenAI,
    model_name: str,
    task_id: str,
    observation: dict[str, Any],
) -> dict[str, Any] | None:
    """Ask a chat-completions-compatible model for the next action."""
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

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_tokens=256,
        messages=[
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
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_json_action(content)


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
    """Execute one full task episode and emit structured logs."""
    reset_payload = request_json(session, "POST", f"{base_url}/reset", {"task_id": task_id})
    session_id = reset_payload["session_id"]
    observation = reset_payload["observation"]
    emit("[START]", {"task_id": task_id, "session_id": session_id})

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
                        "reward": 0,
                        "done": False,
                        "error": str(exc),
                    },
                )
                action = dry_run_action(task_id, observation)

        if action is None:
            emit(
                "[STEP]",
                {
                    "step": observation.get("step_number", 0) + 1,
                    "action": None,
                    "reward": 0,
                    "done": False,
                    "error": "unparseable_llm_json",
                },
            )
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
                    "reward": 0,
                    "done": False,
                    "error": str(exc),
                },
            )
            action = {"action_type": "submit"}
            step_payload = request_json(
                session,
                "POST",
                f"{base_url}/step",
                {"session_id": session_id, "action": action},
            )

        observation = step_payload["observation"]
        emit(
            "[STEP]",
            {
                "step": observation.get("step_number"),
                "action": action,
                "reward": step_payload.get("reward", 0),
                "done": step_payload.get("done", False),
            },
        )

        if step_payload.get("done"):
            emit(
                "[END]",
                {
                    "task_id": task_id,
                    "score": step_payload.get("info", {}).get("grader_score"),
                    "steps": observation.get("step_number"),
                },
            )
            return

    emit("[END]", {"task_id": task_id, "score": None, "steps": observation.get("step_number", 0)})


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
            print(str(exc), file=sys.stderr)
            return 1

    session = requests.Session()
    try:
        for task_id in TASK_ORDER:
            run_task(
                session=session,
                base_url=args.base_url.rstrip("/"),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
