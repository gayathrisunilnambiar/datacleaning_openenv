"""End-to-end competition compliance audit for DataCleaningEnv."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

try:
    import yaml
except ImportError:  # pragma: no cover - only used as a fallback guard
    yaml = None


ROOT = Path(__file__).resolve().parent
EXPECTED_DRY_RUN = {
    "easy": 1.0000,
    "medium": 0.9811,
    "hard": 0.8332,
}
EXPECTED_DIRTY = {
    "easy": 0.3921,
    "medium": 0.2930,
    "hard": 0.2493,
}
TOLERANCE = 1e-4
MAX_RUNTIME_SECONDS = 1200.0
LIVE_SPACE_URL = "https://sanvihs2005-data-cleaning-env.hf.space"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class CheckResult:
    label: str
    status: str
    detail: str = ""
    lines: list[str] = field(default_factory=list)


def status_text(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def nearly_equal(value: float, expected: float, tol: float = TOLERANCE) -> bool:
    return abs(float(value) - float(expected)) <= tol


def quiet_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def local_api_server() -> str:
    """Run the FastAPI app under uvicorn for subprocess-based checks."""
    port = free_port()
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 15.0
        while time.time() < deadline:
            try:
                response = requests.get(f"{base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    yield base_url
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Timed out waiting for local uvicorn server to start.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def load_app_module():
    import api.main as main_module

    return importlib.reload(main_module)


def run_subprocess(command: list[str], env: dict[str, str] | None = None, timeout: float = 1200.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def parse_tagged_stdout(stdout: str) -> tuple[bool, list[str], dict[str, tuple[float, int]]]:
    """Validate [START]/[STEP]/[END] ordering and payload structure."""
    failures: list[str] = []
    current_task: str | None = None
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    steps: dict[str, int] = {}
    end_scores: dict[str, tuple[float, int]] = {}

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False, ["stdout was empty"], {}

    for line in lines:
        if not line.startswith("[") or "] " not in line:
            failures.append(f"stdout line missing competition prefix: {line}")
            continue

        prefix, payload_text = line.split("] ", 1)
        prefix = f"{prefix}]"
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON after {prefix}: {exc}")
            continue

        if prefix == "[START]":
            task_id = payload.get("task_id")
            session_id = payload.get("session_id")
            if not isinstance(task_id, str):
                failures.append(f"[START] missing string task_id: {payload}")
                continue
            if task_id in starts:
                failures.append(f"duplicate [START] for task {task_id}")
            if session_id is None:
                failures.append(f"[START] missing session_id for task {task_id}")
            starts[task_id] = starts.get(task_id, 0) + 1
            current_task = task_id
            steps.setdefault(task_id, 0)
        elif prefix == "[STEP]":
            if current_task is None:
                failures.append(f"[STEP] emitted before any [START]: {payload}")
                continue
            required = {"step", "action", "reward", "done"}
            if not required.issubset(payload):
                failures.append(f"[STEP] missing keys for task {current_task}: {payload}")
                continue
            steps[current_task] = steps.get(current_task, 0) + 1
        elif prefix == "[END]":
            task_id = payload.get("task_id")
            score = payload.get("score")
            step_count = payload.get("steps")
            if not isinstance(task_id, str):
                failures.append(f"[END] missing string task_id: {payload}")
                continue
            if task_id not in starts:
                failures.append(f"[END] appeared before [START] for task {task_id}")
            if task_id in ends:
                failures.append(f"duplicate [END] for task {task_id}")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                failures.append(f"[END] missing numeric score for task {task_id}: {payload}")
                continue
            if not 0.0 <= score_value <= 1.0:
                failures.append(f"[END] score out of range for task {task_id}: {score_value}")
            if not isinstance(step_count, int):
                failures.append(f"[END] missing integer steps for task {task_id}: {payload}")
            ends[task_id] = ends.get(task_id, 0) + 1
            end_scores[task_id] = (score_value, int(step_count or 0))
            current_task = None
        elif prefix == "[RATE_LIMIT]":
            continue
        else:
            failures.append(f"unexpected log prefix {prefix}")

    for task_id in EXPECTED_DRY_RUN:
        if starts.get(task_id, 0) != 1:
            failures.append(f"expected exactly one [START] for {task_id}")
        if steps.get(task_id, 0) < 1:
            failures.append(f"expected at least one [STEP] for {task_id}")
        if ends.get(task_id, 0) != 1:
            failures.append(f"expected exactly one [END] for {task_id}")

    return not failures, failures, end_scores


def scripted_baseline_score(task_id: str) -> float:
    from environment.env import DataCleaningEnv
    from inference import dry_run_action

    with quiet_stdout():
        env = DataCleaningEnv(task_id)
        observation = env.reset()
        for _ in range(120):
            action = dry_run_action(task_id, observation.model_dump())
            result = env.step(action)
            observation = result.observation
            if result.done:
                if result.info.grader_score is None:
                    raise AssertionError(f"{task_id} submit result missing grader_score")
                return float(result.info.grader_score)
    raise AssertionError(f"{task_id} scripted baseline exceeded 120 logical steps")


def check_file_structure() -> CheckResult:
    required = [
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "requirements.txt",
        "README.md",
        "api/main.py",
        "environment/env.py",
        "environment/models.py",
        "environment/tasks/task_easy.py",
        "environment/tasks/task_medium.py",
        "environment/tasks/task_hard.py",
        "environment/graders/grader_easy.py",
        "environment/graders/grader_medium.py",
        "environment/graders/grader_hard.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    detail = "all required files present" if not missing else f"missing: {', '.join(missing)}"
    return CheckResult("CHECK 1  File Structure", status_text(not missing), detail)


def check_openenv_yaml() -> CheckResult:
    if yaml is None:
        return CheckResult("CHECK 2  openenv.yaml Schema", "FAIL", "PyYAML is not available")

    payload = yaml.safe_load((ROOT / "openenv.yaml").read_text(encoding="utf-8"))
    failures: list[str] = []

    for key in ("name", "version", "description", "tags", "action_space", "observation_space", "tasks", "baseline_scores"):
        if key not in payload:
            failures.append(f"missing top-level key '{key}'")

    actions = payload.get("action_space", {})
    expected_actions = {
        "drop_duplicates",
        "fill_nulls",
        "cast_column",
        "remove_outliers",
        "rename_column",
        "normalize_values",
        "submit",
    }
    if set(actions) != expected_actions:
        failures.append("action_space does not define all 7 required actions")

    tasks = payload.get("tasks", [])
    if len(tasks) < 3:
        failures.append("tasks must contain at least 3 entries")
    for task in tasks[:3]:
        for field_name in ("id", "difficulty", "max_steps"):
            if field_name not in task:
                failures.append(f"task entry missing '{field_name}'")

    if payload.get("reward_range") != [0.0, 1.0]:
        failures.append("reward_range must be [0.0, 1.0]")

    endpoints = payload.get("endpoints", {})
    for key in ("reset", "step", "state"):
        if key not in endpoints:
            failures.append(f"endpoints missing '{key}'")

    detail = "schema fields present and well-formed" if not failures else "; ".join(failures)
    return CheckResult("CHECK 2  openenv.yaml Schema", status_text(not failures), detail)


def check_models() -> CheckResult:
    from environment.models import Action, Observation, ResetRequest, StepResult

    failures: list[str] = []
    valid_actions = [
        {"action_type": "drop_duplicates"},
        {"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}},
        {"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}},
        {"action_type": "remove_outliers", "column": "price", "params": {"method": "iqr"}},
        {"action_type": "rename_column", "column": "qty", "params": {"new_name": "quantity"}},
        {"action_type": "normalize_values", "column": "gender", "params": {"mapping": {"M": "Male"}}},
        {"action_type": "submit"},
    ]

    for payload in valid_actions:
        try:
            Action.model_validate(payload)
        except ValidationError as exc:
            failures.append(f"Action rejected valid payload {payload}: {exc}")

    required_observation_fields = {
        "task_id",
        "step_number",
        "current_df",
        "dirty_columns",
        "columns_meta",
        "episode_reward_so_far",
        "done",
        "max_steps",
    }
    if not required_observation_fields.issubset(Observation.model_fields):
        failures.append("Observation missing one or more required fields")

    if set(StepResult.model_fields) != {"observation", "reward", "done", "info"}:
        failures.append("StepResult fields do not match expected contract")

    reset_request = ResetRequest(task_id="easy", seed=42)
    if reset_request.task_id != "easy" or reset_request.seed != 42:
        failures.append("ResetRequest did not accept task_id and optional seed")

    try:
        Action.model_validate({"action_type": "totally_invalid"})
        failures.append("Action accepted an invalid action_type")
    except ValidationError:
        pass

    detail = "typed models accept valid payloads and reject invalid ones" if not failures else "; ".join(failures)
    return CheckResult("CHECK 3  Pydantic Model Compliance", status_text(not failures), detail)


def check_environment_functional() -> CheckResult:
    from environment.env import DataCleaningEnv
    from environment.models import Observation, StepResult

    failures: list[str] = []
    lines: list[str] = []

    for task_id in ("easy", "medium", "hard"):
        try:
            with quiet_stdout():
                env = DataCleaningEnv(task_id)
                observation = env.reset()
                if not isinstance(observation, Observation):
                    raise AssertionError("reset() did not return Observation")

                first_step = env.step({"action_type": "drop_duplicates"})
                if not isinstance(first_step, StepResult):
                    raise AssertionError("step(drop_duplicates) did not return StepResult")
                if not isinstance(first_step.reward, float):
                    raise AssertionError("reward is not a float")
                if not -0.05 <= first_step.reward <= 1.30:
                    raise AssertionError(f"reward out of range: {first_step.reward}")

                state = env.state()
                if not isinstance(state, Observation):
                    raise AssertionError("state() did not return Observation")

                submit_result = env.step({"action_type": "submit"})
                if not submit_result.done:
                    raise AssertionError("submit did not end the episode")
                submit_score = float(submit_result.info.grader_score or 0.0)
                if not 0.0 <= submit_score <= 1.0:
                    raise AssertionError(f"submit score out of range: {submit_score}")

            baseline_score = scripted_baseline_score(task_id)
            if not nearly_equal(baseline_score, EXPECTED_DRY_RUN[task_id]):
                raise AssertionError(
                    f"baseline mismatch for {task_id}: {baseline_score:.4f} != {EXPECTED_DRY_RUN[task_id]:.4f}"
                )

            lines.append(
                f"         {task_id:<6} reset->step->state->submit ........... PASS (score: {baseline_score:.4f})"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{task_id}: {exc}")
            lines.append(
                f"         {task_id:<6} reset->step->state->submit ........... FAIL ({exc})"
            )

    return CheckResult(
        "CHECK 4  Environment Functional Test",
        status_text(not failures),
        "; ".join(failures) if failures else "all deterministic task flows passed",
        lines,
    )


def check_grader_ranges() -> CheckResult:
    from environment.graders import GRADER_REGISTRY

    failures: list[str] = []
    lines = [
        "         ┌─────────┬────────────┬────────────┬─────────────┐",
        "         │ Task    │ dirty_score│ clean_score│ partial_score│",
        "         ├─────────┼────────────┼────────────┼─────────────┤",
    ]

    for task_id in ("easy", "medium", "hard"):
        grader = GRADER_REGISTRY[task_id]()
        dirty_score = float(grader.score(grader.task.get_dirty_df()))
        clean_score = float(grader.score(grader.task.get_ground_truth_df()))
        partial_score = float(grader.partial_score(grader.task.get_dirty_df()))

        if not 0.0 <= dirty_score <= 1.0:
            failures.append(f"{task_id} dirty_score out of range")
        if not nearly_equal(clean_score, 1.0):
            failures.append(f"{task_id} clean_score is not 1.0")
        if not 0.0 <= partial_score <= 1.0:
            failures.append(f"{task_id} partial_score out of range")
        if not nearly_equal(dirty_score, EXPECTED_DIRTY[task_id]):
            failures.append(f"{task_id} dirty_score mismatch")
        lines.append(
            f"         │ {task_id:<7} │   {dirty_score:0.4f}   │   {clean_score:0.4f}   │    {partial_score:0.4f}   │"
        )

    lines.append("         └─────────┴────────────┴────────────┴─────────────┘")
    return CheckResult(
        "CHECK 5  Grader Score Ranges",
        status_text(not failures),
        "; ".join(failures) if failures else "grader scores match expected ranges",
        lines,
    )


def check_api_endpoints() -> CheckResult:
    main_module = load_app_module()
    failures: list[str] = []
    lines: list[str] = []

    with quiet_stdout():
        client = TestClient(main_module.app)
        reset_response = client.post("/reset", json={"task_id": "easy"})
        reset_json = reset_response.json()
        session_id = reset_json.get("session_id")

        step_response = client.post(
            "/step",
            json={"session_id": session_id, "action": {"action_type": "drop_duplicates"}},
        )
        state_response = client.get(f"/state?session_id={session_id}")
        checks = [
            ("GET", "/", client.get("/"), lambda r: r.status_code == 200),
            ("GET", "/health", client.get("/health"), lambda r: r.status_code == 200 and "status" in r.json()),
            ("GET", "/tasks", client.get("/tasks"), lambda r: r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) >= 3),
            ("GET", "/metadata", client.get("/metadata"), lambda r: r.status_code == 200),
            ("GET", "/schema", client.get("/schema"), lambda r: r.status_code == 200),
            ("POST", "/reset", reset_response, lambda r: r.status_code == 200 and "session_id" in r.json()),
            ("POST", "/step", step_response, lambda r: r.status_code == 200 and "observation" in r.json()),
            ("GET", "/state", state_response, lambda r: r.status_code == 200 and r.json().get("session_id") == session_id),
            ("POST", "/validate", client.post("/validate"), lambda r: r.status_code == 200 and "passed" in r.json()),
            ("POST", "/mcp", client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}), lambda r: r.status_code == 200),
        ]

    passed_count = 0
    for method, path, response, predicate in checks:
        passed = predicate(response)
        passed_count += int(passed)
        lines.append(
            f"         {method:<4} {path:<10} -> {response.status_code:<3} {status_text(passed)}"
        )
        if not passed:
            failures.append(f"{method} {path} returned unexpected payload/status")

    return CheckResult(
        "CHECK 6  API Endpoint Compliance",
        status_text(not failures),
        f"{passed_count}/10 endpoints satisfied the contract" if not failures else "; ".join(failures),
        lines,
    )


def evaluate_dry_run(base_url: str) -> tuple[subprocess.CompletedProcess[str], float]:
    env = os.environ.copy()
    env["API_BASE_URL"] = base_url
    start = time.time()
    completed = run_subprocess([sys.executable, "inference.py", "--dry-run"], env=env)
    elapsed = time.time() - start
    return completed, elapsed


def check_log_format_and_runtime() -> tuple[CheckResult, CheckResult]:
    failures: list[str] = []
    runtime_failures: list[str] = []
    log_lines: list[str] = []

    with local_api_server() as base_url:
        completed, elapsed = evaluate_dry_run(base_url)

    if completed.returncode != 0:
        failures.append(f"inference.py --dry-run exited with {completed.returncode}")
    if completed.stderr.strip():
        # stderr diagnostics are fine; tracebacks are not.
        if "Traceback" in completed.stderr:
            failures.append("dry-run emitted a traceback to stderr")

    valid_logs, parse_failures, end_scores = parse_tagged_stdout(completed.stdout)
    failures.extend(parse_failures)
    if not valid_logs:
        failures.append("dry-run log stream failed structural validation")

    for task_id in ("easy", "medium", "hard"):
        if task_id in end_scores:
            score, steps = end_scores[task_id]
            if not nearly_equal(score, EXPECTED_DRY_RUN[task_id]):
                failures.append(
                    f"[END] score mismatch for {task_id}: {score:.4f} != {EXPECTED_DRY_RUN[task_id]:.4f}"
                )
            log_lines.append(f"         [END] {task_id:<6} score={score:.4f} steps={steps}")

    if elapsed >= MAX_RUNTIME_SECONDS:
        runtime_failures.append(
            f"dry-run exceeded {MAX_RUNTIME_SECONDS:.0f}s ({elapsed:.2f}s)"
        )

    log_result = CheckResult(
        "CHECK 7  Log Format Compliance",
        status_text(not failures),
        "dry-run emitted valid [START]/[STEP]/[END] logs" if not failures else "; ".join(failures),
        log_lines,
    )
    runtime_result = CheckResult(
        "CHECK 9  Runtime",
        status_text(not runtime_failures),
        f"{elapsed:.2f}s / {MAX_RUNTIME_SECONDS:.0f}s max" if not runtime_failures else "; ".join(runtime_failures),
    )
    return log_result, runtime_result


def check_environment_variables() -> CheckResult:
    inference_source = (ROOT / "inference.py").read_text(encoding="utf-8")
    failures: list[str] = []

    for token in ('API_BASE_URL', 'MODEL_NAME', 'HF_TOKEN'):
        if token not in inference_source:
            failures.append(f"inference.py does not reference {token}")

    unreachable_env = os.environ.copy()
    unreachable_env["API_BASE_URL"] = "http://127.0.0.1:9"
    completed = run_subprocess([sys.executable, "inference.py", "--dry-run"], env=unreachable_env, timeout=60.0)

    if completed.returncode != 0:
        failures.append(f"unreachable API_BASE_URL run exited with {completed.returncode}")
    if "Traceback" in completed.stderr:
        failures.append("unreachable API_BASE_URL produced a traceback")

    end_scores: dict[str, float] = {}
    for line in [item.strip() for item in completed.stdout.splitlines() if item.strip()]:
        if not line.startswith("["):
            failures.append(f"unexpected stdout line for unreachable API_BASE_URL: {line}")
            continue
        if "] " not in line:
            failures.append(f"malformed competition log line: {line}")
            continue
        prefix, payload_text = line.split("] ", 1)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON in unreachable API_BASE_URL output: {exc}")
            continue
        if f"{prefix}]" == "[END]":
            task_id = payload.get("task_id")
            score = payload.get("score")
            if isinstance(task_id, str):
                end_scores[task_id] = float(score)

    for task_id in ("easy", "medium", "hard"):
        if task_id not in end_scores:
            failures.append(f"missing [END] for {task_id} on unreachable API_BASE_URL")
            continue
        if not nearly_equal(end_scores[task_id], 0.0):
            failures.append(f"{task_id} did not emit score=0.0 for unreachable API_BASE_URL")

    return CheckResult(
        "CHECK 8  Environment Variable Handling",
        status_text(not failures),
        "API_BASE_URL, MODEL_NAME, and HF_TOKEN are read at runtime; unreachable base URL degrades cleanly"
        if not failures
        else "; ".join(failures),
    )


def check_live_space() -> CheckResult:
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            health = client.get(f"{LIVE_SPACE_URL}/health")
            if health.status_code != 200:
                return CheckResult(
                    "CHECK 10 Live Space Ping",
                    "FAIL",
                    f"/health returned {health.status_code}",
                )

            reset = client.post(f"{LIVE_SPACE_URL}/reset", json={"task_id": "easy"})
            if reset.status_code != 200:
                return CheckResult(
                    "CHECK 10 Live Space Ping",
                    "FAIL",
                    f"/reset returned {reset.status_code}",
                )
            payload = reset.json()
            if "session_id" not in payload:
                return CheckResult(
                    "CHECK 10 Live Space Ping",
                    "FAIL",
                    "/reset response missing session_id",
                )
    except Exception as exc:  # noqa: BLE001 - network is optional for this script
        return CheckResult(
            "CHECK 10 Live Space Ping",
            "SKIP",
            f"network unavailable or live Space unreachable ({exc})",
        )

    return CheckResult("CHECK 10 Live Space Ping", "PASS", "live /health and /reset succeeded")


def print_report(results: list[CheckResult]) -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     DataCleaningEnv — Competition Compliance Report     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    for result in results:
        if result.lines:
            print(f"{result.label}")
            for line in result.lines:
                print(line)
            if result.detail:
                print(f"         Summary: {result.status} ({result.detail})")
        else:
            dots = "." * max(1, 54 - len(result.label))
            print(f"{result.label} {dots} {result.status}", end="")
            if result.detail:
                print(f" ({result.detail})")
            else:
                print()
    print()

    fail_count = sum(1 for result in results if result.status == "FAIL")
    skip_count = sum(1 for result in results if result.status == "SKIP")
    pass_count = sum(1 for result in results if result.status == "PASS")

    print("══════════════════════════════════════════════════════════")
    if fail_count:
        print(f"RESULT: {pass_count}/{len(results)} checks passed")
        print("SUBMISSION STATUS: NOT READY")
    elif skip_count:
        print(f"RESULT: {pass_count}/{len(results)} checks passed, {skip_count} skipped")
        print("SUBMISSION STATUS: READY (live ping skipped)")
    else:
        print(f"RESULT: {pass_count}/{len(results)} checks passed")
        print("SUBMISSION STATUS: READY")
    print("══════════════════════════════════════════════════════════")


def main() -> int:
    results = [
        check_file_structure(),
        check_openenv_yaml(),
        check_models(),
        check_environment_functional(),
        check_grader_ranges(),
        check_api_endpoints(),
    ]
    log_result, runtime_result = check_log_format_and_runtime()
    results.extend(
        [
            log_result,
            check_environment_variables(),
            runtime_result,
            check_live_space(),
        ]
    )
    print_report(results)
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
