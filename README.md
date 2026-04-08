---
title: DataCleaningEnv
emoji: 🧹
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - data-cleaning
---

# 🧹 DataCleaningEnv

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/openenv)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![HF Space](https://img.shields.io/badge/HF%20Space-Live-yellow)](https://sanvihs2005-data-cleaning-env.hf.space)
[![Tests: 26 passing](https://img.shields.io/badge/Tests-26%20passing-green)](#section-13--running-tests)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

DataCleaningEnv is an OpenEnv benchmark for iterative tabular data cleaning, where an agent improves a dirty DataFrame one action at a time and is graded on the quality of the cleaning process.

## Section 2 — Why This Matters

Data cleaning consumes an estimated 60-80% of a data scientist's time, yet OpenEnv-style environments have mostly focused on reasoning, browsing, or code generation rather than tabular data quality. DataCleaningEnv fills that gap by measuring the cleaning process itself, not just the final answer: whether an agent chooses useful next transformations, avoids no-op loops, and improves the table under a step budget. That makes it useful both for competition judges skimming for benchmark quality and for ML researchers training enterprise-grade agents for data pipeline automation.

## Section 3 — Environment Overview

Each episode follows a simple `reset -> step -> state -> submit` loop. `reset` starts a task and returns the first observation, `step` applies a single cleaning action, `state` inspects the current session without mutation, and `submit` ends the episode with a final grader score.

| Concept | What It Is Here |
|---|---|
| Observation | Current dirty DataFrame + metadata |
| Action | A cleaning operation |
| Reward | Column-level similarity improvement |
| Episode | One cleaning session on one task |
| Policy | The LLM agent (GPT-4, Gemini, etc.) |

## Section 4 — Task Suite

| Task | Difficulty | Domain | Rows | Issues | Max Steps |
|---|---|---|---:|---|---:|
| easy | Easy | Employee Records | 50 | Duplicates; missing values | 20 |
| medium | Medium | Sales Transactions | 120 | Type errors; date formats; outliers; arithmetic errors | 30 |
| hard | Hard | Hospital Admissions | 200 | All above; gender variants; unit mixing; typos; booleans | 40 |
| random | Variable | Procedural | 60-150 | 2-4 randomly selected issues | 25 |

## Section 5 — Action Space

| Action | Params Required | Example |
|---|---|---|
| drop_duplicates | none | `{"action_type": "drop_duplicates"}` |
| fill_nulls | column, strategy | `{"action_type": "fill_nulls", "column": "age", "params": {"strategy": "median"}}` |
| cast_column | column, dtype | `{"action_type": "cast_column", "column": "age", "params": {"dtype": "int"}}` |
| remove_outliers | column, method | `{"action_type": "remove_outliers", "column": "price", "params": {"method": "iqr"}}` |
| rename_column | column, new_name | `{"action_type": "rename_column", "column": "qty", "params": {"new_name": "quantity"}}` |
| normalize_values | column, mapping | `{"action_type": "normalize_values", "column": "gender", "params": {"mapping": {"M": "Male"}}}` |
| submit | none | `{"action_type": "submit"}` |

## Section 6 — Reward Function

```text
reward = column_improvement + submit_bonus - redundancy_penalty
column_improvement = Σ per-column similarity gains this step
submit_bonus       = +0.30 if submit AND overall_similarity > 0.80
redundancy_penalty = -0.05 if action caused zero change (no-op)
```

| Column Type | Metric | Range |
|---|---|---|
| Numeric | 1 − (MAE / value_range) | 0.0–1.0 |
| Categorical | Exact match ratio | 0.0–1.0 |
| Datetime | Within-1-day match ratio | 0.0–1.0 |
| Boolean | Exact match ratio | 0.0–1.0 |

## Section 7 — Baseline Scores

| Task | Dirty Calibration | Dry-run Score | Notes |
|---|---:|---:|---|
| easy | 0.3921 | 1.0000 | Fully solvable by rule policy |
| medium | 0.2930 | 0.9811 | Near-perfect |
| hard | 0.2493 | 0.8332 | Gap is intentional headroom for smarter LLM agents |

The `hard` gap is intentional. The remaining error comes from `weight_kg` unit conversion: some rows are implicitly in lbs, and a rule-based policy cannot reliably infer which rows need semantic conversion back to kg. That is exactly the kind of enterprise data-cleaning judgment this benchmark is designed to separate.

## Section 8 — API Reference

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| GET | `/` | API directory | No |
| GET | `/health` | Liveness check | No |
| GET | `/tasks` | List all tasks | No |
| GET | `/metadata` | Environment metadata | No |
| GET | `/schema` | Action/observation schemas | No |
| POST | `/reset` | Start new episode | No |
| POST | `/step` | Apply one cleaning action | No |
| GET | `/state` | Current session state | No |
| POST | `/validate` | Internal compliance self-check | No |
| POST | `/mcp` | MCP JSON-RPC shim | No |
| GET | `/admin/...` | Session management | Yes (optional) |

`X-API-Key` authentication is only enforced when the `API_KEY` environment variable is set. The HF Space is intended to run with `API_KEY` unset so the validator can access all required endpoints freely.

## Section 9 — Local Setup

Prerequisites:

- Python 3.11
- `pip`
- Docker (optional, for container verification)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the API locally:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Health check:

```bash
curl http://localhost:7860/health
```

Windows PowerShell variant:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
Invoke-RestMethod "http://localhost:7860/health"
```

Docker variant:

```powershell
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

## Section 10 — Running the Baseline

Dry-run:

```bash
python inference.py --dry-run
```

Dry-run (PowerShell):

```powershell
.\.venv\Scripts\python inference.py --dry-run
```

Live Gemini:

```powershell
$env:GEMINI_API_KEY="your-key"
.\.venv\Scripts\python inference.py --provider gemini --model gemini-2.5-flash
```

Live OpenAI:

```powershell
$env:OPENAI_API_KEY="your-key"
.\.venv\Scripts\python inference.py --provider openai --model gpt-4o-mini
```

| Variable | Purpose | Required For |
|---|---|---|
| API_BASE_URL | FastAPI server URL | Always |
| MODEL_NAME | LLM model identifier | Live mode only |
| HF_TOKEN | Hugging Face deployment or hosted access token | HF deployment |

## Section 11 — Log Format

The evaluator-facing stdout format is:

```text
[START] {"task_id": "easy", "session_id": "..."}
[STEP]  {"step": 1, "action": {"action_type": "drop_duplicates"}, "reward": 0.18, "done": false}
[END]   {"task_id": "easy", "score": 1.0000, "steps": 5}
```

## Section 12 — Project Structure

```text
DataClean_OpenEnv/
├── api/
│   └── main.py                  # FastAPI routes and session management
├── environment/
│   ├── env.py                   # Core reset/step/state environment loop
│   ├── logging_config.py        # structlog configuration and context binding
│   ├── models.py                # Pydantic request/response contracts
│   ├── graders/
│   │   ├── base_grader.py       # Shared similarity helpers
│   │   ├── grader_easy.py       # Easy task grader
│   │   ├── grader_medium.py     # Medium task grader
│   │   ├── grader_hard.py       # Hard task grader
│   │   └── grader_random.py     # Procedural task grader
│   └── tasks/
│       ├── base_task.py         # Shared task interface
│       ├── task_easy.py         # Employee Records task
│       ├── task_medium.py       # Sales Transactions task
│       ├── task_hard.py         # Hospital Admissions task
│       └── task_random.py       # Procedural random task
├── server/
│   └── app.py                   # Server entrypoint shim
├── tests/
│   ├── test_api.py              # HTTP contract coverage
│   ├── test_auth.py             # Optional API key auth coverage
│   ├── test_env.py              # Environment transition logic
│   ├── test_graders.py          # Grader correctness and score bounds
│   ├── test_inference.py        # Dry-run baseline and provider config
│   ├── test_logging.py          # structlog context binding
│   └── test_models.py           # Pydantic model validation
├── compliance_check.py          # Full competition audit script
├── Dockerfile                   # HF Space / container build definition
├── inference.py                 # Root baseline runner
├── openenv.yaml                 # OpenEnv benchmark specification
├── requirements.txt             # Python dependencies
└── README.md                    # Competition-facing documentation
```

## Section 13 — Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Expected: the official competition checklist references 26 passing tests; the current local regression suite contains additional auth and logging coverage beyond that baseline.

| File | What It Tests |
|---|---|
| `test_api.py` | All HTTP endpoints, status codes |
| `test_env.py` | Environment step/reset/state logic |
| `test_graders.py` | Grader score ranges and correctness |
| `test_models.py` | Pydantic model validation |
| `test_inference.py` | Dry-run log format compliance |
| `test_auth.py` | API key auth enabled/disabled modes |
| `test_logging.py` | structlog context binding |

## Section 14 — Competition Compliance

| Check | Status | How Verified |
|---|---|---|
| HF Space deploys | ⚠️ | Live ping currently refused connections during the 2026-04-08 audit; local `/health` and `/reset` contracts pass |
| OpenEnv spec compliant | ✅ | `openenv.yaml` + typed models + `compliance_check.py` |
| Dockerfile builds | ✅ | Verified clean build (12/12 steps) |
| Baseline reproduces | ✅ | Dry-run scores: easy `1.0000`, medium `0.9811`, hard `0.8332` |
| 3+ tasks with graders | ✅ | easy / medium / hard + random |
| Log format correct | ✅ | `[START]` / `[STEP]` / `[END]` verified by `compliance_check.py` |
| Runtime < 20 min | ✅ | Local compliance run completed in ~10s; competition reference budget remains ~18s |
| inference.py in root | ✅ | Confirmed |
| OpenAI client used | ✅ | OpenAI SDK drives all live LLM calls |

## Section 15 — Deployment

HF Space URL:

```text
https://sanvihs2005-data-cleaning-env.hf.space
```

Docker commands:

```powershell
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

Environment variables to set for hosted deployment:

- `API_KEY` only if you want to protect admin and submit endpoints
- `OPENAI_API_KEY` for live OpenAI runs
- `GEMINI_API_KEY` for live Gemini runs
- `API_BASE_URL` for the runner target
- `MODEL_NAME` for live-model selection
- `HF_TOKEN` for hosted Hugging Face access when needed
