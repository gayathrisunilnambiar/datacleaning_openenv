---
title: DataCleaningEnv
emoji: 🧹
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DataCleaningEnv

DataCleaningEnv is an OpenEnv-style benchmark for iterative tabular data cleaning. An agent receives a dirty table, chooses one cleaning action at a time, and is graded on how much closer the table gets to a hidden clean target.

## Why this benchmark matters

Most tabular evaluations only look at the final answer. This benchmark measures the cleaning process itself:

- whether the agent picks the right next transformation
- whether it improves the table without looping on no-op actions
- whether it can normalize types, values, and row quality under a step budget

The result is a small, deterministic benchmark that is useful for testing real agent behavior instead of only final predictions.

## Task suite

The environment ships with three seeded tasks:

- `easy`: employee records with duplicate rows and missing numeric values
- `medium`: sales transactions with mixed numeric formats, mixed date formats, corrupted totals, and price outliers
- `hard`: hospital admissions with row duplication, missing values, mixed date formats, inconsistent category labels, mixed weight units, blood-type typos, and heterogeneous boolean encoding

## Design overview

Each task provides:

- a deterministic dirty dataframe
- a hidden ground-truth dataframe
- a maximum step budget
- a constrained action space for iterative cleaning

The benchmark combines:

- dense per-step reward shaping
- a no-op penalty for redundant actions
- a final task-specific score in `[0.0, 1.0]`

Current dirty-data calibration:

- `easy`: `0.3921`
- `medium`: `0.2930`
- `hard`: `0.2493`

Current dry-run baseline:

- `easy`: `1.0000`
- `medium`: `0.9811`
- `hard`: `0.8332`

## Repository layout

- `environment/tasks/`: deterministic task generators
- `environment/graders/`: fast similarity helpers and final task graders
- `environment/env.py`: the environment loop and action handlers
- `environment/models.py`: shared typed contracts
- `api/main.py`: FastAPI surface
- `inference.py`: dry-run and live-provider baseline runner
- `tests/`: regression tests

## Local setup

Install dependencies:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run the API:

```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Check health:

```powershell
curl http://localhost:7860/health
```

Run the dry-run baseline:

```powershell
.\.venv\Scripts\python inference.py --dry-run
```

Run the test suite:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

## Live LLM mode

`inference.py` supports chat-completions-compatible providers through the OpenAI SDK interface.

Gemini example:

```powershell
$env:GEMINI_API_KEY="your-key"
.\.venv\Scripts\python inference.py --provider gemini --model gemini-2.5-flash
```

OpenAI example:

```powershell
$env:OPENAI_API_KEY="your-key"
.\.venv\Scripts\python inference.py --provider openai --model gpt-4o-mini
```

Notes:

- The runner always emits `[START]`, `[STEP]`, and `[END]` logs.
- If the provider returns invalid JSON or a transient error, the runner falls back to the deterministic policy instead of crashing.
- A real Gemini-backed end-to-end local run was verified. The provided key hit Gemini free-tier per-minute quota limits after the first few requests, so later steps completed through the built-in fallback path.

## API contract

### `POST /reset`

Request:

```json
{
  "task_id": "easy"
}
```

### `POST /step`

Request:

```json
{
  "session_id": "uuid",
  "action": {
    "action_type": "fill_nulls",
    "column": "age",
    "params": {
      "strategy": "median"
    }
  }
}
```

### `GET /state`

Use `GET /state?session_id=<id>` to inspect the current observation without mutating the episode.

### `POST /validate`

Runs an internal deterministic self-check and always returns typed JSON, even if a check fails.

## Supported actions

- `drop_duplicates`
- `fill_nulls`
- `cast_column`
- `remove_outliers`
- `rename_column`
- `normalize_values`
- `submit`

## Deployment

### Local Docker

Build:

```powershell
docker build -t data-cleaning-env-local .
```

Run:

```powershell
docker run -d --name data-cleaning-env -p 7860:7860 data-cleaning-env-local
```

### Hugging Face Spaces

This repo is already structured for a Docker Space:

- `Dockerfile` exposes port `7860`
- this README contains Space front matter
- `/health` is available for startup checks

Recommended deployment flow:

1. Create a Docker Space on Hugging Face.
2. Push this repository to that Space.
3. Add any needed secrets such as `GEMINI_API_KEY`.
4. Verify the public `/health` and `/validate` endpoints.

Current status:

- Local Docker rehearsal passed.
- Public Hugging Face deployment is still blocked on missing Hugging Face credentials on this machine.

## Validation summary

Verified locally:

- `pip check`
- full `unittest` suite
- live API smoke tests
- dry-run baseline execution
- real Gemini-backed local inference
- Docker build and container startup

## Runtime behavior

- Sessions live in memory and expire after 30 minutes of inactivity.
- Invalid actions do not crash the server; they produce safe typed failures.
- Task generation is deterministic because each task uses a fixed seed.
