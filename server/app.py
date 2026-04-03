"""OpenEnv server entry point wrapper around the deployed FastAPI app."""

from __future__ import annotations

import os

import uvicorn

from api.main import app


def main() -> None:
    """Run the FastAPI app using environment-configurable host and port."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("APP_PORT", "7860")))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
