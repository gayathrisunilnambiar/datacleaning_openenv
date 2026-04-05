"""Regression test for the dry-run baseline runner."""

from __future__ import annotations

import io
import socket
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import requests
import uvicorn

from api.main import app
from inference import PROVIDER_CONFIGS, main, resolve_provider_config


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class InferenceTests(unittest.TestCase):
    """The dry-run baseline should complete all tasks without crashing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=cls.port, log_level="error")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()

        base_url = f"http://127.0.0.1:{cls.port}"
        for _ in range(50):
            try:
                requests.get(f"{base_url}/health", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for the local test server to start.")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.should_exit = True
        cls.thread.join(timeout=5)

    def test_dry_run_main_exits_zero_and_emits_end_lines(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            with mock.patch(
                "sys.argv",
                [
                    "inference.py",
                    "--dry-run",
                    "--base-url",
                    f"http://127.0.0.1:{self.port}",
                ],
            ):
                exit_code = main()

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.count("[START]"), 3)
        self.assertEqual(output.count("[END]"), 3)

    def test_resolve_provider_config_for_gemini(self) -> None:
        provider, model_name = resolve_provider_config("gemini", None, None)
        self.assertEqual(provider.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(provider.base_url, PROVIDER_CONFIGS["gemini"].base_url)
        self.assertEqual(model_name, "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
