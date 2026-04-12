"""Regression test for the dry-run baseline runner."""

from __future__ import annotations

import io
import os
import socket
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import requests
import uvicorn

from api.main import app
from inference import API_BASE_URL, HF_TOKEN, LOCAL_IMAGE_NAME, MODEL_NAME, main


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

    def test_dry_run_main_exits_zero_and_emits_required_log_shapes(self) -> None:
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
        self.assertGreaterEqual(output.count("[STEP]"), 3)
        self.assertEqual(output.count("[END]"), 3)
        self.assertIn("env=data-cleaning-env", output)
        self.assertIn("reward=", output)
        self.assertIn("done=", output)
        self.assertIn("score=", output)

    def test_submission_env_vars_exist_with_only_allowed_defaults(self) -> None:
        if "API_BASE_URL" not in os.environ:
            self.assertEqual(API_BASE_URL, "https://router.huggingface.co/v1")
        if "MODEL_NAME" not in os.environ:
            self.assertEqual(MODEL_NAME, "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(HF_TOKEN, os.getenv("HF_TOKEN"))
        self.assertEqual(LOCAL_IMAGE_NAME, os.getenv("LOCAL_IMAGE_NAME"))


if __name__ == "__main__":
    unittest.main()
