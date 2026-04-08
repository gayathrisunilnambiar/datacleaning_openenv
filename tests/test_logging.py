import json
import unittest
from io import StringIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


class TestStructlogContext(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_reset_logs_session_and_task(self):
        """
        POST /reset must emit a log line containing both
        session_id and task_id.
        """
        buf = StringIO()
        with patch("sys.stdout", buf):
            self.client.post("/reset", json={"task_id": "easy"})

        output = buf.getvalue()
        for line in output.splitlines():
            try:
                record = json.loads(line)
                if record.get("event") == "episode_started":
                    self.assertIn("session_id", record)
                    self.assertEqual(record["task_id"], "easy")
                    return
            except json.JSONDecodeError:
                continue
        self.fail("No episode_started log line found in stdout")

    def test_step_logs_include_session(self):
        """
        POST /step must emit a log line with session_id
        inherited from context - not passed explicitly.
        """
        reset = self.client.post("/reset", json={"task_id": "easy"}).json()
        session_id = reset["session_id"]

        buf = StringIO()
        with patch("sys.stdout", buf):
            self.client.post(
                "/step",
                json={
                    "session_id": session_id,
                    "action": {"action_type": "drop_duplicates"},
                },
            )

        output = buf.getvalue()
        for line in output.splitlines():
            try:
                record = json.loads(line)
                if record.get("event") == "step_taken":
                    self.assertEqual(
                        record["session_id"],
                        session_id,
                        "session_id missing from step log",
                    )
                    return
            except json.JSONDecodeError:
                continue
        self.fail("No step_taken log line found in stdout")

    def test_context_does_not_leak_between_requests(self):
        """
        Context vars must be request-scoped.
        A /health call after a /step must NOT contain session_id.
        """
        reset = self.client.post("/reset", json={"task_id": "easy"}).json()
        session_id = reset["session_id"]

        self.client.post(
            "/step",
            json={
                "session_id": session_id,
                "action": {"action_type": "drop_duplicates"},
            },
        )

        buf = StringIO()
        with patch("sys.stdout", buf):
            self.client.get("/health")

        output = buf.getvalue()
        for line in output.splitlines():
            try:
                record = json.loads(line)
                if record.get("event") == "health_check":
                    self.assertNotIn(
                        "session_id",
                        record,
                        "session_id leaked into /health log",
                    )
                    return
            except json.JSONDecodeError:
                continue
        self.fail("No health_check log line found in stdout")
