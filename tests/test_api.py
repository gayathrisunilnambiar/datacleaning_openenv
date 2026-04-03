"""Tests for the FastAPI surface."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app


class ApiTests(unittest.TestCase):
    """Endpoint-level regression checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_tasks_endpoint_lists_all_tasks(self) -> None:
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 3)
        self.assertEqual(sorted(item["task_id"] for item in payload), ["easy", "hard", "medium"])

    def test_reset_step_and_state_flow(self) -> None:
        reset_response = self.client.post("/reset", json={"task_id": "easy"})
        self.assertEqual(reset_response.status_code, 200)
        session_id = reset_response.json()["session_id"]

        step_response = self.client.post(
            "/step",
            json={"session_id": session_id, "action": {"action_type": "drop_duplicates"}},
        )
        self.assertEqual(step_response.status_code, 200)
        self.assertIn("observation", step_response.json())

        state_response = self.client.get(f"/state?session_id={session_id}")
        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(state_response.json()["session_id"], session_id)

    def test_state_unknown_session_returns_404(self) -> None:
        response = self.client.get("/state?session_id=missing-session")
        self.assertEqual(response.status_code, 404)

    def test_validate_endpoint_returns_typed_success_payload(self) -> None:
        response = self.client.post("/validate")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("passed", payload)
        self.assertIn("checks", payload)
        self.assertIn("graders", payload["checks"])

    def test_step_endpoint_rejects_invalid_typed_action(self) -> None:
        reset_response = self.client.post("/reset", json={"task_id": "easy"})
        session_id = reset_response.json()["session_id"]

        response = self.client.post(
            "/step",
            json={"session_id": session_id, "action": {"action_type": "submit", "column": "age"}},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
