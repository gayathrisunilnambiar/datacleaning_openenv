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
        self.assertEqual(response.json()["status"], "healthy")

    def test_metadata_schema_and_mcp_endpoints_exist(self) -> None:
        metadata_response = self.client.get("/metadata")
        self.assertEqual(metadata_response.status_code, 200)
        self.assertEqual(metadata_response.json()["name"], "data-cleaning-env")

        schema_response = self.client.get("/schema")
        self.assertEqual(schema_response.status_code, 200)
        schema_payload = schema_response.json()
        self.assertIn("action", schema_payload)
        self.assertIn("observation", schema_payload)
        self.assertIn("state", schema_payload)

        mcp_response = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(mcp_response.status_code, 200)
        self.assertEqual(mcp_response.json()["jsonrpc"], "2.0")

    def test_tasks_endpoint_lists_all_tasks(self) -> None:
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(sorted(item["task_id"] for item in payload), ["easy", "hard", "medium", "random"])

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

    def test_reset_defaults_to_easy_when_task_id_omitted(self) -> None:
        response = self.client.post("/reset", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["observation"]["task_id"], "easy")

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
