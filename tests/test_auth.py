"""Tests for optional API key authentication.

Covers two modes:
  1. Auth disabled (API_KEY env var NOT set) — all routes pass freely.
  2. Auth enabled  (API_KEY env var IS set)  — protected routes enforce key,
     but public/competition routes remain open.
"""

from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class TestAuthDisabled(unittest.TestCase):
    """When API_KEY env var is not set — all routes must pass."""

    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("API_KEY", None)
            import api.main as main_module

            importlib.reload(main_module)
            self.client = TestClient(main_module.app)

    def test_health_no_key(self):
        """Public route — must return 200 with no key."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_reset_no_key(self):
        """Public route — must return 200 with no key."""
        response = self.client.post("/reset", json={"task_id": "easy"})
        self.assertEqual(response.status_code, 200)

    def test_validate_no_key(self):
        """Public route — must return 200 with no key."""
        response = self.client.post("/validate")
        self.assertEqual(response.status_code, 200)

    def test_admin_no_key_when_auth_disabled(self):
        """Admin route — must return 200 when auth is disabled."""
        response = self.client.get("/admin/sessions")
        self.assertEqual(response.status_code, 200)


class TestAuthEnabled(unittest.TestCase):
    """When API_KEY env var IS set — protected routes enforce key."""

    VALID_KEY = "test-secret-key-123"

    def setUp(self):
        os.environ["API_KEY"] = self.VALID_KEY
        # Reload app so _API_KEY picks up the new env var value
        import api.main as main_module

        importlib.reload(main_module)
        self.client = TestClient(main_module.app)

    def tearDown(self):
        os.environ.pop("API_KEY", None)

    def test_public_route_no_key_still_works(self):
        """Public routes must NEVER require a key even when auth enabled."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_public_reset_no_key_still_works(self):
        """/reset must work without key even when auth enabled."""
        response = self.client.post("/reset", json={"task_id": "easy"})
        self.assertEqual(response.status_code, 200)

    def test_admin_no_key_returns_401(self):
        """Admin route must return 401 with no key when auth enabled."""
        response = self.client.get("/admin/sessions")
        self.assertEqual(response.status_code, 401)

    def test_admin_wrong_key_returns_401(self):
        """Admin route must return 401 with wrong key."""
        response = self.client.get(
            "/admin/sessions", headers={"X-API-Key": "wrong-key"}
        )
        self.assertEqual(response.status_code, 401)

    def test_admin_correct_key_returns_200(self):
        """Admin route must return 200 with correct key."""
        response = self.client.get(
            "/admin/sessions", headers={"X-API-Key": self.VALID_KEY}
        )
        self.assertEqual(response.status_code, 200)

    def test_validator_critical_endpoints_no_key(self):
        """
        Simulate competition validator — hits all OpenEnv endpoints
        without any API key. All must return 200, never 401.
        This test failing = submission disqualified.
        """
        validator_endpoints = [
            ("GET", "/health"),
            ("GET", "/tasks"),
            ("POST", "/validate"),
            ("GET", "/"),
        ]
        for method, path in validator_endpoints:
            with self.subTest(method=method, path=path):
                if method == "GET":
                    response = self.client.get(path)
                else:
                    response = self.client.post(path)
                self.assertNotEqual(
                    response.status_code,
                    401,
                    f"{method} {path} returned 401 — "
                    f"competition validator would be blocked",
                )


if __name__ == "__main__":
    unittest.main()
