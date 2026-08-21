"""Unit tests for API authentication middleware.

Covers ``app.api.auth.verify_api_key``:
- Dev mode (no key configured) → always passes
- Valid Bearer token → passes
- Missing header → 401
- Malformed header → 401
- Wrong key → 401
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from app.api.auth import verify_api_key


# =====================================================================
# Test app — minimal FastAPI app to exercise the dependency
# =====================================================================

_app = FastAPI()


@_app.get("/test-auth")
async def _test_auth(token: str = Header(..., alias="Authorization")) -> str:
    """Endpoint that requires auth via verify_api_key."""
    return await verify_api_key(token)


_client = TestClient(_app)


# =====================================================================
# Tests
# =====================================================================

class TestVerifyApiKey:
    """Test the verify_api_key dependency function."""

    def test_dev_mode_no_key(self) -> None:
        """When RAG_API_KEY is not set, any header should pass (dev mode)."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RAG_API_KEY", None)
            resp = _client.get("/test-auth", headers={"Authorization": "Bearer anything"})
            assert resp.status_code == 200
            assert resp.json() == "dev-mode"

    def test_valid_token(self) -> None:
        """Valid Bearer token should pass when RAG_API_KEY is configured."""
        with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}):
            resp = _client.get("/test-auth", headers={"Authorization": "Bearer secret-key"})
            assert resp.status_code == 200
            assert resp.json() == "secret-key"

    def test_wrong_token(self) -> None:
        """Wrong token should return 401."""
        with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}):
            resp = _client.get("/test-auth", headers={"Authorization": "Bearer wrong"})
            assert resp.status_code == 401
            assert "Invalid API key" in resp.json()["detail"]

    def test_missing_header(self) -> None:
        """Missing Authorization header should return 422 (FastAPI validation)."""
        with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}):
            resp = _client.get("/test-auth")
            assert resp.status_code == 422

    def test_malformed_no_bearer_prefix(self) -> None:
        """Header without 'Bearer ' prefix should return 401."""
        with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}):
            resp = _client.get("/test-auth", headers={"Authorization": "secret-key"})
            assert resp.status_code == 401
            assert "Missing or malformed" in resp.json()["detail"]

    def test_malformed_wrong_scheme(self) -> None:
        """Header with wrong scheme should return 401."""
        with patch.dict(os.environ, {"RAG_API_KEY": "secret-key"}):
            resp = _client.get("/test-auth", headers={"Authorization": "Basic secret-key"})
            assert resp.status_code == 401
