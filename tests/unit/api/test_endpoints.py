"""Unit tests for API endpoints with mocked dependencies.

These tests verify that:
- Endpoints accept valid requests and return correct response shapes
- Auth is enforced (dev mode bypasses key validation)
- Error handlers map GMAOError to proper HTTP status codes
- Query parameter validation works (top_k bounds, etc.)

All heavy dependencies (orchestrators, DB, Qdrant, LLM) are mocked
so these tests run without external services.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import os

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import app


# =====================================================================
# Unset RAG_API_KEY so auth runs in dev mode during tests
# =====================================================================
@pytest.fixture(autouse=True)
def _dev_mode_auth(monkeypatch):
    """Force dev mode (no API key check) for all endpoint tests."""
    monkeypatch.delenv("RAG_API_KEY", raising=False)


# =====================================================================
# Auth header used by all endpoint tests (dev mode — key not checked)
# =====================================================================
AUTH_HEADERS = {"Authorization": "Bearer dev-test-token"}


# =====================================================================
# Helpers — fake domain objects
# =====================================================================

@dataclass
class FakeRetrievedChunk:
    chunk_id: str = "c1"
    content: str = "Le moteur vibre"
    score: float = 0.95
    rank: int = 1
    source_name: str = "manual.pdf"
    source_type: str = "document"
    id_document: int | None = None
    id_panne: int | None = None
    id_equipement: int | None = None
    retrieval_strategy: str = "bm25"


@dataclass
class FakeRankedChunk:
    chunk_id: str = "c1"
    content: str = "Le moteur vibre"
    source_name: str = "manual.pdf"
    source_type: str = "document"
    retrieval_score: float = 0.95
    rerank_score: float = 0.98
    rank: int = 1
    id_document: int | None = None
    id_panne: int | None = None
    id_equipement: int | None = None
    retrieval_strategy: str = "bm25"
    reranker_strategy: str = "cross_encoder"


@dataclass
class FakeRetrievalReport:
    query: str = "test"
    results: tuple = ()
    strategy_name: str = "bm25"
    total_candidates: int = 5
    duration_ms: float = 10.0


@dataclass
class FakeCitation:
    chunk_id: str = "c1"
    source_name: str = "manual.pdf"
    source_type: str = "document"
    rerank_score: float = 0.9


@dataclass
class FakeLLMResponse:
    answer: str = "Le moteur vibre à cause du rotor."
    query: str = "test"
    strategy_name: str = "openai"
    citations: tuple = ()


# =====================================================================
# Fixture — build a TestClient with dependency overrides
# =====================================================================

@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient with all orchestrators mocked via dependency_overrides."""
    _setup_overrides()
    yield TestClient(app, raise_server_exceptions=False)
    _teardown_overrides()


def _make_registry(strategies: list[str]) -> MagicMock:
    """Create a mock registry that returns the given strategy names."""
    reg = MagicMock()
    reg.supported_strategies.return_value = strategies
    return reg


def _setup_overrides() -> None:
    """Configure dependency overrides for all orchestrators."""

    # --- Retrieval orchestrator ---
    retrieval_orch = MagicMock()
    retrieval_orch.retrieve.return_value = FakeRetrievalReport(
        results=(FakeRetrievedChunk(),),
    )
    retrieval_orch.strategy_name = "bm25"
    retrieval_orch.registry = _make_registry(["bm25", "hybrid"])

    # --- Reranker orchestrator ---
    reranker_orch = MagicMock()
    reranker_orch.rerank.return_value = [FakeRankedChunk()]
    reranker_orch.strategy_name = "cross_encoder"
    reranker_orch.registry = _make_registry(["cross_encoder", "none"])

    # --- LLM orchestrator ---
    llm_orch = MagicMock()
    llm_orch.generate.return_value = FakeLLMResponse(
        citations=(FakeCitation(),),
    )
    llm_orch.registry = _make_registry(["openai", "gemini"])

    # --- Embedding orchestrator ---
    embedding_orch = MagicMock()
    embedding_orch.registry = _make_registry(["sentence_transformers"])

    # --- Ingestion orchestrators ---
    data_source_orch = MagicMock()
    parser_orch = MagicMock()
    chunker_orch = MagicMock()
    storage_orch = MagicMock()

    # Register overrides
    app.dependency_overrides[deps.get_retrieval_orchestrator] = lambda: retrieval_orch
    app.dependency_overrides[deps.get_reranker_orchestrator] = lambda: reranker_orch
    app.dependency_overrides[deps.get_llm_orchestrator] = lambda: llm_orch
    app.dependency_overrides[deps.get_embedding_orchestrator] = lambda: embedding_orch
    app.dependency_overrides[deps.get_data_source_orchestrator] = lambda: data_source_orch
    app.dependency_overrides[deps.get_parser_orchestrator] = lambda: parser_orch
    app.dependency_overrides[deps.get_chunker_orchestrator] = lambda: chunker_orch
    app.dependency_overrides[deps.get_storage_orchestrator] = lambda: storage_orch


def _teardown_overrides() -> None:
    """Clear all dependency overrides."""
    app.dependency_overrides.clear()


# =====================================================================
# POST /api/v1/rag/search
# =====================================================================

class TestSearchEndpoint:
    def test_search_with_rerank_and_generate(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/search", json={
            "query": "Pourquoi la pompe vibre ?",
            "rerank": True,
            "generate": True,
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "Pourquoi la pompe vibre ?"
        assert isinstance(data["answer"], str)
        assert isinstance(data["results"], list)
        assert isinstance(data["citations"], list)
        assert data["strategy_info"]["retrieval"] == "bm25"
        assert data["duration_ms"] >= 0

    def test_search_no_rerank(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/search", json={
            "query": "test",
            "rerank": False,
            "generate": False,
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == ""
        assert data["strategy_info"]["reranker"] is None

    def test_search_empty_query_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/search", json={"query": ""}, headers=AUTH_HEADERS)
        assert resp.status_code == 422  # Pydantic validation

    def test_search_top_k_bounds(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/search", json={"query": "x", "top_k": 51}, headers=AUTH_HEADERS)
        assert resp.status_code == 422

    def test_search_requires_auth(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/search", json={"query": "test"})
        assert resp.status_code == 422  # missing Authorization header

    def test_search_llm_failure_returns_graceful_response(self, client: TestClient) -> None:
        """When LLM fails (rate limit, connection, etc.), the endpoint
        returns 200 with empty answer + llm_error instead of a raw error."""
        from app.exceptions import LLMRateLimitError

        # Override LLM orchestrator to simulate failure
        failing_llm = MagicMock()
        failing_llm.generate.side_effect = LLMRateLimitError(
            message="OpenAI rate limit or quota exceeded."
        )
        app.dependency_overrides[deps.get_llm_orchestrator] = lambda: failing_llm
        try:
            resp = client.post("/api/v1/rag/search", json={
                "query": "test",
                "rerank": True,
                "generate": True,
            }, headers=AUTH_HEADERS)
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == ""
            assert data["llm_error"] is not None
            assert "rate limit" in data["llm_error"].lower()
            assert isinstance(data["results"], list)  # chunks still returned
            assert data["strategy_info"]["retrieval"] == "bm25"
        finally:
            _setup_overrides()

    def test_search_llm_success_has_no_llm_error(self, client: TestClient) -> None:
        """When LLM succeeds, llm_error must be null."""
        resp = client.post("/api/v1/rag/search", json={
            "query": "test",
            "generate": True,
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_error"] is None
        assert len(data["answer"]) > 0


# =====================================================================
# POST /api/v1/rag/retrieve
# =====================================================================

class TestRetrieveEndpoint:
    def test_retrieve(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/retrieve", json={"query": "vibration"}, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"  # mock returns FixedRetrievalReport with query="test"
        assert data["strategy_name"] == "bm25"
        assert isinstance(data["results"], list)

    def test_retrieve_with_filters(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/retrieve", json={
            "query": "test",
            "filters": {"id_equipement": 42, "source_type": "panne"},
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200


# =====================================================================
# POST /api/v1/rag/rerank
# =====================================================================

class TestRerankEndpoint:
    def test_rerank(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/rerank", json={
            "query": "test",
            "candidates": [{
                "chunk_id": "c1",
                "content": "Le moteur vibre",
                "score": 0.95,
                "rank": 1,
                "source_name": "manual.pdf",
                "source_type": "document",
            }],
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert isinstance(data["results"], list)

    def test_rerank_empty_candidates_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/rerank", json={
            "query": "test",
            "candidates": [],
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 422


# =====================================================================
# POST /api/v1/ingest/database
# =====================================================================

class TestIngestDatabaseEndpoint:
    def test_ingest_database(self, client: TestClient) -> None:
        resp = client.post("/api/v1/ingest/database", json={
            "host": "localhost",
            "database": "gmao",
            "user": "root",
            "password": "",
            "table": "interventions",
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_files"] == 1
        assert isinstance(data["results"], list)


# =====================================================================
# POST /api/v1/ingest/files
# =====================================================================

class TestIngestFilesEndpoint:
    def test_ingest_batch(self, client: TestClient) -> None:
        resp = client.post("/api/v1/ingest/files", json={
            "paths": ["/tmp/a.txt", "/tmp/b.txt"],
        }, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_files"] == 2

    def test_ingest_empty_paths_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/ingest/files", json={"paths": []}, headers=AUTH_HEADERS)
        assert resp.status_code == 422


# =====================================================================
# GET /api/v1/health
# =====================================================================

class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "version" in data

    def test_health_no_auth_required(self) -> None:
        """Health endpoint should work without Authorization header."""
        _setup_overrides()
        try:
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.get("/api/v1/health")
            assert resp.status_code == 200
        finally:
            _teardown_overrides()


# =====================================================================
# GET /api/v1/strategies
# =====================================================================

class TestStrategiesEndpoint:
    def test_strategies(self, client: TestClient) -> None:
        resp = client.get("/api/v1/strategies", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "bm25" in data["retrieval"]
        assert "cross_encoder" in data["reranker"]
        assert "openai" in data["llm"]
        assert "sentence_transformers" in data["embedding"]


# =====================================================================
# Error handler mapping
# =====================================================================

class TestErrorHandlers:
    def test_validation_error_returns_400(self, client: TestClient) -> None:
        """EmptyQueryError → 400."""
        from app.exceptions import EmptyQueryError

        retrieval_orch = MagicMock()
        retrieval_orch.retrieve.side_effect = EmptyQueryError()
        retrieval_orch.registry = _make_registry([])

        app.dependency_overrides[deps.get_retrieval_orchestrator] = lambda: retrieval_orch
        try:
            resp = client.post("/api/v1/rag/retrieve", json={"query": "test"}, headers=AUTH_HEADERS)
            assert resp.status_code == 400
        finally:
            _setup_overrides()
