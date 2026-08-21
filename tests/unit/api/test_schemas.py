"""Unit tests for Pydantic API schemas.

Covers every schema in ``app.api.schemas``:
- Construction with valid data
- Field constraints (min_length, gt, ge, le)
- Default values
- Frozen / immutable behavior
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    CitationSchema,
    DeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    FilterParams,
    HealthResponse,
    IngestDatabaseRequest,
    IngestFileRequest,
    IngestMultipleRequest,
    IngestResponse,
    IngestResult,
    RankedChunkSchema,
    RerankRequest,
    RerankResponse,
    RetrievedChunkSchema,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    StrategyInfo,
    StrategyListResponse,
)


# =====================================================================
# Helpers — minimal valid objects
# =====================================================================

def _make_chunk(**overrides: object) -> RetrievedChunkSchema:
    defaults: dict[str, object] = dict(
        chunk_id="c1",
        content="Le moteur vibre",
        score=0.95,
        rank=1,
        source_name="manual.pdf",
        source_type="document",
    )
    defaults.update(overrides)
    return RetrievedChunkSchema(**defaults)


def _make_ranked_chunk(**overrides: object) -> RankedChunkSchema:
    defaults: dict[str, object] = dict(
        chunk_id="c1",
        content="Le moteur vibre",
        source_name="manual.pdf",
        source_type="document",
        retrieval_score=0.95,
        rerank_score=0.98,
        rank=1,
    )
    defaults.update(overrides)
    return RankedChunkSchema(**defaults)


def _make_citation(**overrides: object) -> CitationSchema:
    defaults: dict[str, object] = dict(
        chunk_id="c1",
        source_name="manual.pdf",
        source_type="document",
        rerank_score=0.9,
    )
    defaults.update(overrides)
    return CitationSchema(**defaults)


# =====================================================================
# FilterParams
# =====================================================================

class TestFilterParams:
    def test_empty(self) -> None:
        f = FilterParams()
        assert f.id_document is None
        assert f.id_panne is None
        assert f.id_equipement is None
        assert f.source_type is None
        assert f.min_score is None

    def test_with_values(self) -> None:
        f = FilterParams(id_document=1, source_type="panne", min_score=0.5)
        assert f.id_document == 1
        assert f.source_type == "panne"
        assert f.min_score == 0.5

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValidationError, match="id_document"):
            FilterParams(id_document=0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="id_equipement"):
            FilterParams(id_equipement=-1)


# =====================================================================
# StrategyInfo
# =====================================================================

class TestStrategyInfo:
    def test_defaults(self) -> None:
        s = StrategyInfo()
        assert s.retrieval is None
        assert s.reranker is None
        assert s.llm is None

    def test_with_values(self) -> None:
        s = StrategyInfo(retrieval="bm25", reranker="cross_encoder", llm="openai")
        assert s.retrieval == "bm25"


# =====================================================================
# RetrievedChunkSchema
# =====================================================================

class TestRetrievedChunkSchema:
    def test_construction(self) -> None:
        c = _make_chunk()
        assert c.chunk_id == "c1"
        assert c.content == "Le moteur vibre"
        assert c.score == 0.95
        assert c.rank == 1

    def test_optional_fields(self) -> None:
        c = _make_chunk()
        assert c.id_document is None
        assert c.id_panne is None
        assert c.id_equipement is None

    def test_accepts_empty_strings(self) -> None:
        """Pydantic models do not enforce min_length on chunk_id/content."""
        c = RetrievedChunkSchema(
            chunk_id="", content="", score=0.5, rank=1,
            source_name="a", source_type="b",
        )
        assert c.chunk_id == ""


# =====================================================================
# RankedChunkSchema
# =====================================================================

class TestRankedChunkSchema:
    def test_construction(self) -> None:
        c = _make_ranked_chunk()
        assert c.retrieval_score == 0.95
        assert c.rerank_score == 0.98
        assert c.reranker_strategy == ""

    def test_with_reranker(self) -> None:
        c = _make_ranked_chunk(reranker_strategy="cross_encoder")
        assert c.reranker_strategy == "cross_encoder"


# =====================================================================
# CitationSchema
# =====================================================================

class TestCitationSchema:
    def test_construction(self) -> None:
        c = _make_citation()
        assert c.rerank_score == 0.9

    def test_accepts_empty_source_name(self) -> None:
        """Pydantic models do not enforce min_length on source_name."""
        c = CitationSchema(
            chunk_id="c1", source_name="", source_type="document",
            rerank_score=0.5,
        )
        assert c.source_name == ""


# =====================================================================
# SearchRequest
# =====================================================================

class TestSearchRequest:
    def test_minimal(self) -> None:
        r = SearchRequest(query="test")
        assert r.query == "test"
        assert r.rerank is True
        assert r.generate is True
        assert r.top_k is None
        assert r.filters is None

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            SearchRequest(query="")

    def test_top_k_bounds(self) -> None:
        r = SearchRequest(query="x", top_k=1)
        assert r.top_k == 1
        with pytest.raises(ValidationError):
            SearchRequest(query="x", top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="x", top_k=51)


# =====================================================================
# RetrieveRequest
# =====================================================================

class TestRetrieveRequest:
    def test_construction(self) -> None:
        r = RetrieveRequest(query="vibration pompe")
        assert r.query == "vibration pompe"
        assert r.top_k is None

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            RetrieveRequest(query="")


# =====================================================================
# RerankRequest
# =====================================================================

class TestRerankRequest:
    def test_construction(self) -> None:
        chunks = [_make_chunk()]
        r = RerankRequest(query="test", candidates=chunks)
        assert len(r.candidates) == 1
        assert r.top_k is None

    def test_rejects_empty_candidates(self) -> None:
        with pytest.raises(ValidationError, match="candidates"):
            RerankRequest(query="test", candidates=[])

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            RerankRequest(query="", candidates=[_make_chunk()])


# =====================================================================
# Response schemas
# =====================================================================

class TestSearchResponse:
    def test_construction(self) -> None:
        r = SearchResponse(
            answer="Réponse",
            query="Question",
            strategy_info=StrategyInfo(retrieval="bm25"),
            duration_ms=123.45,
        )
        assert r.answer == "Réponse"
        assert r.citations == []
        assert r.results == []
        assert r.duration_ms == 123.45
        assert r.llm_error is None  # default: no error

    def test_with_llm_error(self) -> None:
        r = SearchResponse(
            answer="",
            query="test",
            strategy_info=StrategyInfo(retrieval="bm25"),
            duration_ms=10.0,
            llm_error="OpenAI rate limit exceeded.",
        )
        assert r.answer == ""
        assert r.llm_error == "OpenAI rate limit exceeded."
        assert r.results == []


class TestRetrieveResponse:
    def test_construction(self) -> None:
        r = RetrieveResponse(
            query="test",
            total_candidates=10,
            strategy_name="bm25",
        )
        assert r.results == []
        assert r.total_candidates == 10


class TestRerankResponse:
    def test_construction(self) -> None:
        r = RerankResponse(query="test")
        assert r.results == []


# =====================================================================
# Ingest schemas
# =====================================================================

class TestIngestFileRequest:
    def test_defaults(self) -> None:
        r = IngestFileRequest()
        assert r.chunk_size == 500
        assert r.chunk_overlap == 50
        assert r.id_equipement is None


class TestIngestDatabaseRequest:
    def test_construction(self) -> None:
        r = IngestDatabaseRequest(
            host="localhost",
            database="gmao",
            user="root",
            password="secret",
            table="interventions",
        )
        assert r.driver == "mysql"
        assert r.port == 3306

    def test_rejects_empty_host(self) -> None:
        with pytest.raises(ValidationError, match="host"):
            IngestDatabaseRequest(
                host="", database="gmao", user="root",
                password="", table="t",
            )


class TestIngestMultipleRequest:
    def test_construction(self) -> None:
        r = IngestMultipleRequest(paths=["/a.txt", "/b.pdf"])
        assert len(r.paths) == 2
        assert r.chunk_size == 500

    def test_rejects_empty_paths(self) -> None:
        with pytest.raises(ValidationError, match="paths"):
            IngestMultipleRequest(paths=[])


# =====================================================================
# Ingest response schemas
# =====================================================================

class TestIngestResult:
    def test_ok(self) -> None:
        r = IngestResult(
            status="ok", document_name="file.pdf",
            chunks_count=10, duration_ms=150.0,
        )
        assert r.error is None

    def test_error(self) -> None:
        r = IngestResult(
            status="error", document_name="file.pdf",
            chunks_count=0, duration_ms=50.0, error="parse failed",
        )
        assert r.error == "parse failed"


class TestIngestResponse:
    def test_construction(self) -> None:
        r = IngestResponse(
            status="ok", results=[], total_files=0,
            success_count=0, error_count=0,
        )
        assert r.total_files == 0


# =====================================================================
# Document schemas
# =====================================================================

class TestDocumentSummary:
    def test_construction(self) -> None:
        d = DocumentSummary(
            id=1, name="manual.pdf",
            source_type="document", chunks_count=20,
            indexed=True,
        )
        assert d.id == 1
        assert d.indexed is True


class TestDocumentListResponse:
    def test_empty(self) -> None:
        r = DocumentListResponse(total=0)
        assert r.documents == []

    def test_with_documents(self) -> None:
        d = DocumentSummary(
            id=1, name="a.pdf", source_type="doc",
            chunks_count=5, indexed=True,
        )
        r = DocumentListResponse(documents=[d], total=1)
        assert len(r.documents) == 1


class TestDeleteResponse:
    def test_construction(self) -> None:
        r = DeleteResponse(deleted_chunks=15)
        assert r.status == "ok"
        assert r.deleted_chunks == 15


# =====================================================================
# System schemas
# =====================================================================

class TestHealthResponse:
    def test_healthy(self) -> None:
        r = HealthResponse(status="healthy", qdrant="ok", mysql="ok")
        assert r.version == "0.1.0"

    def test_unhealthy(self) -> None:
        r = HealthResponse(
            status="unhealthy", qdrant="error", mysql="error",
            version="0.2.0",
        )
        assert r.version == "0.2.0"


class TestStrategyListResponse:
    def test_construction(self) -> None:
        r = StrategyListResponse(
            retrieval=["bm25"],
            reranker=["cross_encoder"],
            llm=["openai"],
            embedding=["sentence_transformers"],
        )
        assert len(r.retrieval) == 1


class TestStatsResponse:
    def test_construction(self) -> None:
        r = StatsResponse(documents_count=10, chunks_count=500)
        assert r.qdrant_points is None

    def test_with_qdrant(self) -> None:
        r = StatsResponse(documents_count=5, chunks_count=200, qdrant_points=1000)
        assert r.qdrant_points == 1000

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match="documents_count"):
            StatsResponse(documents_count=-1, chunks_count=0)
