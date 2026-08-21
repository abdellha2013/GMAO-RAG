"""Unit tests for RerankerOrchestrator (mocked — no model download)."""
from __future__ import annotations

import pytest

from app.exceptions import RerankerValidationError, RerankingError
from app.models.reranking import RankedChunk
from app.models.retrieval import RetrievedChunk
from app.reranker.base import RerankerStrategy
from app.reranker.orchestrator import RerankerOrchestrator
from app.reranker.registry import RerankerRegistry


class _FakeReranker(RerankerStrategy):
    name = "fake"

    def supports(self, query, candidates):
        return True

    def rerank(self, query, candidates, *, top_k, **kwargs):
        return [
            RankedChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                source_name=c.source_name,
                source_type=c.source_type,
                retrieval_score=c.score,
                rerank_score=1.0 / (i + 1),
                rank=i + 1,
                metadata=dict(c.metadata),
                retrieval_strategy=c.retrieval_strategy,
                reranker_strategy=self.name,
            )
            for i, c in enumerate(candidates[:top_k])
        ]


class _RejectingReranker(RerankerStrategy):
    name = "rejector"

    def supports(self, query, candidates):
        return False

    def rerank(self, query, candidates, *, top_k, **kwargs):
        return []


def _make_candidates(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=str(i),
            content=f"Contenu {i}.",
            score=0.9 - i * 0.1,
            rank=i + 1,
            source_name=f"doc{i}.pdf",
            source_type="document",
            retrieval_strategy="qdrant",
        )
        for i in range(n)
    ]


@pytest.fixture
def orchestrator() -> RerankerOrchestrator:
    registry = RerankerRegistry()
    registry.register(_FakeReranker)
    return RerankerOrchestrator(registry, strategy_name="fake", default_top_k=3, max_top_k=3)


def test_rerank_returns_ranked_chunks(orchestrator: RerankerOrchestrator) -> None:
    results = orchestrator.rerank("query", _make_candidates(3))
    assert len(results) == 3
    assert all(isinstance(r, RankedChunk) for r in results)
    assert results[0].rank == 1


def test_rerank_empty_candidates(orchestrator: RerankerOrchestrator) -> None:
    results = orchestrator.rerank("query", [])
    assert results == []


def test_rerank_empty_query(orchestrator: RerankerOrchestrator) -> None:
    with pytest.raises(RerankerValidationError, match="query"):
        orchestrator.rerank("", _make_candidates())
    with pytest.raises(RerankerValidationError, match="query"):
        orchestrator.rerank("   ", _make_candidates())


@pytest.mark.parametrize("top_k", [0, -1, True, "3"])
def test_rerank_invalid_top_k(orchestrator: RerankerOrchestrator, top_k: object) -> None:
    with pytest.raises(RerankerValidationError, match="top_k"):
        orchestrator.rerank("query", _make_candidates(), top_k=top_k)  # type: ignore[arg-type]


def test_rerank_top_k_truncated_by_max(orchestrator: RerankerOrchestrator) -> None:
    results = orchestrator.rerank("query", _make_candidates(5), top_k=99)
    assert len(results) == 3


def test_rerank_strategy_not_found() -> None:
    registry = RerankerRegistry()
    orch = RerankerOrchestrator(registry, strategy_name="nonexistent")
    with pytest.raises(Exception):
        orch.rerank("query", _make_candidates())


def test_rerank_strategy_rejects_inputs() -> None:
    registry = RerankerRegistry()
    registry.register(_RejectingReranker)
    orch = RerankerOrchestrator(registry, strategy_name="rejector")
    with pytest.raises(RerankerValidationError, match="does not support"):
        orch.rerank("query", _make_candidates())


def test_orchestrator_init_validates_registry() -> None:
    with pytest.raises(RerankerValidationError, match="registry"):
        RerankerOrchestrator("not a registry")  # type: ignore[arg-type]


def test_orchestrator_init_validates_strategy_name() -> None:
    registry = RerankerRegistry()
    with pytest.raises(RerankerValidationError, match="strategy_name"):
        RerankerOrchestrator(registry, strategy_name="")


def test_orchestrator_init_validates_top_k() -> None:
    registry = RerankerRegistry()
    with pytest.raises(RerankerValidationError, match="default_top_k"):
        RerankerOrchestrator(registry, default_top_k=0)
    with pytest.raises(RerankerValidationError, match="max_top_k"):
        RerankerOrchestrator(registry, max_top_k=-1)
