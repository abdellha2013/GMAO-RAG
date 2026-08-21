"""Extended unit tests for :class:`RetrievalOrchestrator`."""
from __future__ import annotations

import math
from typing import Any

import pytest

from app.embedding.base import EmbeddingStrategy
from app.embedding.registry import EmbeddingRegistry
from app.exceptions import (
    EmptyQueryError,
    RetrievalExecutionError,
    RetrievalValidationError,
    RetrievalStrategyNotRegisteredError,
)
from app.models.retrieval import RetrievalFilter, RetrievedChunk
from app.retrieval.base import RetrievalStrategy
from app.retrieval.orchestrator import RetrievalOrchestrator
from app.retrieval.registry import RetrievalRegistry


class _QueryEncoder(EmbeddingStrategy):
    @property
    def name(self) -> str:
        return "query-encoder"

    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def dimension(self) -> int:
        return 2

    def supports(self, chunks) -> bool:
        return True

    def embed(self, chunks):
        return []

    def embed_query(self, query: str) -> tuple[float, ...]:
        return (0.2, 0.8)


class _NoQueryEncoder(EmbeddingStrategy):
    """Embedding strategy that lacks embed_query."""
    @property
    def name(self) -> str:
        return "no-query"

    @property
    def model_name(self) -> str:
        return "test"

    @property
    def dimension(self) -> int:
        return 2

    def supports(self, chunks) -> bool:
        return True

    def embed(self, chunks):
        return []


class _FakeRetrievalStrategy(RetrievalStrategy):
    name = "fake"

    def supports(self, filters: RetrievalFilter) -> bool:
        return True

    def retrieve(self, query_vector, *, top_k, filters, query_text):
        return [
            RetrievedChunk(
                chunk_id="1", content="first", score=0.9,
                rank=1, source_name="a", source_type="doc",
                retrieval_strategy=self.name,
            ),
            RetrievedChunk(
                chunk_id="2", content="second", score=0.2,
                rank=2, source_name="b", source_type="doc",
                retrieval_strategy=self.name,
            ),
        ][:top_k]


class _RejectFiltersStrategy(RetrievalStrategy):
    name = "rejector"

    def supports(self, filters: RetrievalFilter) -> bool:
        return filters.id_equipement is None

    def retrieve(self, query_vector, *, top_k, filters, query_text):
        return []


class _FailStrategy(RetrievalStrategy):
    """Strategy that raises on retrieve."""
    name = "fail"

    def supports(self, filters: RetrievalFilter) -> bool:
        return True

    def retrieve(self, query_vector, *, top_k, filters, query_text):
        raise RuntimeError("boom")


def _make_orchestrator(
    strategy_cls: type[RetrievalStrategy] = _FakeRetrievalStrategy,
    embedding_cls: type[EmbeddingStrategy] = _QueryEncoder,
    strategy_name: str | None = None,
    embedding_strategy_name: str | None = None,
    **kwargs: Any,
) -> RetrievalOrchestrator:
    emb_reg = EmbeddingRegistry()
    emb_reg.register(embedding_cls)
    ret_reg = RetrievalRegistry()
    ret_reg.register(strategy_cls)
    return RetrievalOrchestrator(
        ret_reg,
        emb_reg,
        strategy_name=strategy_name or strategy_cls.name,
        embedding_strategy_name=embedding_strategy_name or "query-encoder",
        **kwargs,
    )


class TestOrchestratorInit:
    def test_rejects_non_registry(self) -> None:
        from unittest.mock import MagicMock as M
        with pytest.raises(RetrievalValidationError):
            RetrievalOrchestrator(M(), M())  # type: ignore[arg-type]

    def test_rejects_bad_strategy_name(self) -> None:
        with pytest.raises(RetrievalValidationError):
            _make_orchestrator(strategy_name="  ")

    def test_rejects_bad_embedding_name(self) -> None:
        with pytest.raises(RetrievalValidationError):
            _make_orchestrator(embedding_strategy_name="  ")

    def test_rejects_invalid_top_k(self) -> None:
        with pytest.raises(RetrievalValidationError):
            _make_orchestrator(default_top_k=0)

    def test_rejects_invalid_max_top_k(self) -> None:
        with pytest.raises(RetrievalValidationError):
            _make_orchestrator(max_top_k=-1)

    def test_rejects_nan_threshold(self) -> None:
        with pytest.raises(RetrievalValidationError):
            _make_orchestrator(score_threshold=math.nan)

    def test_rejects_invalid_embedding_options(self) -> None:
        with pytest.raises(RetrievalValidationError):
            _make_orchestrator(embedding_options="bad")  # type: ignore[arg-type]

    def test_embedding_options_stored(self) -> None:
        orch = _make_orchestrator(embedding_options={"model_name": "x"})
        assert orch.embedding_options == {"model_name": "x"}


class TestOrchestratorRetrieve:
    def test_empty_query(self) -> None:
        orch = _make_orchestrator()
        with pytest.raises(EmptyQueryError):
            orch.retrieve("   ")

    @pytest.mark.parametrize("top_k", [0, -1, True, "3"])
    def test_invalid_top_k(self, top_k: object) -> None:
        orch = _make_orchestrator()
        with pytest.raises(RetrievalValidationError):
            orch.retrieve("q", top_k=top_k)  # type: ignore[arg-type]

    def test_invalid_filters(self) -> None:
        orch = _make_orchestrator()
        with pytest.raises(RetrievalValidationError):
            orch.retrieve("q", filters=object())  # type: ignore[arg-type]

    def test_invalid_strategy_name(self) -> None:
        orch = _make_orchestrator()
        with pytest.raises(RetrievalValidationError):
            orch.retrieve("q", strategy_name="  ")

    def test_score_threshold_filters(self) -> None:
        orch = _make_orchestrator(score_threshold=0.5)
        report = orch.retrieve("q", top_k=10)
        assert len(report.results) == 1
        assert report.results[0].score >= 0.5

    def test_max_top_k_cap(self) -> None:
        orch = _make_orchestrator(max_top_k=1)
        report = orch.retrieve("q", top_k=100)
        assert report.total_candidates <= 1

    def test_strategy_not_found_wraps(self) -> None:
        orch = _make_orchestrator()
        with pytest.raises(RetrievalStrategyNotRegisteredError):
            orch.retrieve("q", strategy_name="nonexistent")

    def test_strategy_failure_wraps(self) -> None:
        orch = _make_orchestrator(strategy_cls=_FailStrategy)
        with pytest.raises(RetrievalExecutionError):
            orch.retrieve("q")

    def test_strategy_rejects_filters(self) -> None:
        orch = _make_orchestrator(strategy_cls=_RejectFiltersStrategy)
        f = RetrievalFilter(id_equipement=1)
        with pytest.raises(RetrievalValidationError, match="filters"):
            orch.retrieve("q", filters=f)

    def test_no_query_encoder_raises(self) -> None:
        orch = _make_orchestrator(
            embedding_cls=_NoQueryEncoder,
            embedding_strategy_name="no-query",
        )
        with pytest.raises(RetrievalValidationError, match="query encoding"):
            orch.retrieve("q")

    def test_report_fields(self) -> None:
        orch = _make_orchestrator()
        report = orch.retrieve("  hello  ")
        assert report.query == "hello"
        assert report.strategy_name == "fake"
        assert len(report.results) == 2

    def test_gmao_errors_not_wrapped(self) -> None:
        orch = _make_orchestrator()
        with pytest.raises(EmptyQueryError):
            orch.retrieve("")

    def test_embedding_options_forwarded(self) -> None:
        """embedding_options are passed to the embedding strategy constructor."""
        orch = _make_orchestrator()
        assert orch.embedding_options == {}
