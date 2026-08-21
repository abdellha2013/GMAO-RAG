"""Mocked unit tests for QdrantVectorRetrieval and HybridRetrieval."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.exceptions import (
    IncompatibleEmbeddingModelError,
    RetrievalConnectionError,
    RetrievalExecutionError,
    RetrievalValidationError,
)
from app.models.retrieval import RetrievalFilter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _make_qdrant_strategy(**overrides):
    """Create a QdrantVectorRetrieval with mocked client and dsn."""
    defaults = {
        "collection_name": "test_col",
        "host": "localhost",
        "port": 6333,
        "dsn": None,
    }
    defaults.update(overrides)
    with patch(
        "app.retrieval.strategies.qdrant_retrieval._get_qdrant_client"
    ) as mock_client_factory:
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        from app.retrieval.strategies.qdrant_retrieval import (
            QdrantVectorRetrieval,
        )
        strat = QdrantVectorRetrieval(**defaults)
        strat._mock_client = mock_client
        return strat


def _make_hybrid_strategy(**overrides):
    """Create a HybridRetrieval with mocked vector backend."""
    defaults = {
        "collection_name": "test_col",
        "host": "localhost",
        "port": 6333,
        "dsn": None,
    }
    defaults.update(overrides)
    with patch(
        "app.retrieval.strategies.qdrant_retrieval._get_qdrant_client"
    ):
        from app.retrieval.strategies.hybrid_retrieval import HybridRetrieval
        strat = HybridRetrieval(**defaults)
        return strat


# ------------------------------------------------------------------
# QdrantVectorRetrieval
# ------------------------------------------------------------------
class TestQdrantFilter:
    def test_filter_none_when_empty(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter()
        assert strat._filter(f) is None

    def test_filter_id_equipement(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter(id_equipement=42)
        qf = strat._filter(f)
        assert qf is not None

    def test_filter_document_type(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter(id_document=1)
        qf = strat._filter(f)
        assert qf is not None

    def test_filter_panne_type(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter(id_panne=1)
        qf = strat._filter(f)
        assert qf is not None

    def test_supports_always_true(self) -> None:
        strat = _make_qdrant_strategy()
        assert strat.supports(RetrievalFilter()) is True
        assert strat.supports(RetrievalFilter(id_document=1)) is True

    def test_filter_source_type_document(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter(source_type="document")
        qf = strat._filter(f)
        assert qf is not None

    def test_filter_source_type_panne(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter(source_type="panne")
        qf = strat._filter(f)
        assert qf is not None

    def test_filter_id_equipement_only(self) -> None:
        strat = _make_qdrant_strategy()
        f = RetrievalFilter(id_equipement=5)
        qf = strat._filter(f)
        assert qf is not None


class TestQdrantHydrate:
    def test_empty_ids_returns_empty(self) -> None:
        strat = _make_qdrant_strategy()
        assert strat._hydrate([], RetrievalFilter()) == []

    def test_missing_dsn_raises_validation(self) -> None:
        strat = _make_qdrant_strategy(dsn=None)
        # Override dsn to ensure it's truly empty even if env has values
        strat.dsn = ""
        with pytest.raises(RetrievalValidationError, match="MYSQL_DSN"):
            strat._hydrate([1], RetrievalFilter())


class TestQdrantDimension:
    def test_incompatible_dimension_raises(self) -> None:
        strat = _make_qdrant_strategy()
        mock_info = MagicMock()
        mock_vp = MagicMock()
        mock_vp.size = 128
        mock_info.config.params.vectors = mock_vp
        strat._mock_client.get_collection.return_value = mock_info

        with pytest.raises(IncompatibleEmbeddingModelError):
            strat._check_dimension([0.1] * 64)  # wrong size

    def test_compatible_dimension_passes(self) -> None:
        strat = _make_qdrant_strategy()
        mock_info = MagicMock()
        mock_vp = MagicMock()
        mock_vp.size = 128
        mock_info.config.params.vectors = mock_vp
        strat._mock_client.get_collection.return_value = mock_info

        strat._check_dimension([0.1] * 128)  # correct size

    def test_named_vectors_checked(self) -> None:
        strat = _make_qdrant_strategy()
        mock_info = MagicMock()
        mock_vp = MagicMock()
        mock_vp.size = 64
        mock_info.config.params.vectors = {"default": mock_vp}
        strat._mock_client.get_collection.return_value = mock_info

        with pytest.raises(IncompatibleEmbeddingModelError):
            strat._check_dimension([0.1] * 128)

    def test_empty_named_vectors_raises(self) -> None:
        strat = _make_qdrant_strategy()
        mock_info = MagicMock()
        mock_info.config.params.vectors = {}
        strat._mock_client.get_collection.return_value = mock_info

        with pytest.raises(RetrievalExecutionError, match="no vector"):
            strat._check_dimension([0.1] * 128)


class TestQdrantRetrieve:
    def test_empty_points_returns_empty(self) -> None:
        strat = _make_qdrant_strategy(dsn=None)
        mock_info = MagicMock()
        mock_vp = MagicMock()
        mock_vp.size = 2
        mock_info.config.params.vectors = mock_vp
        strat._mock_client.get_collection.return_value = mock_info
        strat._mock_client.query_points.return_value = MagicMock(points=[])
        result = strat.retrieve(
            [0.1] * 2, top_k=5, filters=RetrievalFilter(), query_text="test"
        )
        assert result == []


# ------------------------------------------------------------------
# HybridRetrieval
# ------------------------------------------------------------------
class TestHybridRetrieval:
    def test_missing_dsn_raises_validation(self) -> None:
        strat = _make_hybrid_strategy(dsn=None)
        strat.vector.dsn = ""
        with pytest.raises(RetrievalValidationError, match="MYSQL_DSN"):
            strat._lexical("test", top_k=5, filters=RetrievalFilter())

    def test_supports_delegates_to_vector(self) -> None:
        strat = _make_hybrid_strategy()
        assert strat.supports(RetrievalFilter()) is True

    def test_rrf_k_stored(self) -> None:
        strat = _make_hybrid_strategy(rrf_k=30)
        assert strat.rrf_k == 30


# ------------------------------------------------------------------
# chunk_from_row
# ------------------------------------------------------------------
class TestChunkFromRow:
    def test_builds_chunk(self) -> None:
        from app.retrieval.strategies.qdrant_retrieval import chunk_from_row

        row = {
            "id_chunk": 42,
            "contenu": "test content",
            "source_name": "doc.pdf",
            "source_type": "document",
            "id_document": 1,
            "id_panne": None,
            "id_equipement": 7,
        }
        chunk = chunk_from_row(
            row, score=0.8, rank=3, strategy_name="qdrant"
        )
        assert chunk.chunk_id == "42"
        assert chunk.content == "test content"
        assert chunk.score == 0.8
        assert chunk.rank == 3
        assert chunk.id_document == 1
        assert chunk.metadata["id_chunk"] == 42
