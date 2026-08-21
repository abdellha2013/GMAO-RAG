"""Extended unit tests for retrieval data models and registry."""
from __future__ import annotations

import math

import pytest

from app.exceptions import (
    InvalidRetrievalStrategyError,
    RetrievalStrategyNotRegisteredError,
    RetrievalValidationError,
)
from app.models.retrieval import RetrievalFilter, RetrievedChunk, RetrievalReport
from app.retrieval.base import RetrievalStrategy
from app.retrieval.registry import RetrievalRegistry


class _TestStrategy(RetrievalStrategy):
    name = "test"

    def supports(self, filters: RetrievalFilter) -> bool:
        return True

    def retrieve(self, query_vector, *, top_k, filters, query_text):
        return []


class _BadStrategy:
    name = "bad"  # type: ignore[assignment]


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
class TestRegistry:
    def test_register_and_get(self) -> None:
        registry = RetrievalRegistry()
        registry.register(_TestStrategy)
        assert registry.get("test") is _TestStrategy
        assert registry.has("test")

    def test_has_returns_false_for_non_string(self) -> None:
        registry = RetrievalRegistry()
        assert registry.has(42) is False  # type: ignore[arg-type]
        assert registry.has(None) is False  # type: ignore[arg-type]
        assert registry.has("") is False

    def test_unregister_nonexistent_raises(self) -> None:
        registry = RetrievalRegistry()
        with pytest.raises(RetrievalStrategyNotRegisteredError):
            registry.unregister("nope")

    def test_get_nonexistent_raises(self) -> None:
        registry = RetrievalRegistry()
        with pytest.raises(RetrievalStrategyNotRegisteredError):
            registry.get("nope")

    def test_duplicate_register_raises(self) -> None:
        registry = RetrievalRegistry()
        registry.register(_TestStrategy)
        with pytest.raises(RetrievalValidationError):
            registry.register(_TestStrategy)

    def test_rejects_non_strategy_class(self) -> None:
        registry = RetrievalRegistry()
        with pytest.raises(InvalidRetrievalStrategyError):
            registry.register(_BadStrategy)  # type: ignore[arg-type]

    def test_name_normalization(self) -> None:
        registry = RetrievalRegistry()
        registry.register(_TestStrategy)
        assert registry.get("  TEST  ") is _TestStrategy

    def test_rejects_empty_name(self) -> None:
        registry = RetrievalRegistry()
        with pytest.raises(RetrievalValidationError):
            registry.get("")

    def test_clear(self) -> None:
        registry = RetrievalRegistry()
        registry.register(_TestStrategy)
        registry.clear()
        assert registry.supported_strategies() == ()

    def test_supported_strategies_sorted(self) -> None:
        class _Alpha(RetrievalStrategy):
            name = "alpha"
            def supports(self, filters): return True
            def retrieve(self, query_vector, *, top_k, filters, query_text): return []

        class _Omega(RetrievalStrategy):
            name = "omega"
            def supports(self, filters): return True
            def retrieve(self, query_vector, *, top_k, filters, query_text): return []

        registry = RetrievalRegistry()
        registry.register(_Omega)
        registry.register(_Alpha)
        assert registry.supported_strategies() == ("alpha", "omega")


# ------------------------------------------------------------------
# base.py — __init_subclass__
# ------------------------------------------------------------------
class TestInitSubclass:
    def test_missing_name_raises(self) -> None:
        with pytest.raises(InvalidRetrievalStrategyError):
            class _(RetrievalStrategy):
                def supports(self, filters): return True
                def retrieve(self, query_vector, *, top_k, filters, query_text): return []

    def test_empty_name_raises(self) -> None:
        with pytest.raises(InvalidRetrievalStrategyError):
            class _(RetrievalStrategy):
                name = "   "
                def supports(self, filters): return True
                def retrieve(self, query_vector, *, top_k, filters, query_text): return []

    def test_valid_name_passes(self) -> None:
        class Good(RetrievalStrategy):
            name = "good"
            def supports(self, filters): return True
            def retrieve(self, query_vector, *, top_k, filters, query_text): return []
        assert Good.name == "good"


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
class TestRetrievalFilter:
    def test_source_type_normalized(self) -> None:
        f = RetrievalFilter(source_type="  PDF  ")
        assert f.source_type == "pdf"

    def test_empty_source_type_raises(self) -> None:
        with pytest.raises(ValueError, match="source_type"):
            RetrievalFilter(source_type="   ")

    @pytest.mark.parametrize("value", [0, -1, True, "1"])
    def test_invalid_id_document(self, value: object) -> None:
        with pytest.raises(ValueError):
            RetrievalFilter(id_document=value)  # type: ignore[arg-type]

    def test_min_score_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_score"):
            RetrievalFilter(min_score=math.nan)

    def test_min_score_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_score"):
            RetrievalFilter(min_score=math.inf)


class TestRetrievedChunk:
    def test_construction(self) -> None:
        c = RetrievedChunk(
            chunk_id="1",
            content="text",
            score=0.5,
            rank=1,
            source_name="src",
            source_type="doc",
            retrieval_strategy="qdrant",
        )
        assert c.chunk_id == "1"

    @pytest.mark.parametrize("field", ["chunk_id", "content", "source_name", "source_type"])
    def test_empty_string_rejected(self, field: str) -> None:
        base = {
            "chunk_id": "x", "content": "x", "score": 0.5,
            "rank": 1, "source_name": "x", "source_type": "x",
            "retrieval_strategy": "x",
        }
        base[field] = ""
        with pytest.raises(ValueError, match=field):
            RetrievedChunk(**base)


class TestRetrievalReport:
    def test_empty_report(self) -> None:
        r = RetrievalReport(query="q", strategy_name="s")
        assert r.is_empty

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="query"):
            RetrievalReport(query="", strategy_name="s")
