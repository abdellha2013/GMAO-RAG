"""Unit tests for RerankerRegistry."""
from __future__ import annotations

import pytest

from app.exceptions import (
    InvalidRerankerStrategyError,
    RerankerStrategyNotRegisteredError,
    RerankerValidationError,
)
from app.reranker.base import RerankerStrategy
from app.reranker.registry import RerankerRegistry


class _FakeReranker(RerankerStrategy):
    name = "fake"

    def supports(self, query, candidates):
        return True

    def rerank(self, query, candidates, *, top_k, **kwargs):
        return []


def test_registry_stores_classes_without_instantiating() -> None:
    registry = RerankerRegistry()
    registry.register(_FakeReranker)

    assert registry.get("FAKE") is _FakeReranker
    assert registry.has("fake")
    assert registry.supported_strategies() == ("fake",)

    registry.unregister("fake")
    with pytest.raises(RerankerStrategyNotRegisteredError):
        registry.get("fake")


def test_registry_rejects_invalid_strategy() -> None:
    registry = RerankerRegistry()
    with pytest.raises(InvalidRerankerStrategyError):
        registry.register(object)  # type: ignore[arg-type]


def test_registry_rejects_duplicate_strategy() -> None:
    registry = RerankerRegistry()
    registry.register(_FakeReranker)
    with pytest.raises(RerankerValidationError):
        registry.register(_FakeReranker)


def test_registry_has_returns_false_for_unknown() -> None:
    registry = RerankerRegistry()
    assert registry.has("nonexistent") is False
    assert registry.has("") is False
    assert registry.has(42) is False  # type: ignore[arg-type]


def test_registry_clear() -> None:
    registry = RerankerRegistry()
    registry.register(_FakeReranker)
    assert registry.has("fake")
    registry.clear()
    assert not registry.supported_strategies()


def test_registry_unregister_nonexistent_raises() -> None:
    registry = RerankerRegistry()
    with pytest.raises(RerankerStrategyNotRegisteredError):
        registry.unregister("ghost")
