"""Unit tests for LLMRegistry."""
from __future__ import annotations

import pytest

from app.exceptions import (
    InvalidLLMStrategyError,
    LLMStrategyNotRegisteredError,
    LLMValidationError,
)
from app.llm.base import LLMStrategy
from app.llm.registry import LLMRegistry


class _FakeLLM(LLMStrategy):
    name = "fake"

    def supports(self, query, candidates):
        return True

    def generate(self, query, candidates, *, max_tokens, temperature, **kwargs):
        from app.models.llm import LLMResponse
        return LLMResponse(answer="fake", query=query, strategy_name="fake", model_name="fake")


def test_registry_stores_classes_without_instantiating() -> None:
    registry = LLMRegistry()
    registry.register(_FakeLLM)

    assert registry.get("FAKE") is _FakeLLM
    assert registry.has("fake")
    assert registry.supported_strategies() == ("fake",)

    registry.unregister("fake")
    with pytest.raises(LLMStrategyNotRegisteredError):
        registry.get("fake")


def test_registry_rejects_invalid_strategy() -> None:
    registry = LLMRegistry()
    with pytest.raises(InvalidLLMStrategyError):
        registry.register(object)  # type: ignore[arg-type]


def test_registry_rejects_duplicate_strategy() -> None:
    registry = LLMRegistry()
    registry.register(_FakeLLM)
    with pytest.raises(LLMValidationError):
        registry.register(_FakeLLM)


def test_registry_has_returns_false_for_unknown() -> None:
    registry = LLMRegistry()
    assert registry.has("nonexistent") is False
    assert registry.has("") is False
    assert registry.has(42) is False  # type: ignore[arg-type]


def test_registry_clear() -> None:
    registry = LLMRegistry()
    registry.register(_FakeLLM)
    assert registry.has("fake")
    registry.clear()
    assert not registry.supported_strategies()


def test_registry_unregister_nonexistent_raises() -> None:
    registry = LLMRegistry()
    with pytest.raises(LLMStrategyNotRegisteredError):
        registry.unregister("ghost")
