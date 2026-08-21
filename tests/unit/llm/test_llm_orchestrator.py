"""Unit tests for LLMOrchestrator (mocked — no API call)."""
from __future__ import annotations

import pytest

from app.exceptions import LLMValidationError, LLMGenerationError
from app.models.llm import LLMResponse
from app.models.reranking import RankedChunk
from app.llm.base import LLMStrategy
from app.llm.orchestrator import LLMOrchestrator
from app.llm.registry import LLMRegistry


class _FakeLLM(LLMStrategy):
    name = "fake"

    def supports(self, query, candidates):
        return True

    def generate(self, query, candidates, *, max_tokens, temperature, **kwargs):
        return LLMResponse(
            answer=f"Réponse à: {query} ({len(candidates)} chunks)",
            query=query,
            strategy_name=self.name,
            model_name="fake-model",
            tokens_input=10,
            tokens_output=5,
            citations=tuple(),
        )


class _RejectingLLM(LLMStrategy):
    name = "rejector"

    def supports(self, query, candidates):
        return False

    def generate(self, query, candidates, *, max_tokens, temperature, **kwargs):
        return LLMResponse(answer="never", query=query, strategy_name="r", model_name="r")


def _make_candidates(n: int = 3) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=str(i),
            content=f"Contenu {i}.",
            source_name=f"doc{i}.pdf",
            source_type="document",
            retrieval_score=0.9 - i * 0.1,
            rerank_score=0.8 - i * 0.1,
            rank=i + 1,
            retrieval_strategy="qdrant",
            reranker_strategy="cross-encoder",
        )
        for i in range(n)
    ]


@pytest.fixture
def orchestrator() -> LLMOrchestrator:
    registry = LLMRegistry()
    registry.register(_FakeLLM)
    return LLMOrchestrator(registry, strategy_name="fake", default_max_tokens=512, default_temperature=0.5)


def test_generate_returns_response(orchestrator: LLMOrchestrator) -> None:
    result = orchestrator.generate("test query", _make_candidates(3))
    assert isinstance(result, LLMResponse)
    assert "test query" in result.answer
    assert result.strategy_name == "fake"


def test_generate_empty_candidates(orchestrator: LLMOrchestrator) -> None:
    result = orchestrator.generate("query", [])
    assert "Aucun contexte" in result.answer
    assert result.strategy_name == "fake"


def test_generate_empty_query(orchestrator: LLMOrchestrator) -> None:
    with pytest.raises(LLMValidationError, match="query"):
        orchestrator.generate("", _make_candidates())
    with pytest.raises(LLMValidationError, match="query"):
        orchestrator.generate("   ", _make_candidates())


@pytest.mark.parametrize("max_tokens", [0, -1, True, "3"])
def test_generate_invalid_max_tokens(orchestrator: LLMOrchestrator, max_tokens: object) -> None:
    with pytest.raises(LLMValidationError, match="max_tokens"):
        orchestrator.generate("q", _make_candidates(), max_tokens=max_tokens)  # type: ignore[arg-type]


@pytest.mark.parametrize("temperature", [True, "warm"])
def test_generate_invalid_temperature(orchestrator: LLMOrchestrator, temperature: object) -> None:
    with pytest.raises(LLMValidationError, match="temperature"):
        orchestrator.generate("q", _make_candidates(), temperature=temperature)  # type: ignore[arg-type]


def test_generate_truncates_by_max_candidates() -> None:
    registry = LLMRegistry()
    registry.register(_FakeLLM)
    orch = LLMOrchestrator(registry, strategy_name="fake", max_candidates=2)
    result = orch.generate("q", _make_candidates(5))
    assert "2 chunks" in result.answer


def test_generate_strategy_not_found() -> None:
    registry = LLMRegistry()
    orch = LLMOrchestrator(registry, strategy_name="nonexistent")
    with pytest.raises(Exception):
        orch.generate("q", _make_candidates())


def test_generate_strategy_rejects_inputs() -> None:
    registry = LLMRegistry()
    registry.register(_RejectingLLM)
    orch = LLMOrchestrator(registry, strategy_name="rejector")
    with pytest.raises(LLMValidationError, match="does not support"):
        orch.generate("q", _make_candidates())


def test_orchestrator_init_validates_registry() -> None:
    with pytest.raises(LLMValidationError, match="registry"):
        LLMOrchestrator("not a registry")  # type: ignore[arg-type]


def test_orchestrator_init_validates_strategy_name() -> None:
    registry = LLMRegistry()
    with pytest.raises(LLMValidationError, match="strategy_name"):
        LLMOrchestrator(registry, strategy_name="")


def test_orchestrator_init_validates_max_tokens() -> None:
    registry = LLMRegistry()
    with pytest.raises(LLMValidationError, match="default_max_tokens"):
        LLMOrchestrator(registry, default_max_tokens=0)


def test_orchestrator_init_validates_max_candidates() -> None:
    registry = LLMRegistry()
    with pytest.raises(LLMValidationError, match="max_candidates"):
        LLMOrchestrator(registry, max_candidates=-1)
