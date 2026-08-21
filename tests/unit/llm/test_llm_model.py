"""Unit tests for LLM data models."""
from __future__ import annotations

import math

import pytest

from app.models.llm import Citation, LLMResponse


def _make_citation(**overrides: object) -> Citation:
    defaults: dict[str, object] = dict(
        chunk_id="1",
        source_name="manual.pdf",
        source_type="document",
        rerank_score=0.9,
    )
    defaults.update(overrides)
    return Citation(**defaults)  # type: ignore[arg-type]


def _make_response(**overrides: object) -> LLMResponse:
    defaults: dict[str, object] = dict(
        answer="Le moteur vibre à cause du rotor.",
        query="Pourquoi vibre le moteur ?",
        strategy_name="openai",
        model_name="gpt-4o-mini",
    )
    defaults.update(overrides)
    return LLMResponse(**defaults)  # type: ignore[arg-type]


def test_citation_construction() -> None:
    c = _make_citation()
    assert c.chunk_id == "1"
    assert c.source_name == "manual.pdf"
    assert c.rerank_score == 0.9


def test_citation_is_frozen() -> None:
    c = _make_citation()
    with pytest.raises(AttributeError):
        c.chunk_id = "2"  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["chunk_id", "source_name", "source_type"])
def test_citation_rejects_empty_strings(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _make_citation(**{field_name: ""})


@pytest.mark.parametrize("value", [math.nan, math.inf, "0.5", True])
def test_citation_rejects_invalid_rerank_score(value: object) -> None:
    with pytest.raises(ValueError, match="rerank_score"):
        _make_citation(rerank_score=value)  # type: ignore[arg-type]


def test_response_construction() -> None:
    r = _make_response()
    assert r.answer == "Le moteur vibre à cause du rotor."
    assert r.strategy_name == "openai"
    assert r.tokens_input == 0
    assert r.citations == ()


def test_response_with_citations() -> None:
    c1 = _make_citation(chunk_id="10")
    c2 = _make_citation(chunk_id="20", rerank_score=0.7)
    r = _make_response(citations=(c1, c2))
    assert len(r.citations) == 2
    assert r.citations[0].chunk_id == "10"


def test_response_is_frozen() -> None:
    r = _make_response()
    with pytest.raises(AttributeError):
        r.answer = "test"  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["answer", "query", "strategy_name", "model_name"])
def test_response_rejects_empty_strings(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _make_response(**{field_name: ""})


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_response_rejects_invalid_tokens_input(value: object) -> None:
    with pytest.raises(ValueError, match="tokens_input"):
        _make_response(tokens_input=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_response_rejects_invalid_tokens_output(value: object) -> None:
    with pytest.raises(ValueError, match="tokens_output"):
        _make_response(tokens_output=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -1, "0.5", True])
def test_response_rejects_invalid_duration_ms(value: object) -> None:
    with pytest.raises(ValueError, match="duration_ms"):
        _make_response(duration_ms=value)  # type: ignore[arg-type]


def test_response_rejects_invalid_citations_type() -> None:
    with pytest.raises(ValueError, match="citations"):
        _make_response(citations=[_make_citation()])  # type: ignore[arg-type]


def test_response_rejects_invalid_citations_element() -> None:
    with pytest.raises(ValueError, match="citations"):
        _make_response(citations=("not a citation",))  # type: ignore[arg-type]
