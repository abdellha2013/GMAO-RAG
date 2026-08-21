"""Unit tests for reranking data models."""
from __future__ import annotations

import math

import pytest

from app.models.reranking import RankedChunk


def _make_ranked_chunk(**overrides: object) -> RankedChunk:
    defaults: dict[str, object] = dict(
        chunk_id="1",
        content="Le moteur vibre.",
        source_name="manual.pdf",
        source_type="document",
        retrieval_score=0.8,
        rerank_score=4.5,
        rank=1,
    )
    defaults.update(overrides)
    return RankedChunk(**defaults)  # type: ignore[arg-type]


def test_ranked_chunk_construction() -> None:
    chunk = _make_ranked_chunk()
    assert chunk.chunk_id == "1"
    assert chunk.retrieval_score == 0.8
    assert chunk.rerank_score == 4.5
    assert chunk.rank == 1


def test_ranked_chunk_preserves_optional_fields() -> None:
    chunk = _make_ranked_chunk(
        id_document=42,
        id_panne=7,
        id_equipement=3,
        retrieval_strategy="qdrant",
        reranker_strategy="cross-encoder",
        metadata={"key": "value"},
    )
    assert chunk.id_document == 42
    assert chunk.id_panne == 7
    assert chunk.id_equipement == 3
    assert chunk.metadata == {"key": "value"}


def test_ranked_chunk_is_frozen() -> None:
    chunk = _make_ranked_chunk()
    with pytest.raises(AttributeError):
        chunk.rank = 2  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["chunk_id", "content", "source_name", "source_type"])
def test_ranked_chunk_rejects_empty_strings(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _make_ranked_chunk(**{field_name: ""})


@pytest.mark.parametrize("value", [math.nan, math.inf, "0.5", True])
def test_ranked_chunk_rejects_invalid_retrieval_score(value: object) -> None:
    with pytest.raises(ValueError, match="retrieval_score"):
        _make_ranked_chunk(retrieval_score=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, "0.5", True])
def test_ranked_chunk_rejects_invalid_rerank_score(value: object) -> None:
    with pytest.raises(ValueError, match="rerank_score"):
        _make_ranked_chunk(rerank_score=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_ranked_chunk_rejects_invalid_rank(value: object) -> None:
    with pytest.raises(ValueError, match="rank"):
        _make_ranked_chunk(rank=value)  # type: ignore[arg-type]
