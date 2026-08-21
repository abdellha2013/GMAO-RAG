"""Unit tests for CrossEncoderReranker (mocked — no model download)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from app.exceptions import RerankerValidationError, RerankingError
from app.models.retrieval import RetrievedChunk
from app.reranker.strategies.cross_encoder import CrossEncoderReranker


def _make_candidates(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=str(i),
            content=f"Contenu du chunk {i}.",
            score=0.9 - i * 0.1,
            rank=i + 1,
            source_name=f"doc{i}.pdf",
            source_type="document",
            retrieval_strategy="qdrant",
        )
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    CrossEncoderReranker.clear_model_cache()
    yield
    CrossEncoderReranker.clear_model_cache()


@pytest.fixture()
def mock_cross_encoder():
    """Inject a fake sentence_transformers module with a mock CrossEncoder."""
    mock_model = MagicMock()
    mock_cls = MagicMock(return_value=mock_model)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.CrossEncoder = mock_cls  # type: ignore[attr-defined]

    prev = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = fake_st
    yield mock_cls
    if prev is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = prev


def test_empty_candidates_returns_empty() -> None:
    reranker = CrossEncoderReranker(model_name="test-model", device="cpu")
    results = reranker.rerank("query", [], top_k=5)
    assert results == []


def test_query_validation() -> None:
    reranker = CrossEncoderReranker(model_name="test-model", device="cpu")
    with pytest.raises(RerankerValidationError, match="query"):
        reranker.rerank("", _make_candidates(), top_k=5)
    with pytest.raises(RerankerValidationError, match="query"):
        reranker.rerank("   ", _make_candidates(), top_k=5)


def test_top_k_validation() -> None:
    reranker = CrossEncoderReranker(model_name="test-model", device="cpu")
    with pytest.raises(RerankerValidationError, match="top_k"):
        reranker.rerank("query", _make_candidates(), top_k=0)
    with pytest.raises(RerankerValidationError, match="top_k"):
        reranker.rerank("query", _make_candidates(), top_k=-1)
    with pytest.raises(RerankerValidationError, match="top_k"):
        reranker.rerank("query", _make_candidates(), top_k=True)  # type: ignore[arg-type]


def test_candidate_type_validation() -> None:
    reranker = CrossEncoderReranker(model_name="test-model", device="cpu")
    with pytest.raises(RerankerValidationError, match="candidates"):
        reranker.rerank("query", ["not a chunk"], top_k=5)  # type: ignore[list-item]


def test_rerank_returns_ranked_chunks(mock_cross_encoder: MagicMock) -> None:
    mock_model = mock_cross_encoder.return_value
    mock_model.predict.return_value = [0.9, 0.1, 0.5]

    reranker = CrossEncoderReranker(model_name="mock-model")
    candidates = _make_candidates(3)
    results = reranker.rerank("test query", candidates, top_k=3)

    assert len(results) == 3
    assert results[0].rerank_score == 0.9
    assert results[0].rank == 1
    assert results[0].chunk_id == "0"
    assert results[1].rerank_score == 0.5
    assert results[2].rerank_score == 0.1


def test_rerank_respects_top_k(mock_cross_encoder: MagicMock) -> None:
    mock_model = mock_cross_encoder.return_value
    mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.8, 0.3]

    reranker = CrossEncoderReranker(model_name="mock-model")
    candidates = _make_candidates(5)
    results = reranker.rerank("query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].rerank_score == 0.9
    assert results[1].rerank_score == 0.8


def test_rerank_preserves_retrieval_score(mock_cross_encoder: MagicMock) -> None:
    mock_model = mock_cross_encoder.return_value
    mock_model.predict.return_value = [4.2]

    reranker = CrossEncoderReranker(model_name="mock-model")
    candidates = _make_candidates(1)
    results = reranker.rerank("query", candidates, top_k=1)

    assert results[0].retrieval_score == 0.9
    assert results[0].rerank_score == 4.2


def test_rerank_preserves_metadata(mock_cross_encoder: MagicMock) -> None:
    mock_model = mock_cross_encoder.return_value
    mock_model.predict.return_value = [0.7]

    reranker = CrossEncoderReranker(model_name="mock-model")
    candidates = _make_candidates(1)
    results = reranker.rerank("query", candidates, top_k=1)

    assert results[0].metadata["retrieval_rank"] == 1
    assert results[0].reranker_strategy == "cross-encoder"
    assert results[0].retrieval_strategy == "qdrant"


def test_rerank_handles_fewer_candidates_than_top_k(mock_cross_encoder: MagicMock) -> None:
    mock_model = mock_cross_encoder.return_value
    mock_model.predict.return_value = [0.5, 0.3]

    reranker = CrossEncoderReranker(model_name="mock-model")
    candidates = _make_candidates(2)
    results = reranker.rerank("query", candidates, top_k=10)

    assert len(results) == 2


def test_rerank_wraps_prediction_error(mock_cross_encoder: MagicMock) -> None:
    mock_model = mock_cross_encoder.return_value
    mock_model.predict.side_effect = RuntimeError("GPU OOM")

    reranker = CrossEncoderReranker(model_name="mock-model")
    with pytest.raises(RerankingError):
        reranker.rerank("query", _make_candidates(1), top_k=1)


def test_init_validates_model_name() -> None:
    with pytest.raises(RerankerValidationError, match="model_name"):
        CrossEncoderReranker(model_name="")
    with pytest.raises(RerankerValidationError, match="model_name"):
        CrossEncoderReranker(model_name="  ")


def test_init_validates_batch_size() -> None:
    with pytest.raises(RerankerValidationError, match="batch_size"):
        CrossEncoderReranker(batch_size=0)
    with pytest.raises(RerankerValidationError, match="batch_size"):
        CrossEncoderReranker(batch_size=-1)
    with pytest.raises(RerankerValidationError, match="batch_size"):
        CrossEncoderReranker(batch_size=True)  # type: ignore[arg-type]


def test_init_validates_device() -> None:
    with pytest.raises(RerankerValidationError, match="device"):
        CrossEncoderReranker(device="tpu")
    with pytest.raises(RerankerValidationError, match="device"):
        CrossEncoderReranker(device="")


def test_properties() -> None:
    r = CrossEncoderReranker(model_name="test", batch_size=8, device="cpu")
    assert r.model_name == "test"
    assert r.batch_size == 8
    assert r.device == "cpu"
