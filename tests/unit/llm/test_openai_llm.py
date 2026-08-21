"""Unit tests for OpenAILLM (mocked — no API call)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from app.exceptions import LLMValidationError, LLMGenerationError
from app.models.reranking import RankedChunk
from app.llm.strategies.openai_llm import OpenAILLM


def _make_candidates(n: int = 3) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=str(i),
            content=f"Contenu du chunk {i}.",
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


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    OpenAILLM.clear_client_cache()
    yield
    OpenAILLM.clear_client_cache()


@pytest.fixture()
def mock_openai():
    """Inject a fake openai module with a mock OpenAI client."""
    mock_client = MagicMock()
    mock_cls = MagicMock(return_value=mock_client)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = mock_cls  # type: ignore[attr-defined]

    prev = sys.modules.get("openai")
    sys.modules["openai"] = fake_openai
    yield mock_client
    if prev is None:
        sys.modules.pop("openai", None)
    else:
        sys.modules["openai"] = prev


def test_empty_candidates_returns_default_response() -> None:
    llm = OpenAILLM(api_key="test-key")
    result = llm.generate("query", [])
    assert "Aucun contexte" in result.answer
    assert result.strategy_name == "openai"


def test_query_validation() -> None:
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="query"):
        llm.generate("", _make_candidates())
    with pytest.raises(LLMValidationError, match="query"):
        llm.generate("   ", _make_candidates())


def test_max_tokens_validation() -> None:
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="max_tokens"):
        llm.generate("q", _make_candidates(), max_tokens=0)
    with pytest.raises(LLMValidationError, match="max_tokens"):
        llm.generate("q", _make_candidates(), max_tokens=-1)
    with pytest.raises(LLMValidationError, match="max_tokens"):
        llm.generate("q", _make_candidates(), max_tokens=True)  # type: ignore[arg-type]


def test_temperature_validation() -> None:
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="temperature"):
        llm.generate("q", _make_candidates(), temperature="hot")  # type: ignore[arg-type]
    with pytest.raises(LLMValidationError, match="temperature"):
        llm.generate("q", _make_candidates(), temperature=True)  # type: ignore[arg-type]


def test_candidate_type_validation() -> None:
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="candidates"):
        llm.generate("q", ["not a chunk"])  # type: ignore[list-item]


def test_generate_returns_response(mock_openai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Réponse test."), finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
    mock_openai.chat.completions.create.return_value = mock_resp

    llm = OpenAILLM(api_key="test-key")
    result = llm.generate("Pourquoi vibre le moteur ?", _make_candidates(2))

    assert result.answer == "Réponse test."
    assert result.strategy_name == "openai"
    assert result.model_name == "gpt-4o-mini"
    assert result.tokens_input == 50
    assert result.tokens_output == 20
    assert result.duration_ms >= 0
    assert len(result.citations) == 2
    assert result.metadata["candidates_count"] == 2
    assert result.metadata["finish_reason"] == "stop"


def test_generate_citations_match_candidates(mock_openai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_openai.chat.completions.create.return_value = mock_resp

    llm = OpenAILLM(api_key="test-key")
    candidates = _make_candidates(3)
    result = llm.generate("q", candidates)

    assert len(result.citations) == 3
    for i, citation in enumerate(result.citations):
        assert citation.chunk_id == candidates[i].chunk_id
        assert citation.source_name == candidates[i].source_name
        assert citation.rerank_score == candidates[i].rerank_score


def test_generate_empty_choices_raises(mock_openai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.choices = []
    mock_openai.chat.completions.create.return_value = mock_resp

    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMGenerationError, match="empty response"):
        llm.generate("q", _make_candidates(1))


def test_generate_none_content_returns_fallback(mock_openai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=None), finish_reason="stop")]
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_openai.chat.completions.create.return_value = mock_resp

    llm = OpenAILLM(api_key="test-key")
    result = llm.generate("q", _make_candidates(1))
    assert "modèle" in result.answer.lower() or len(result.answer) > 0


def test_generate_wraps_api_error(mock_openai: MagicMock) -> None:
    mock_openai.chat.completions.create.side_effect = RuntimeError("boom")
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMGenerationError):
        llm.generate("q", _make_candidates(1))


def test_generate_wraps_rate_limit_error(mock_openai: MagicMock) -> None:
    from app.exceptions import LLMRateLimitError
    mock_openai.chat.completions.create.side_effect = Exception("Rate limit exceeded 429")
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMRateLimitError):
        llm.generate("q", _make_candidates(1))


def test_generate_wraps_connection_error(mock_openai: MagicMock) -> None:
    from app.exceptions import LLMConnectionError
    mock_openai.chat.completions.create.side_effect = Exception("Connection timeout")
    llm = OpenAILLM(api_key="test-key")
    with pytest.raises(LLMConnectionError):
        llm.generate("q", _make_candidates(1))


def test_init_validates_api_key() -> None:
    with pytest.raises(LLMValidationError, match="api_key"):
        OpenAILLM(api_key="")


def test_init_validates_model_name() -> None:
    with pytest.raises(LLMValidationError, match="model_name"):
        OpenAILLM(api_key="test", model_name="  ")


def test_properties() -> None:
    llm = OpenAILLM(api_key="test-key", model_name="gpt-4o")
    assert llm.model_name == "gpt-4o"
