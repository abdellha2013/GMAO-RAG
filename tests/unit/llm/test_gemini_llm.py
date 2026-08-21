"""Unit tests for GeminiLLM (mocked — no API call)."""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

from app.exceptions import LLMValidationError, LLMGenerationError
from app.models.reranking import RankedChunk
from app.llm.strategies.gemini_llm import GeminiLLM


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove GEMINI_MODEL_NAME so tests use the code default, not .env."""
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)


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
    GeminiLLM.clear_client_cache()
    yield
    GeminiLLM.clear_client_cache()


@pytest.fixture()
def mock_genai():
    """Inject a fake google.genai module with a mock Client."""
    mock_client_instance = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_instance)

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = mock_client_cls  # type: ignore[attr-defined]

    fake_types = types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = MagicMock  # type: ignore[attr-defined]
    fake_genai.types = fake_types  # type: ignore[attr-defined]

    # Ensure parent package exists
    fake_google = types.ModuleType("google")
    prev_genai = sys.modules.get("google.genai")
    prev_google = sys.modules.get("google")
    prev_types = sys.modules.get("google.genai.types")
    sys.modules["google"] = fake_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_types

    yield mock_client_instance

    if prev_google is None:
        sys.modules.pop("google", None)
    else:
        sys.modules["google"] = prev_google
    if prev_genai is None:
        sys.modules.pop("google.genai", None)
    else:
        sys.modules["google.genai"] = prev_genai
    if prev_types is None:
        sys.modules.pop("google.genai.types", None)
    else:
        sys.modules["google.genai.types"] = prev_types


def test_empty_candidates_returns_default_response() -> None:
    llm = GeminiLLM(api_key="test-key")
    result = llm.generate("query", [])
    assert "Aucun contexte" in result.answer
    assert result.strategy_name == "gemini"


def test_query_validation() -> None:
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="query"):
        llm.generate("", _make_candidates())
    with pytest.raises(LLMValidationError, match="query"):
        llm.generate("   ", _make_candidates())


def test_max_tokens_validation() -> None:
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="max_tokens"):
        llm.generate("q", _make_candidates(), max_tokens=0)
    with pytest.raises(LLMValidationError, match="max_tokens"):
        llm.generate("q", _make_candidates(), max_tokens=-1)
    with pytest.raises(LLMValidationError, match="max_tokens"):
        llm.generate("q", _make_candidates(), max_tokens=True)  # type: ignore[arg-type]


def test_temperature_validation() -> None:
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="temperature"):
        llm.generate("q", _make_candidates(), temperature="hot")  # type: ignore[arg-type]
    with pytest.raises(LLMValidationError, match="temperature"):
        llm.generate("q", _make_candidates(), temperature=True)  # type: ignore[arg-type]


def test_candidate_type_validation() -> None:
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMValidationError, match="candidates"):
        llm.generate("q", ["not a chunk"])  # type: ignore[list-item]


def test_generate_returns_response(mock_genai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.text = "Réponse test Gemini."
    mock_resp.usage_metadata.prompt_token_count = 30
    mock_resp.usage_metadata.candidates_token_count = 15
    mock_genai.models.generate_content.return_value = mock_resp

    llm = GeminiLLM(api_key="test-key")
    result = llm.generate("Pourquoi vibre le moteur ?", _make_candidates(2))

    assert result.answer == "Réponse test Gemini."
    assert result.strategy_name == "gemini"
    assert result.model_name == os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    assert result.tokens_input == 30
    assert result.tokens_output == 15
    assert result.duration_ms >= 0
    assert len(result.citations) == 2
    assert result.metadata["candidates_count"] == 2


def test_generate_citations_match_candidates(mock_genai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.text = "ok"
    mock_resp.usage_metadata.prompt_token_count = 10
    mock_resp.usage_metadata.candidates_token_count = 5
    mock_genai.models.generate_content.return_value = mock_resp

    llm = GeminiLLM(api_key="test-key")
    candidates = _make_candidates(3)
    result = llm.generate("q", candidates)

    assert len(result.citations) == 3
    for i, citation in enumerate(result.citations):
        assert citation.chunk_id == candidates[i].chunk_id
        assert citation.source_name == candidates[i].source_name
        assert citation.rerank_score == candidates[i].rerank_score


def test_generate_none_text_returns_fallback(mock_genai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.text = None
    mock_resp.usage_metadata = None
    mock_genai.models.generate_content.return_value = mock_resp

    llm = GeminiLLM(api_key="test-key")
    result = llm.generate("q", _make_candidates(1))
    assert len(result.answer) > 0


def test_generate_empty_text_returns_fallback(mock_genai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.usage_metadata = None
    mock_genai.models.generate_content.return_value = mock_resp

    llm = GeminiLLM(api_key="test-key")
    result = llm.generate("q", _make_candidates(1))
    assert len(result.answer) > 0


def test_generate_wraps_rate_limit_error(mock_genai: MagicMock) -> None:
    from app.exceptions import LLMRateLimitError
    mock_genai.models.generate_content.side_effect = Exception("Rate limit 429 exceeded")
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMRateLimitError):
        llm.generate("q", _make_candidates(1))


def test_generate_wraps_connection_error(mock_genai: MagicMock) -> None:
    from app.exceptions import LLMConnectionError
    mock_genai.models.generate_content.side_effect = Exception("Connection timeout")
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMConnectionError):
        llm.generate("q", _make_candidates(1))


def test_generate_wraps_generic_error(mock_genai: MagicMock) -> None:
    mock_genai.models.generate_content.side_effect = RuntimeError("boom")
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMGenerationError):
        llm.generate("q", _make_candidates(1))


def test_init_validates_api_key() -> None:
    with pytest.raises(LLMValidationError, match="api_key"):
        GeminiLLM(api_key="")


def test_init_validates_model_name() -> None:
    with pytest.raises(LLMValidationError, match="model_name"):
        GeminiLLM(api_key="test", model_name="  ")


def test_properties() -> None:
    llm = GeminiLLM(api_key="test-key", model_name="gemini-pro")
    assert llm.model_name == "gemini-pro"


def test_generate_empty_usage_metadata(mock_genai: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.text = "answer"
    mock_resp.usage_metadata = None
    mock_genai.models.generate_content.return_value = mock_resp

    llm = GeminiLLM(api_key="test-key")
    result = llm.generate("q", _make_candidates(1))
    assert result.tokens_input == 0
    assert result.tokens_output == 0


def test_generate_resource_exhausted_raises_rate_limit(mock_genai: MagicMock) -> None:
    from app.exceptions import LLMRateLimitError
    mock_genai.models.generate_content.side_effect = Exception("resource_exhausted")
    llm = GeminiLLM(api_key="test-key")
    with pytest.raises(LLMRateLimitError):
        llm.generate("q", _make_candidates(1))
