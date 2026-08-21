"""Manual integration test for OpenAILLM.

Run with: GMAO_MANUAL_TESTS=1 python -m pytest tests/unit/llm/test_llm_manual.py -v -s
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os

import pytest

if not os.environ.get("GMAO_MANUAL_TESTS"):
    pytest.skip(
        "Manual tests skipped by default. "
        "Set GMAO_MANUAL_TESTS=1 to enable.",
        allow_module_level=True,
    )

from app.models.reranking import RankedChunk
from app.llm.strategies.openai_llm import OpenAILLM


def _make_realistic_candidates() -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id="report.pdf:42",
            content=(
                "Le moteur électrique présente des vibrations anormales lors des cycles "
                "de démarrage. Cela peut être dû à un déséquilibre du rotor ou à une "
                "usure des paliers."
            ),
            source_name="report.pdf",
            source_type="document",
            retrieval_score=0.81,
            rerank_score=0.997,
            rank=1,
            retrieval_strategy="qdrant",
            reranker_strategy="cross-encoder",
        ),
        RankedChunk(
            chunk_id="manual.pdf:15",
            content=(
                "Procédure de maintenance préventive : vérifier le niveau d'huile "
                "toutes les 500 heures de fonctionnement."
            ),
            source_name="manual.pdf",
            source_type="document",
            retrieval_score=0.74,
            rerank_score=0.45,
            rank=2,
            retrieval_strategy="qdrant",
            reranker_strategy="cross-encoder",
        ),
        RankedChunk(
            chunk_id="panne:7",
            content=(
                "Panne signalée le 2024-03-15 : la pompe à huile ne débite plus. "
                "Remplacement du joint torique effectué."
            ),
            source_name="panne:7",
            source_type="panne",
            retrieval_score=0.68,
            rerank_score=0.30,
            rank=3,
            retrieval_strategy="qdrant",
            reranker_strategy="cross-encoder",
        ),
    ]


@pytest.mark.manual
def test_manual_llm_generation() -> None:
    query = "Pourquoi le moteur présente-t-il des vibrations ?"
    candidates = _make_realistic_candidates()

    llm = OpenAILLM()
    result = llm.generate(query, candidates)

    print("\n" + "=" * 60)
    print("LLM RESPONSE")
    print("=" * 60)
    print(f'\nQuery:\n"{query}"\n')
    print(f"Model: {result.model_name}")
    print(f"Tokens: {result.tokens_input} input / {result.tokens_output} output")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print(f"\nCitations ({len(result.citations)}):")
    for c in result.citations:
        print(f"  - {c.source_name} (score: {c.rerank_score:.3f})")
    print(f"\nAnswer:\n{result.answer}")
    print("=" * 60)

    assert len(result.answer) > 0
    assert result.tokens_input > 0
