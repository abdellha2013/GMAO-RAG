"""Manual integration test for CrossEncoderReranker.

Run with: python -m pytest tests/unit/reranker/test_reranker_manual.py -v -s -m manual
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# import pytest

# if not os.environ.get("GMAO_MANUAL_TESTS"):
#     pytest.skip(
#         "Manual tests skipped by default. "
#         "Set GMAO_MANUAL_TESTS=1 to enable.",
#         allow_module_level=True,
#     )

from app.models.retrieval import RetrievedChunk
from app.reranker.strategies.cross_encoder import CrossEncoderReranker


def _make_realistic_candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="report.pdf:42",
            content=(
                "Le moteur électrique présente des vibrations anormales lors des cycles "
                "de démarrage. Cela peut être dû à un déséquilibre du rotor ou à une "
                "usure des paliers."
            ),
            score=0.81,
            rank=1,
            source_name="report.pdf",
            source_type="document",
            retrieval_strategy="qdrant",
        ),
        RetrievedChunk(
            chunk_id="manual.pdf:15",
            content=(
                "Procédure de maintenance préventive : vérifier le niveau d'huile "
                "toutes les 500 heures de fonctionnement."
            ),
            score=0.74,
            rank=2,
            source_name="manual.pdf",
            source_type="document",
            retrieval_strategy="qdrant",
        ),
        RetrievedChunk(
            chunk_id="panne:7",
            content=(
                "Panne signalée le 2024-03-15 : la pompe à huile ne débite plus. "
                "Remplacement du joint torique effectué."
            ),
            score=0.68,
            rank=3,
            source_name="panne:7",
            source_type="panne",
            retrieval_strategy="qdrant",
        ),
        RetrievedChunk(
            chunk_id="report.pdf:88",
            content=(
                "Les capteurs de température indiquent des valeurs dans les normes. "
                "Aucune alarme enregistrée sur la période analysée."
            ),
            score=0.55,
            rank=4,
            source_name="report.pdf",
            source_type="document",
            retrieval_strategy="qdrant",
        ),
    ]


# @pytest.mark.manual
def test_manual_reranking() -> None:
    query = "Pourquoi le moteur présente-t-il des vibrations ?"
    candidates = _make_realistic_candidates()

    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    results = reranker.rerank(query, candidates, top_k=len(candidates))

    print("\n" + "=" * 60)
    print("RERANKING RESULT")
    print("=" * 60)
    print(f'\nQuery:\n"{query}"\n')
    print(f"Number of candidates: {len(candidates)}")

    for r in results:
        print("-" * 60)
        print(f"Rank: {r.rank}")
        print(f"Chunk ID: {r.chunk_id}")
        print(f"Retrieval score: {r.retrieval_score:.2f}")
        print(f"Rerank score: {r.rerank_score:.4f}")
        print(f"Source: {r.source_name}")
        print(f"\nContent:\n{r.content[:120]}...")
        print("-" * 60)

    assert len(results) > 0
    assert results[0].rank == 1



test_manual_reranking()