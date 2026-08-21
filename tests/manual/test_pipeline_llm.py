"""Test manuel du pipeline complet : Retrieval → Reranking → LLM.

Exécution :
    cd /home/abdellah-daif/GMAO-RAG
    .venv/bin/python tests/manual/test_pipeline_llm.py
"""
from __future__ import annotations

from app.models.retrieval import RetrievedChunk
from app.reranker.strategies.cross_encoder import CrossEncoderReranker
from app.llm.strategies.gemini_llm import GeminiLLM


def main() -> None:
    query = "Pourquoi le moteur présente-t-il des vibrations ?"

    candidates = [
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

    # ── Étape 1 : Reranking ──────────────────────────────────
    print("=" * 60)
    print("ÉTAPE 1 : RERANKING")
    print("=" * 60)
    print(f'\nQuery : "{query}"\n')
    print(f"Candidats retrieval : {len(candidates)}")

    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    ranked = reranker.rerank(query, candidates, top_k=len(candidates))

    print(f"Après reranking   : {len(ranked)}\n")
    for r in ranked:
        print(f"  [{r.rank}] {r.source_name:20s} "
              f"retrieval={r.retrieval_score:.2f}  rerank={r.rerank_score:.4f}")
        print(f"      {r.content[:80]}...")
        print()

    # ── Étape 2 : LLM ───────────────────────────────────────
    print("=" * 60)
    print("ÉTAPE 2 : LLM (Gemini)")
    print("=" * 60 + "\n")

    llm = GeminiLLM()
    response = llm.generate(query, ranked)

    print(f"Modèle   : {response.model_name}")
    print(f"Tokens   : {response.tokens_input} input / {response.tokens_output} output")
    print(f"Durée    : {response.duration_ms:.0f}ms")
    print(f"Citations: {len(response.citations)}\n")

    print("-" * 60)
    print("RÉPONSE :")
    print("-" * 60)
    print(response.answer)
    print("-" * 60)


if __name__ == "__main__":
    main()
