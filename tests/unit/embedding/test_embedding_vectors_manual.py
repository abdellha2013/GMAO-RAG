"""Manual demonstration: generate and display real embedding vectors.

Run directly; it intentionally downloads/loads the configured local model:

    .venv/bin/python tests/unit/embedding/test_embedding_vectors_manual.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding import build_default_orchestrator
from app.exceptions import EmbeddingModelError
from app.models.chunk import Chunk
from app.models.embedding import Embedding


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    """Return cosine similarity between two embedding vectors."""
    dot_product = sum(
        first * second
        for first, second in zip(left.vector, right.vector, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left.vector))
    right_norm = math.sqrt(sum(value * value for value in right.vector))

    return dot_product / (left_norm * right_norm)


def format_vector(vector: tuple[float, ...], columns: int = 8) -> str:
    """Format the complete vector in readable rows."""
    values = [f"{value: .6f}" for value in vector]
    rows = [
        ", ".join(values[index:index + columns])
        for index in range(0, len(values), columns)
    ]
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def main() -> None:
    chunks = [
        Chunk(
            content="Le moteur présente une vibration anormale.",
            chunk_index=0,
            source_name="demo-maintenance",
            source_type="txt",
            chunk_id="demo-maintenance:0",
        ),
        Chunk(
            content="Une vibration excessive est détectée sur le moteur.",
            chunk_index=1,
            source_name="demo-maintenance",
            source_type="txt",
            chunk_id="demo-maintenance:1",
        ),
        Chunk(
            content="Le stock de pièces détachées est insuffisant.",
            chunk_index=2,
            source_name="demo-maintenance",
            source_type="txt",
            chunk_id="demo-maintenance:2",
        ),
    ]

    print("=" * 72)
    print("GMAO-RAG — Démonstration des embeddings")
    print("=" * 72)

    try:
        orchestrator = build_default_orchestrator(
            batch_size=8,
            normalize_embeddings=True,
            device="auto",
        )
        embeddings = orchestrator.embed(chunks)
    except EmbeddingModelError as exc:
        print(f"\nImpossible de charger le modèle : {exc}")
        print("Installez les dépendances avec : uv sync")
        raise SystemExit(1) from exc

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        norm = math.sqrt(sum(value * value for value in embedding.vector))
        print(f"\nChunk : {chunk.chunk_id}")
        print(f"Texte : {chunk.content}")
        print(f"Modèle : {embedding.model_name}")
        print(f"Dimension : {embedding.dimension}")
        print(f"Norme L2 : {norm:.6f}")
        print("Vecteur complet :")
        print(format_vector(embedding.vector))

    print("\n" + "=" * 72)
    print("Similarités cosinus")
    print("=" * 72)
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        score = cosine_similarity(
            embeddings[left_index],
            embeddings[right_index],
        )
        print(
            f"{chunks[left_index].chunk_id} ↔ "
            f"{chunks[right_index].chunk_id} : {score:.4f}"
        )


if __name__ == "__main__":
    main()
