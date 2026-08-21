"""
Test d'intégration manuel pour le pipeline de chunking structuré.

Pipeline testé (StructuredChunker isolé) :

    Contenu brut (JSON / CSV)
        -> ParsedDocument
        -> StructuredChunker
        -> list[Chunk]

Contrairement au test Markdown, il n'existe pas ici de parser dédié
(JSONParser / CSVParser) dans le code fourni : on construit donc
directement les ParsedDocument pour isoler et valider le
comportement du StructuredChunker sur chaque source_type supporté
("json", "csv", et par extension "xlsx" / "mysql" qui suivent le
même chemin que "csv" côté extraction de records).

Usage :
    python test_structured_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 3 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.exceptions import ChunkerValidationError
from app.models.parsing import ParsedDocument
from app.chunker.registry import ChunkerRegistry
from app.chunker.strategies.structured import StructuredChunker


# ==========================================================
# Configuration
# ==========================================================

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40

SOURCE_NAME_JSON = "clients.json"
SOURCE_NAME_CSV = "clients.csv"


# ==========================================================
# Génération du contenu de test
# ==========================================================

def build_json_content() -> str:
    """
    Génère un tableau JSON de "records" (cas d'usage typique :
    export API, résultats de requête, etc.).
    """

    records = [
        {
            "id": i,
            "nom": f"Client {i}",
            "email": f"client{i}@example.com",
            "ville": "Beni Mellal" if i % 2 == 0 else "Casablanca",
            "solde": round(1000 + i * 37.5, 2),
            "actif": i % 3 != 0,
        }
        for i in range(1, 9)
    ]

    return json.dumps(records, ensure_ascii=False, indent=2)


def build_csv_content() -> str:
    """
    Génère un contenu CSV simple : une ligne = un "record" logique
    pour le chunker (traitement via _extract_line_records).
    """

    lines = ["id,nom,ville,solde"]
    for i in range(1, 13):
        ville = "Beni Mellal" if i % 2 == 0 else "Casablanca"
        lines.append(f"{i},Client {i},{ville},{1000 + i * 37.5:.2f}")

    return "\n".join(lines)


# ==========================================================
# Construction du chunker
# ==========================================================

def build_chunker_strategy() -> StructuredChunker:

    print("\n[1] Initializing StructuredChunker...")

    registry = ChunkerRegistry()
    registry.register(StructuredChunker)

    print("Registered chunker strategies:", registry.supported_types())

    strategy = StructuredChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("OK StructuredChunker initialized")
    print(f"  chunk_size    = {strategy.chunk_size}")
    print(f"  chunk_overlap = {strategy.chunk_overlap}")

    return strategy


# ==========================================================
# Construction des ParsedDocument de test
# ==========================================================

def build_parsed_document(
    source_name: str,
    source_type: str,
    content: str,
) -> ParsedDocument:

    return ParsedDocument(
        source_name=source_name,
        source_type=source_type,
        content=content,
        metadata={"origin": "manual_integration_test"},
    )


# ==========================================================
# Affichage
# ==========================================================

def display_parsed_document(document: ParsedDocument) -> None:

    print("\n" + "=" * 70)
    print(f"ParsedDocument [{document.source_type}]")
    print("=" * 70)
    print(f"Source name : {document.source_name}")
    print(f"Source type : {document.source_type}")
    print(f"Content     : {len(document.content)} characters")
    print("\nContent (extrait) :")
    print("-" * 70)
    preview = document.content[:400]
    print(preview + ("..." if len(document.content) > 400 else ""))
    print("-" * 70)


def display_chunks(chunks) -> None:

    print("\n" + "=" * 70)
    print("Generated Chunks")
    print("=" * 70)
    print(f"\nNumber of chunks: {len(chunks)}")

    for chunk in chunks:

        metadata = chunk.metadata or {}
        idx = metadata.get("chunk_index")

        print("\n" + "-" * 70)
        print(f"Chunk {idx}")
        print("-" * 70)
        print(f"Index               : {idx}")
        print(f"Source              : {chunk.source_name}")
        print(f"Source type         : {chunk.source_type}")
        print(f"Structured type     : {metadata.get('structured_source_type')}")
        print(f"Length              : {len(chunk.content)} characters")
        print("\nContent:")
        print(chunk.content)


# ==========================================================
# Validation
# ==========================================================

def validate_chunks(
    chunks,
    document: ParsedDocument,
    expected_source_type: str,
) -> None:
    """
    Vérifie :
      - la présence et la cohérence des chunk_index (metadata)
      - l'absence de chunk vide
      - le respect (avec tolérance overlap) de chunk_size
      - la préservation du source_name / source_type
      - la non-régression "un seul chunk contient tout le document"
        quand plusieurs chunks sont attendus
    """

    print("\n" + "=" * 70)
    print(f"[Validation] Chunks pour source_type='{expected_source_type}'")
    print("=" * 70)

    assert chunks, "Aucun chunk n'a été généré."

    full_content_length = len(document.content)
    tolerance = CHUNK_OVERLAP + 1  # overlap peut légèrement rallonger un chunk

    for expected_index, chunk in enumerate(chunks):

        metadata = chunk.metadata or {}

        assert metadata.get("chunk_index") == expected_index, (
            f"chunk_index invalide dans le chunk {expected_index} "
            f"(reçu: {metadata.get('chunk_index')})."
        )

        assert chunk.content and chunk.content.strip(), (
            f"Chunk {expected_index} contient un contenu vide."
        )

        assert chunk.source_name == document.source_name, (
            f"source_name invalide dans le chunk {expected_index}."
        )

        assert chunk.source_type == document.source_type, (
            f"source_type invalide dans le chunk {expected_index}: "
            f"{chunk.source_type}"
        )

        assert metadata.get("chunker") == "structured", (
            f"metadata 'chunker' invalide dans le chunk {expected_index}."
        )

        assert metadata.get("structured_source_type") == expected_source_type, (
            f"metadata 'structured_source_type' invalide dans le "
            f"chunk {expected_index}."
        )

        assert len(chunk.content) <= CHUNK_SIZE + tolerance, (
            f"Chunk {expected_index} dépasse chunk_size + tolérance "
            f"overlap ({len(chunk.content)} > {CHUNK_SIZE + tolerance})."
        )

        if len(chunks) > 1:
            assert len(chunk.content) < full_content_length, (
                f"Chunk {expected_index} semble contenir tout le "
                f"document au lieu d'un seul fragment."
            )

    print(f"OK Number of chunks: {len(chunks)}")
    print("OK Tous les chunks générés sont valides.")


def validate_error_cases(strategy: StructuredChunker) -> None:
    """
    Vérifie que le chunker rejette correctement :
      - un ParsedDocument vide
      - un source_type non supporté
    """

    print("\n" + "=" * 70)
    print("[Validation] Cas d'erreur")
    print("=" * 70)

    # Cas 1 : contenu vide
    empty_document = build_parsed_document(
        source_name="empty.json",
        source_type="json",
        content="   ",
    )
    try:
        strategy.chunk(empty_document)
        raise AssertionError(
            "Un ChunkerValidationError était attendu pour un contenu vide."
        )
    except ChunkerValidationError:
        print("OK Contenu vide correctement rejeté (ChunkerValidationError).")

    # Cas 2 : source_type non supporté
    unsupported_document = build_parsed_document(
        source_name="notes.txt",
        source_type="plaintext",
        content="ceci n'est pas un type structuré",
    )
    supported = strategy.supports(unsupported_document)
    assert not supported, (
        "supports() aurait dû renvoyer False pour un source_type "
        "non structuré."
    )
    print("OK supports() rejette correctement un source_type non structuré.")

    try:
        strategy.chunk(unsupported_document)
        raise AssertionError(
            "Un ChunkerValidationError était attendu pour un "
            "source_type non supporté."
        )
    except ChunkerValidationError:
        print("OK chunk() rejette correctement un source_type non supporté.")


# ==========================================================
# Pipeline principal
# ==========================================================

def run_for_source(
    strategy: StructuredChunker,
    source_name: str,
    source_type: str,
    content: str,
) -> None:

    print("\n" + "#" * 70)
    print(f"# TEST source_type = '{source_type}'")
    print("#" * 70)

    document = build_parsed_document(source_name, source_type, content)
    display_parsed_document(document)

    print("\nChecking chunker support...")
    supports = strategy.supports(document)
    print(f"Supports document: {supports}")
    assert supports, f"StructuredChunker ne supporte pas '{source_type}'."

    print("\nChunking ParsedDocument...")
    chunks = strategy.chunk(document)
    print("OK Chunking successful")

    display_chunks(chunks)
    validate_chunks(chunks, document, expected_source_type=source_type)


def main() -> None:

    print("=" * 70)
    print("STRUCTURED CHUNKER - TEST D'INTÉGRATION MANUEL")
    print("=" * 70)

    strategy = build_chunker_strategy()

    # ---- JSON -------------------------------------------------
    run_for_source(
        strategy,
        source_name=SOURCE_NAME_JSON,
        source_type="json",
        content=build_json_content(),
    )

    # ---- CSV (même chemin que xlsx / mysql côté extraction) ----
    run_for_source(
        strategy,
        source_name=SOURCE_NAME_CSV,
        source_type="csv",
        content=build_csv_content(),
    )

    # ---- Cas d'erreur -------------------------------------------
    validate_error_cases(strategy)

    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"chunk_size        : {CHUNK_SIZE}")
    print(f"chunk_overlap     : {CHUNK_OVERLAP}")
    print("Types testés      : json, csv")
    print("Cas d'erreur      : contenu vide, source_type non supporté")

    print("\nOK JSON chunking successful")
    print("OK CSV chunking successful")
    print("OK Cas d'erreur validés")

    print("\n" + "=" * 70)
    print("OK STRUCTURED CHUNKER PIPELINE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()