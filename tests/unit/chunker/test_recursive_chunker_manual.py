"""
Manual test for RecursiveChunker.

This test verifies the RecursiveChunker strategy with every
supported textual source type:

    - txt
    - text
    - pdf
    - docx
    - html

The test uses one ParsedDocument for each source type.

The objective is to validate:

    ParsedDocument
        ↓
    RecursiveChunker
        ↓
    list[Chunk]

This is a manual/integration-style test and does not require
pytest.
"""

from __future__ import annotations

from app.chunker.strategies.recursive import RecursiveChunker
from app.models.parsing import ParsedDocument
from pathlib import Path 
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


# ============================================================
# Test Documents
# ============================================================

DOCUMENTS = [
    {
        "source_name": "example.txt",
        "source_type": "txt",
        "mime_type": "text/plain",
        "content": """
La maintenance industrielle joue un rôle essentiel dans la
continuité de la production.

Une maintenance préventive permet d'identifier les anomalies
avant qu'elles ne provoquent une panne importante.

Les équipements industriels doivent être surveillés
régulièrement afin de réduire les arrêts non planifiés.

Les interventions de maintenance doivent être enregistrées
avec précision afin de conserver un historique fiable des
équipements.

Cet historique peut ensuite être utilisé pour analyser les
pannes et améliorer les plans de maintenance.
""",
    },
    {
        "source_name": "example-text",
        "source_type": "text",
        "mime_type": "text/plain",
        "content": """
Le compresseur principal alimente le réseau pneumatique de
l'usine.

Une inspection régulière permet de vérifier la pression,
la température et les vibrations du compresseur.

Lorsque les valeurs mesurées dépassent les seuils définis,
une intervention de maintenance doit être planifiée.

Les opérations réalisées doivent être enregistrées dans
le système de gestion de maintenance.
""",
    },
    {
        "source_name": "example.pdf",
        "source_type": "pdf",
        "mime_type": "application/pdf",
        "content": """
Rapport de maintenance industrielle

Introduction

Ce document présente l'état général des équipements de
production.

Équipement 1

La pompe hydraulique principale fonctionne normalement.
Une inspection visuelle a été réalisée et aucune anomalie
critique n'a été détectée.

Équipement 2

Le moteur électrique du convoyeur présente des vibrations
supérieures aux valeurs habituelles. Une intervention de
maintenance préventive est recommandée.

Conclusion

Les équipements doivent continuer à être surveillés afin
de prévenir les arrêts imprévus de la production.
""",
    },
    {
        "source_name": "example.docx",
        "source_type": "docx",
        "mime_type": (
            "application/"
            "vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        "content": """
Rapport d'intervention

Équipement : Pompe hydraulique principale

La pompe hydraulique principale a fait l'objet d'une
inspection préventive.

Les techniciens ont vérifié l'état général de la pompe,
les niveaux de pression et les éventuelles fuites.

Aucune anomalie critique n'a été constatée pendant
l'intervention.

Prochaine intervention

Une nouvelle inspection devra être réalisée dans trois mois.
""",
    },
    {
        "source_name": "example.html",
        "source_type": "html",
        "mime_type": "text/html",
        "content": """
Maintenance des équipements

Les équipements industriels doivent être maintenus afin
d'assurer la continuité de la production.

Pompe hydraulique

La pompe hydraulique principale est utilisée pour alimenter
le circuit hydraulique de production.

Un contrôle périodique permet de détecter les fuites,
les vibrations et les problèmes de pression.

Compresseur

Le compresseur principal fournit l'air comprimé nécessaire
aux différentes machines de production.

Les paramètres de fonctionnement doivent être surveillés
régulièrement.
""",
    },
]


# ============================================================
# ParsedDocument factory
# ============================================================

def build_document(data: dict) -> ParsedDocument:
    """
    Build a ParsedDocument from test data.
    """

    return ParsedDocument(
        source_name=data["source_name"],
        source_type=data["source_type"],
        content=data["content"].strip(),
        metadata={
            "test": True,
            "mime_type": data["mime_type"],
        },
    )


# ============================================================
# Display helpers
# ============================================================

def print_separator() -> None:
    """Print a visual separator."""
    print("=" * 80)


def print_chunk(
    index: int,
    chunk,
) -> None:
    """Display one generated chunk."""

    print(f"\n--- Chunk {index + 1} ---")

    print(f"Index       : {chunk.chunk_index}")
    print(f"Source      : {chunk.source_name}")
    print(f"Source type : {chunk.source_type}")
    print(f"Length      : {len(chunk.content)} characters")

    print("\nContent:")
    print(chunk.content)

    print("\nMetadata:")
    for key, value in chunk.metadata.items():
        print(f"  {key}: {value}")


# ============================================================
# Main test
# ============================================================

def main() -> None:
    """
    Execute the RecursiveChunker manual test.
    """

    print_separator()
    print("RECURSIVE CHUNKER MANUAL TEST")
    print_separator()

    # ========================================================
    # 1. Initialize chunker
    # ========================================================

    print("\n[1] Initializing RecursiveChunker...")

    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("✓ RecursiveChunker initialized")

    print(f"Strategy       : {chunker.name}")
    print(f"Chunk size     : {chunker.chunk_size}")
    print(f"Chunk overlap  : {chunker.chunk_overlap}")
    print(f"Supported type : {chunker.source_types}")

    # ========================================================
    # 2. Test every source type
    # ========================================================

    total_documents = len(DOCUMENTS)
    successful = 0
    failed = 0

    for document_number, data in enumerate(
        DOCUMENTS,
        start=1,
    ):

        print_separator()

        print(
            f"\n[{document_number}/{total_documents}] "
            f"Testing {data['source_type'].upper()}"
        )

        print_separator()

        # ----------------------------------------------------
        # Build ParsedDocument
        # ----------------------------------------------------

        document = build_document(data)

        print("\n[2] ParsedDocument")

        print(
            f"Source name : {document.source_name}"
        )

        print(
            f"Source type : {document.source_type}"
        )

        print(
            f"Content     : "
            f"{len(document.content)} characters"
        )

        # ----------------------------------------------------
        # Check support
        # ----------------------------------------------------

        print("\n[3] Checking support...")

        supported = chunker.supports(document)

        print(
            f"Supports document: {supported}"
        )

        if not supported:

            print(
                "✗ ERROR: RecursiveChunker does "
                "not support this source type."
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # Chunk document
        # ----------------------------------------------------

        print("\n[4] Chunking document...")

        try:

            chunks = chunker.chunk(document)

        except Exception as exc:

            print(
                "\n✗ Chunking failed:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            failed += 1
            continue

        print(
            f"✓ Chunking successful"
        )

        # ----------------------------------------------------
        # Validate result
        # ----------------------------------------------------

        print("\n[5] Validation")

        if not chunks:

            print(
                "✗ ERROR: No chunks were generated."
            )

            failed += 1
            continue

        print(
            f"✓ Number of chunks: {len(chunks)}"
        )

        # ----------------------------------------------------
        # Validate every chunk
        # ----------------------------------------------------

        validation_ok = True

        for index, chunk in enumerate(chunks):

            # Content must not be empty
            if not chunk.content.strip():

                print(
                    f"✗ Chunk {index + 1} "
                    "contains empty content."
                )

                validation_ok = False

            # Chunk must not exceed configured size
            if len(chunk.content) > CHUNK_SIZE:

                print(
                    f"✗ Chunk {index + 1} "
                    f"exceeds chunk_size: "
                    f"{len(chunk.content)} > "
                    f"{CHUNK_SIZE}"
                )

                validation_ok = False

            # Correct source
            if chunk.source_name != document.source_name:

                print(
                    f"✗ Chunk {index + 1} "
                    "has incorrect source_name."
                )

                validation_ok = False

            # Correct source type
            if chunk.source_type != document.source_type:

                print(
                    f"✗ Chunk {index + 1} "
                    "has incorrect source_type."
                )

                validation_ok = False

            # Correct index
            if chunk.chunk_index != index:

                print(
                    f"✗ Chunk {index + 1} "
                    f"has incorrect chunk_index: "
                    f"{chunk.chunk_index}"
                )

                validation_ok = False

        if validation_ok:

            print(
                "✓ All generated chunks are valid."
            )

        else:

            print(
                "✗ Chunk validation failed."
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # Display chunks
        # ----------------------------------------------------

        print("\n[6] Generated chunks")

        for index, chunk in enumerate(chunks):

            print_chunk(
                index,
                chunk,
            )

        successful += 1

    # ========================================================
    # Final summary
    # ========================================================

    print_separator()

    print("\nFINAL SUMMARY")

    print_separator()

    print(
        f"Documents tested : {total_documents}"
    )

    print(
        f"Successful       : {successful}"
    )

    print(
        f"Failed           : {failed}"
    )

    print(
        f"Chunk size       : {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap    : {CHUNK_OVERLAP}"
    )

    if failed == 0:

        print(
            "\n✓ ALL RECURSIVE CHUNKER TESTS PASSED"
        )

    else:

        print(
            "\n✗ SOME RECURSIVE CHUNKER TESTS FAILED"
        )

    print_separator()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()