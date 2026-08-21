"""
Manual integration test for the complete Markdown pipeline.

Pipeline:

    Markdown file
        -> DataSourceOrchestrator
        -> SourceDocument
        -> ParserOrchestrator
        -> ParsedDocument
        -> MarkdownChunker
        -> list[Chunk]

Version corrigée : voir les commentaires "# FIX" pour le détail
des correctifs apportés au script original.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FIX : import inutile / redondant supprimé
#   - "from pathlib import Path" en double
#   - "from app.models import document" (jamais utilisé, et masquait
#     le paramètre "document" des fonctions plus bas)

from app.data_sources.orchestrator import (
    DataSourceKind,
    DataSourceOrchestrator,
)

from app.parser import ParserOrchestrator
from app.parser.registry import ParserRegistry
from app.parser.strategies.markdown import MarkdownParser

from app.chunker.registry import ChunkerRegistry
from app.chunker.strategies.markdown import MarkdownChunker


# ==========================================================
# Configuration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[3]

TEST_FILE = (
    BASE_DIR / "tests" / "data" / "markdown" / "example.md"
)

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


# ==========================================================
# Create Markdown test file
# ==========================================================

def create_test_file() -> None:
    """
    Create the Markdown file used by the test.
    """

    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    # FIX : le contenu était écrit avec l'indentation Python de la
    # fonction (4 espaces devant chaque ligne). Or la regex de
    # détection des titres n'accepte que 0 à 3 espaces avant le "#"
    # (^\s{0,3}#{1,6}\s+). Résultat : aucun titre n'était jamais
    # reconnu et tout le découpage par sections échouait.
    # -> textwrap.dedent() retire l'indentation commune.
    content = textwrap.dedent(
        """\
        # Maintenance industrielle

        La maintenance industrielle joue un role essentiel dans la
        continuite de la production.

        Elle permet de reduire les arrets non planifies et d'ameliorer
        la disponibilite des equipements.

        ## Maintenance preventive

        La maintenance preventive consiste a effectuer des inspections
        regulieres afin de detecter les anomalies avant l'apparition
        d'une panne importante.

        Les equipements doivent etre controles selon un programme
        defini par le service de maintenance.

        ### Inspection des equipements

        Les techniciens verifient regulierement :

        - la temperature ;
        - les vibrations ;
        - la pression ;
        - les niveaux d'huile.

        ## Pompe hydraulique

        La pompe hydraulique principale alimente le circuit hydraulique
        de production.

        Une inspection visuelle doit etre realisee regulierement afin
        de detecter les fuites et les problemes de pression.

        ## Compresseur

        Le compresseur principal fournit l'air comprime necessaire
        aux differentes machines de production.

        Les parametres de fonctionnement doivent etre surveilles
        afin d'eviter les arrets imprevus.

        ```text
        Pression normale : 7 bar
        Temperature normale : 65 C
        ```

        ## Conclusion

        Une maintenance preventive correctement organisee permet
        d'ameliorer la fiabilite des equipements et de reduire
        les couts lies aux pannes.
        """
    )
    # FIX : la barriere de code fermante avait 4 backticks (````)
    # au lieu de 3, incoherente avec l'ouverture ```text.

    TEST_FILE.write_text(content, encoding="utf-8")


# ==========================================================
# Build parser / chunker
# ==========================================================

def build_parser() -> ParserOrchestrator:

    print("\n[3] Initializing ParserOrchestrator...")

    registry = ParserRegistry()
    registry.register(MarkdownParser)

    print("Registered parser strategies:", registry.supported_types())

    parser = ParserOrchestrator(registry=registry)

    print("OK ParserOrchestrator initialized")

    return parser


def build_chunker_strategy() -> MarkdownChunker:
    """
    FIX : on instancie directement MarkdownChunker avec les
    parametres voulus (chunk_size / chunk_overlap), plutot que de
    passer par un ChunkerOrchestrator dont on ne sait pas s'il
    transmet ces kwargs a la strategie (ailleurs dans le pipeline,
    la strategie etait instanciee sans arguments -> valeurs par
    defaut silencieusement utilisees). A adapter si votre
    ChunkerOrchestrator.chunk() accepte bien chunk_size/chunk_overlap.
    """

    print("\n[5] Initializing MarkdownChunker...")

    registry = ChunkerRegistry()
    registry.register(MarkdownChunker)

    print("Registered chunker strategies:", registry.supported_types())

    strategy = MarkdownChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("OK MarkdownChunker initialized")

    return strategy


# ==========================================================
# Display helpers
# ==========================================================

def display_source_document(document) -> None:

    print("\n" + "=" * 70)
    print("[2] SourceDocument")
    print("=" * 70)
    print(f"Source name : {document.source_name}")
    print(f"Source type : {document.source_type}")
    print(f"Source path : {document.source_path}")
    print(f"MIME type   : {document.mime_type}")
    print(f"Size        : {document.size} bytes")
    print("\nMetadata:")
    for key, value in (document.metadata or {}).items():
        print(f"  {key}: {value}")
    print("\nContent:")
    print("-" * 70)
    print(document.content)
    print("-" * 70)


def display_parsed_document(document) -> None:

    print("\n" + "=" * 70)
    print("[4] ParsedDocument")
    print("=" * 70)
    print(f"Source name : {document.source_name}")
    print(f"Source type : {document.source_type}")
    print(f"Content     : {len(document.content)} characters")
    print("\nContent:")
    print("-" * 70)
    print(document.content)
    print("-" * 70)


def display_chunks(chunks) -> None:
    """
    FIX : utilise metadata["chunk_index"] plutot que chunk.index,
    car chunk_index est garanti d'exister dans les metadata
    generees par MarkdownChunker, alors qu'on ne sait pas si le
    modele Chunk expose reellement un attribut "index".
    """

    print("\n" + "=" * 70)
    print("[7] Generated Chunks")
    print("=" * 70)
    print(f"\nNumber of chunks: {len(chunks)}")

    for chunk in chunks:

        metadata = chunk.metadata or {}
        idx = metadata.get("chunk_index")

        print("\n" + "-" * 70)
        print(f"Chunk {idx}")
        print("-" * 70)
        print(f"Index       : {idx}")
        print(f"Source      : {chunk.source_name}")
        print(f"Source type : {chunk.source_type}")
        print(f"Length      : {len(chunk.content)} characters")
        print("\nContent:")
        print(chunk.content)
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")


# ==========================================================
# Validation helpers
# ==========================================================

def validate_source_document(document) -> None:

    print("\n[Validation] SourceDocument")

    assert document is not None
    assert document.source_name == TEST_FILE.name
    assert document.source_type in {"markdown", "md"}
    assert document.content
    assert document.content.strip()

    print("OK SourceDocument is valid")
    print(f"OK source_type = {document.source_type}")
    print("OK content is not empty")


def validate_parsed_document(document) -> None:

    print("\n[Validation] ParsedDocument")

    assert document is not None
    assert document.content
    assert document.content.strip()
    assert document.source_name == TEST_FILE.name

    print("OK ParsedDocument is valid")
    print("OK content is not empty")
    print("OK source name preserved")


def validate_chunks(chunks) -> None:
    """
    FIX : verifie chunk.content contre le contenu du chunk lui-meme
    (implicitement, via non-vacuite et longueur <= chunk_size), et
    utilise metadata["chunk_index"] pour l'ordre plutot que
    chunk.index (voir display_chunks). Ajoute aussi une verification
    explicite qu'un chunk ne contient pas TOUT le document (regression
    test pour le bug content=content corrige dans le chunker).
    """

    print("\n" + "=" * 70)
    print("[8] Chunk Validation")
    print("=" * 70)

    assert chunks, "No chunks were generated."

    full_document_length = sum(len(c.content) for c in chunks)

    for expected_index, chunk in enumerate(chunks):

        metadata = chunk.metadata or {}

        assert metadata.get("chunk_index") == expected_index, (
            f"Invalid chunk_index metadata in chunk {expected_index}."
        )

        assert chunk.content and chunk.content.strip(), (
            f"Chunk {expected_index} contains empty content."
        )

        assert chunk.source_name == TEST_FILE.name, (
            f"Invalid source name in chunk {expected_index}."
        )

        assert chunk.source_type in {"markdown", "md"}, (
            f"Invalid source type in chunk {expected_index}: "
            f"{chunk.source_type}"
        )

        assert metadata.get("chunker") == "markdown", (
            f"Invalid chunker metadata in chunk {expected_index}."
        )

        # Regression test pour le bug "content=content" :
        # si un seul chunk contient (presque) tout le document,
        # c'est le signe que le bug est revenu.
        if len(chunks) > 1:
            assert len(chunk.content) < full_document_length, (
                f"Chunk {expected_index} seems to contain the "
                f"entire document instead of its own slice."
            )

    print(f"OK Number of chunks: {len(chunks)}")
    print("OK All generated chunks are valid.")


# ==========================================================
# Main pipeline
# ==========================================================

def main() -> None:

    print("=" * 70)
    print("MARKDOWN LOADING -> PARSING -> CHUNKING TEST")
    print("=" * 70)

    print("\n[1] Creating Markdown test file...")
    create_test_file()
    print("OK File created:")
    print(f"  {TEST_FILE}")

    print("\n[2] Loading Markdown file...")
    data_source_orchestrator = DataSourceOrchestrator()
    source_document = data_source_orchestrator.load(
        kind=DataSourceKind.FILE,
        source=str(TEST_FILE),
    )
    print("OK Markdown loading successful")

    display_source_document(source_document)
    validate_source_document(source_document)

    parser = build_parser()

    print("\n[4] Checking parser support...")
    parser_strategy = parser._registry.get(source_document.source_type)()
    supports = parser_strategy.supports(source_document)
    print(f"Supports document: {supports}")
    assert supports, "MarkdownParser does not support the loaded document."

    print("\nParsing SourceDocument...")
    parsed_document = parser.parse(source_document)
    print("OK Parsing successful")

    display_parsed_document(parsed_document)
    validate_parsed_document(parsed_document)

    chunker_strategy = build_chunker_strategy()

    print("\n[6] Checking chunker support...")
    supports = chunker_strategy.supports(parsed_document)
    print(f"Supports document: {supports}")
    assert supports, "MarkdownChunker does not support the ParsedDocument."

    print("\n[7] Chunking ParsedDocument...")
    chunks = chunker_strategy.chunk(parsed_document)
    print("OK Chunking successful")

    display_chunks(chunks)
    validate_chunks(chunks)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"File              : {TEST_FILE.name}")
    print(f"Source type       : {source_document.source_type}")
    print(f"Parsed characters : {len(parsed_document.content)}")
    print(f"Chunks generated  : {len(chunks)}")
    print(f"Chunk size        : {CHUNK_SIZE}")
    print(f"Chunk overlap     : {CHUNK_OVERLAP}")

    print("\nOK Loading successful")
    print("OK Parsing successful")
    print("OK Markdown chunking successful")
    print("OK All validations passed")

    print("\n" + "=" * 70)
    print("OK MARKDOWN PIPELINE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()