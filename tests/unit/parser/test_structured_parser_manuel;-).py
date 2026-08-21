"""
Simple end-to-end pipeline test.

Pipeline tested
---------------
Input
    ↓
DataSourceOrchestrator
    ↓
SourceDocument
    ↓
ParserOrchestrator
    ↓
ParsedDocument

This test intentionally uses the public orchestrators instead of
calling loaders or parser strategies directly.

The input source can be changed in the configuration section below.
"""

from __future__ import annotations
from pathlib import Path 
import sys 

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.data_sources import DataSourceOrchestrator
from app.parser import ParserOrchestrator, ParserRegistry
from app.parser.strategies import (
    DatabaseParser,
    HTMLParser,
    MarkdownParser,
    StructuredParser,
    TextParser,

)


# ==========================================================
# Configuration
# ==========================================================

# Choose the input type:
#
# "file"
# "database"
# "api"
#
INPUT_KIND = "database"


# ----------------------------------------------------------
# File configuration
# ----------------------------------------------------------

FILE_PATH = Path(
    "tests/data/csv/unicode.csv"
)


# ----------------------------------------------------------
# Database configuration
# ----------------------------------------------------------
#
# Used only when INPUT_KIND = "database".
#

DATABASE_CONFIG = {
    "driver": "mysql",
    "host": "127.0.0.1",
    "port": 3306,
    "database": "gmao_rag_test",
    "user": "root",
    "password": "Zzdv6401",
    "table": "equipements",
}


# ----------------------------------------------------------
# API configuration
# ----------------------------------------------------------
#
# API is currently not implemented in the Data Source Layer.
# This section is prepared for future implementation.
#

API_CONFIG = {
    "url": "https://example.com/api",
    "method": "GET",
    "headers": {},
}


# ==========================================================
# Parser registry
# ==========================================================

def build_parser() -> ParserOrchestrator:
    """
    Build and configure the ParserOrchestrator.

    Returns
    -------
    ParserOrchestrator
        Configured parser orchestrator.
    """

    registry = ParserRegistry()

    registry.register(TextParser)
    # registry.register(TextParser)

    registry.register(MarkdownParser)
    # registry.register(MarkdownParser)

    registry.register(HTMLParser)

    registry.register(StructuredParser)
    # registry.register(StructuredParser)
    # registry.register(StructuredParser)

    registry.register(DatabaseParser)

    return ParserOrchestrator(registry)



# ==========================================================
# Data loading
# ==========================================================

def load_source():
    """
    Load the configured input through DataSourceOrchestrator.

    Returns
    -------
    SourceDocument
        Normalized source document.
    """

    orchestrator = DataSourceOrchestrator()

    if INPUT_KIND == "file":

        return orchestrator.load(
            FILE_PATH,
            kind="file",
        )

    if INPUT_KIND == "database":

        return orchestrator.load(
            DATABASE_CONFIG,
            kind="database",
        )

    if INPUT_KIND == "api":

        return orchestrator.load(
            API_CONFIG,
            kind="api",
        )

    raise ValueError(
        f"Unsupported INPUT_KIND: {INPUT_KIND!r}"
    )


# ==========================================================
# Display helpers
# ==========================================================

def display_source_document(document) -> None:
    """Display basic SourceDocument information."""

    print()
    print("=" * 70)
    print("SOURCE DOCUMENT")
    print("=" * 70)

    print(f"Type        : {type(document).__name__}")
    print(f"Source name : {document.source_name}")
    print(f"Source type : {document.source_type}")
    print(f"Source path : {document.source_path}")
    print(f"MIME type   : {document.mime_type}")
    print(f"Size        : {document.size}")

    print()
    print("Content preview:")
    print("-" * 70)

    preview = document.content[:1000]

    print(preview)

    if len(document.content) > 1000:
        print("\n... [content truncated]")


def display_parsed_document(document) -> None:
    """Display basic ParsedDocument information."""

    print()
    print("=" * 70)
    print("PARSED DOCUMENT")
    print("=" * 70)

    print(f"Type        : {type(document).__name__}")
    print(f"Source name : {document.source_name}")
    print(f"Source type : {document.source_type}")

    print()
    print("Parsed content preview:")
    print("-" * 70)

    preview = document.content[:1000]

    print(preview)

    if len(document.content) > 1000:
        print("\n... [content truncated]")


# ==========================================================
# Main pipeline
# ==========================================================

def main() -> None:
    """
    Execute the complete loading → parsing pipeline.
    """

    print("=" * 70)
    print("GMAO-RAG PIPELINE TEST")
    print("=" * 70)

    print()
    print(f"Input kind : {INPUT_KIND}")

    # ------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------

    print()
    print("[1] Loading source...")

    source_document = load_source()

    print("✓ Loading successful")

    # ------------------------------------------------------
    # 2. Display SourceDocument
    # ------------------------------------------------------

    display_source_document(source_document)

    # ------------------------------------------------------
    # 3. Build parser
    # ------------------------------------------------------

    print()
    print("[2] Initializing ParserOrchestrator...")

    parser = build_parser()

    print("✓ Parser initialized")

    # ------------------------------------------------------
    # 4. Parse
    # ------------------------------------------------------

    print()
    print("[3] Parsing SourceDocument...")

    parsed_document = parser.parse(
        source_document
    )

    print("✓ Parsing successful")

    # ------------------------------------------------------
    # 5. Display ParsedDocument
    # ------------------------------------------------------

    display_parsed_document(
        parsed_document
    )

    # ------------------------------------------------------
    # 6. Final validation
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE VALIDATION")
    print("=" * 70)

    assert source_document is not None
    assert parsed_document is not None

    assert source_document.content
    assert parsed_document.content

    print("✓ SourceDocument created")
    print("✓ SourceDocument contains content")
    print("✓ ParserOrchestrator executed")
    print("✓ ParsedDocument created")
    print("✓ ParsedDocument contains content")

    print()
    print("✓ COMPLETE PIPELINE SUCCESS")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    from app.exceptions import GMAOError

    try:
        main()
    except GMAOError as exc:
        print(f"\n✗ Erreur : {exc}")
        sys.exit(1)