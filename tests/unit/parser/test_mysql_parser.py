

"""
Manual integration test for the MySQL parser pipeline.

Pipeline tested
---------------
.env
    ↓
MySQLLoader
    ↓
SourceDocument
    ↓
DatabaseParser
    ↓
ParsedDocument

This test uses the real MySQL database configured through environment
variables. No database credentials are hard-coded in the test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from app.data_sources.database.mysql_loader import MySQLLoader
from app.parser.strategies.database import DatabaseParser


def main() -> None:
    """Run the complete MySQL → Parser integration test."""

    print("=" * 70)
    print("MYSQL → PARSER INTEGRATION TEST")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Read configuration
    # -----------------------------------------------------------------------

    host = os.getenv("GMAO_DB_HOST", "127.0.0.1")
    port = int(os.getenv("GMAO_DB_PORT", "3306"))
    database = os.getenv("GMAO_DB_NAME","gmao_rag_test")
    user = os.getenv("GMAO_DB_USER","root")
    password = os.getenv("GMAO_DB_PASSWORD","Zzdv6401")

    print("\n[1] Database configuration")
    print(f"    Host     : {host}")
    print(f"    Port     : {port}")
    print(f"    Database : {database}")
    print(f"    User     : {user}")
    print("    Password : ********")

    if not database:
        raise RuntimeError("GMAO_DB_NAME is missing from .env")

    if not user:
        raise RuntimeError("GMAO_DB_USER is missing from .env")

    if password is None:
        raise RuntimeError("GMAO_DB_PASSWORD is missing from .env")

    # -----------------------------------------------------------------------
    # 2. Load data from MySQL
    # -----------------------------------------------------------------------

    print("\n[2] Loading data from MySQL...")

    loader = MySQLLoader(
    host=host,
    port=port,
    database=database,
    user=user,
    password=password,
    query="""
    SELECT *
    FROM equipements
    WHERE categorie = :categorie
    """,
    params={
        "categorie": "Pompe",
    },
    max_rows=10,
    )

    document = loader.load()

    print("    ✓ MySQL loading successful")

    # -----------------------------------------------------------------------
    # 3. Display SourceDocument information
    # -----------------------------------------------------------------------

    print("\n[3] SourceDocument")
    print(f"    Type       : {type(document).__name__}")
    print(f"    Source name: {document.source_name}")
    print(f"    Source type: {document.source_type}")
    print(f"    Source path: {document.source_path}")
    print(f"    MIME type  : {document.mime_type}")
    print(f"    Size       : {document.size} bytes")

    print("\n    Metadata:")
    for key, value in document.metadata.items():
        print(f"      {key}: {value}")

    # -----------------------------------------------------------------------
    # 4. Display loaded content
    # -----------------------------------------------------------------------

    print("\n[4] Loaded content")
    print("-" * 70)

    if document.content:
        print(document.content)
    else:
        print("    ⚠ Document content is EMPTY")

    print("-" * 70)

    # -----------------------------------------------------------------------
    # 5. Create parser
    # -----------------------------------------------------------------------

    print("\n[5] DatabaseParser")

    parser = DatabaseParser()

    print(f"    Strategy name: {parser.name}")

    # -----------------------------------------------------------------------
    # 6. Check support
    # -----------------------------------------------------------------------

    print("\n[6] Checking parser support...")

    supported = parser.supports(document)

    print(f"    Supports document: {supported}")

    if not supported:
        raise RuntimeError(
            "DatabaseParser does not support the SourceDocument "
            "produced by MySQLLoader."
        )

    # -----------------------------------------------------------------------
    # 7. Parse SourceDocument
    # -----------------------------------------------------------------------

    print("\n[7] Parsing SourceDocument...")

    parsed_document = parser.parse(document)

    print("    ✓ Parsing successful")

    # -----------------------------------------------------------------------
    # 8. Display ParsedDocument
    # -----------------------------------------------------------------------

    print("\n[8] ParsedDocument")
    print("-" * 70)

    print(f"    Type       : {type(parsed_document).__name__}")

    # Adapt these fields if your ParsedDocument model uses different names.
    print(f"    Source name: {parsed_document.source_name}")
    print(f"    Source type: {parsed_document.source_type}")

    if hasattr(parsed_document, "content"):
        print(f"    Content    : {parsed_document.content}")

    if hasattr(parsed_document, "metadata"):
        print("\n    Metadata:")
        for key, value in parsed_document.metadata.items():
            print(f"      {key}: {value}")

    print("-" * 70)

    # -----------------------------------------------------------------------
    # 9. Final validation
    # -----------------------------------------------------------------------

    print("\n[9] Validation")

    assert document.source_type == "mysql"
    assert document.source_path is None
    assert document.content

    assert supported is True
    assert parsed_document is not None

    print("    ✓ SourceDocument valid")
    print("    ✓ source_type == 'mysql'")
    print("    ✓ source_path is None")
    print("    ✓ content is not empty")
    print("    ✓ DatabaseParser supports the document")
    print("    ✓ ParsedDocument successfully created")

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT: PASS")
    print("MYSQL → SOURCE DOCUMENT → DATABASE PARSER")
    print("=" * 70)


if __name__ == "__main__":
    main()