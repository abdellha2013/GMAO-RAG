"""
Manual integration test for the PDF parser pipeline.

Pipeline tested
---------------
PDF file
    ↓
PDFLoader
    ↓
SourceDocument
    ↓
TextParser
    ↓
ParsedDocument
"""

from __future__ import annotations

import sys
from pathlib import Path
import sys
from pathlib import Path




# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from app.data_sources.file.pdf_loader import PDFLoader
from app.parser.strategies.text import TextParser


def main() -> None:
    """Run the complete PDF → TextParser integration test."""

    print("=" * 70)
    print("PDF → PARSER INTEGRATION TEST")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. PDF path
    # -----------------------------------------------------------------------

    pdf_path = PROJECT_ROOT / "tests" / "data" / "pdf" / "arabe_pdf.pdf"

    print("\n[1] PDF configuration")
    print(f"    Path: {pdf_path}")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF test file does not exist: {pdf_path}"
        )

    print("    ✓ PDF file exists")

    # -----------------------------------------------------------------------
    # 2. Load PDF
    # -----------------------------------------------------------------------

    print("\n[2] Loading PDF...")

    loader = PDFLoader(pdf_path)

    document = loader.load()

    print("    ✓ PDF loading successful")

    # -----------------------------------------------------------------------
    # 3. Display SourceDocument
    # -----------------------------------------------------------------------

    print("\n[3] SourceDocument")

    print(f"    Type       : {type(document).__name__}")
    print(f"    Source name: {document.source_name}")
    print(f"    Source type: {document.source_type}")
    print(f"    Source path: {document.source_path}")
    print(f"    MIME type  : {document.mime_type}")
    print(f"    Size       : {document.size} bytes")

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    print("\n    Metadata:")

    if document.metadata:
        for key, value in document.metadata.items():
            print(f"      {key}: {value}")
    else:
        print("      {}")

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
    # 5. Create TextParser
    # -----------------------------------------------------------------------

    print("\n[5] TextParser")

    parser = TextParser()

    print(f"    Strategy name: {parser.name}")

    # -----------------------------------------------------------------------
    # 6. Check support
    # -----------------------------------------------------------------------

    print("\n[6] Checking parser support...")

    supported = parser.supports(document)

    print(f"    Supports document: {supported}")

    if not supported:
        raise RuntimeError(
            "TextParser does not support the SourceDocument "
            "produced by PDFLoader."
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
    print(f"    Source name: {parsed_document.source_name}")
    print(f"    Source type: {parsed_document.source_type}")

    if hasattr(parsed_document, "content"):
        print("\n    Content:")
        print(parsed_document.content)

    if hasattr(parsed_document, "metadata"):
        print("\n    Metadata:")

        if parsed_document.metadata:
            for key, value in parsed_document.metadata.items():
                print(f"      {key}: {value}")
        else:
            print("      {}")

    print("-" * 70)

    # -----------------------------------------------------------------------
    # 9. Validation
    # -----------------------------------------------------------------------

    print("\n[9] Validation")

    assert document.source_type == "pdf"
    assert document.source_path is not None
    assert document.content
    assert document.mime_type is not None

    assert supported is True
    assert parsed_document is not None

    print("    ✓ SourceDocument valid")
    print("    ✓ source_type == 'pdf'")
    print("    ✓ source_path is not None")
    print("    ✓ content is not empty")
    print("    ✓ MIME type is available")
    print("    ✓ TextParser supports the PDF document")
    print("    ✓ ParsedDocument successfully created")

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT: PASS")
    print("PDF → PDFLoader → SourceDocument → TextParser → ParsedDocument")
    print("=" * 70)


if __name__ == "__main__":
    main()