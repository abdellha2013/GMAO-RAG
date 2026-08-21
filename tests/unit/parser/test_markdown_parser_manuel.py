"""Manual MarkdownParser tester using real Markdown files.

Exécuter :

    python tests/unit/parser/test_markdown_parser_manuel.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.exceptions import GMAOError
from app.models.document import SourceDocument
from app.parser.strategies.markdown import MarkdownParser

TEST_FILES = [
    "valid.md",
    "unicode.md",
    "complex.md",
    "empty.md",
]

DATA_DIR = Path("tests") / "data" / "markdown"


def print_parsed_document(parsed_document) -> None:
    print(f"Name       : {parsed_document.source_name}")
    print(f"Type       : {parsed_document.source_type}")
    print(f"MIME       : {parsed_document.mime_type}")
    print(f"Size       : {parsed_document.size}")
    print(f"Created    : {parsed_document.created_at}")
    print(f"Modified   : {parsed_document.updated_at}")
    print(f"Parsed at  : {parsed_document.parsed_at}")
    print(f"Is Empty   : {parsed_document.is_empty}")
    print(f"Length     : {parsed_document.content_length}")
    print("\nMetadata")
    print("-" * 40)
    for key, value in parsed_document.metadata.items():
        print(f"{key:<20}: {value}")
    print("\nContent")
    print("-" * 40)
    print(parsed_document.content)


def run_file(filename: str) -> None:
    path = DATA_DIR / filename

    print("=" * 80)
    print(f"Testing : {path}")
    print("=" * 80)

    try:
        content = path.read_text(encoding="utf-8")
        now = datetime.now(timezone.utc)
        document = SourceDocument(
            source_name=filename,
            source_type="markdown",
            source_path=path,
            content=content,
            mime_type="text/markdown",
            size=len(content.encode("utf-8")),
            created_at=now,
            updated_at=now,
            metadata={"source_path": str(path)},
        )

        parser = MarkdownParser()
        print(f"Supports   : {parser.supports(document)}")

        parsed = parser.parse(document)
        print("SUCCESS")
        print_parsed_document(parsed)

    except GMAOError as exc:
        print("GMAO ERROR")
        print(type(exc).__name__)
        print(exc)

    except Exception as exc:
        print("UNEXPECTED ERROR")
        print(type(exc).__name__)
        print(exc)

    print()


def main() -> None:
    for filename in TEST_FILES:
        run_file(filename)


if __name__ == "__main__":
    main()
