"""Manual TextParser tester using real text files.

Exécuter :

    python tests/unit/parser/test_text_manuel.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.exceptions import GMAOError
from app.models.document import SourceDocument
from app.parser.strategies.text import TextParser

TEST_FILES = [
    "alid_utf8.txt",
    "empty.txt",
    "multiline.txt",
    "unicode.txt",
]

DATA_DIR = Path("tests") / "data" / "txt"


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
            source_type="txt",
            content=content,
            mime_type="text/plain",
            size=len(content.encode("utf-8")),
            created_at=now,
            updated_at=now,
            metadata={"source_path": str(path)},
        )

        parser = TextParser()
        print(f"Supports   : {parser.supports(document)}")

        print(parser.parse(document).content)

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
