"""
Test manuel du MarkdownLoader.

Exécuter :

    python tests/unit/data_sources/test_markdown_loader_manual.py
"""

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from app.data_sources.file.markdown_loader import MarkdownLoader
from app.exceptions import GMAOError


TEST_FILES = [
    "valid.md",
    "unicode.md",
    "complex.md",
    "empty.md",
    "missing.md",
]


def test_file(filename: str) -> None:

    path = Path("tests/data/markdown") / filename

    print("=" * 80)
    print(f"Testing : {path}")
    print("=" * 80)

    try:

        loader = MarkdownLoader(path)

        document = loader.load()

        print("SUCCESS")

        print(f"Name      : {document.source_name}")
        print(f"Type      : {document.source_type}")
        print(f"Path      : {document.source_path}")
        print(f"MIME      : {document.mime_type}")
        print(f"Size      : {document.size}")
        print(f"Extension : {document.extension}")
        print(f"Length    : {document.content_length}")
        print(f"Empty     : {document.is_empty}")

        print("\nMetadata")
        print("-" * 40)

        for key, value in document.metadata.items():
            print(f"{key:<20}: {value}")

        print("\nContent")
        print("-" * 40)

        print(document.content)

    except GMAOError as exc:

        print("GMAO ERROR")
        print(type(exc).__name__)
        print(exc)

    except Exception as exc:

        print("UNEXPECTED ERROR")
        print(type(exc).__name__)
        print(exc)

    print()


def main():

    for file in TEST_FILES:
        test_file(file)


if __name__ == "__main__":
    main()