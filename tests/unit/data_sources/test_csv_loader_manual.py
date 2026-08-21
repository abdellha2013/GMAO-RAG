"""
Test manuel du CSVLoader.

Exécuter :

    python tests/unit/data_sources/test_csv_loader_manual.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from app.data_sources.file.csv_loader import CSVLoader
from app.exceptions import GMAOError

TEST_FILES = [
    "valid_comma.csv",
    "valid_semicolon.csv",
    "valid_tab.csv",
    "unicode.csv",
    "headers_only.csv",
    "malformed.csv",
    "empty.csv",
    "missing.csv",
]


def test_file(filename: str) -> None:
    path = Path("tests/data/csv") / filename

    print("=" * 80)
    print(f"Testing : {path}")
    print("=" * 80)

    try:
        loader = CSVLoader(path)
        document = loader.load()

        print("SUCCESS")
        print(f"Name       : {document.source_name}")
        print(f"Type       : {document.source_type}")
        print(f"Path       : {document.source_path}")
        print(f"MIME       : {document.mime_type}")
        print(f"Size       : {document.size} bytes")
        print(f"Created    : {document.created_at}")
        print(f"Modified   : {document.updated_at}")
        print(f"Is Empty   : {document.is_empty}")
        print(f"Length     : {document.content_length}")
        print(f"Extension  : {document.extension}")

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


def main() -> None:
    for file in TEST_FILES:
        test_file(file)


if __name__ == "__main__":
    main()