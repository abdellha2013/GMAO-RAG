import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_sources.file.xlsx_loader import XLSXLoader
from app.exceptions import GMAOError


FILES = [
    "tests/data/xlsx/valid.xlsx",
    "tests/data/xlsx/multiple_sheets.xlsx",
    "tests/data/xlsx/unicode.xlsx",
    "tests/data/xlsx/empty_content.xlsx",
    "tests/data/xlsx/empty.xlsx",
    "tests/data/xlsx/corrupted.xlsx",
    "tests/data/xlsx/file_example_XLSX_50.xlsx",
]


def print_document(doc):

    print(f"Name       : {doc.source_name}")
    print(f"Type       : {doc.source_type}")
    print(f"Path       : {doc.source_path}")
    print(f"MIME       : {doc.mime_type}")
    print(f"Size       : {doc.size} bytes")
    print(f"Created    : {doc.created_at}")
    print(f"Modified   : {doc.updated_at}")
    print(f"Is Empty   : {doc.is_empty}")
    print(f"Length     : {doc.content_length}")
    print(f"Extension  : {doc.extension}")

    print("\nMetadata")
    print("-" * 40)

    for key, value in doc.metadata.items():
        print(f"{key:20}: {value}")

    print("\nContent")
    print("-" * 40)
    print(doc.content)


def main():

    for filename in FILES:

        print()
        print("=" * 80)
        print("Testing :", filename)
        print("=" * 80)

        try:

            loader = XLSXLoader(Path(filename))

            document = loader.load()

            print("SUCCESS")

            print_document(document)

        except GMAOError as exc:

            print("GMAO ERROR")
            print(type(exc).__name__)
            print(exc)

        except Exception as exc:

            print("UNEXPECTED ERROR")
            print(type(exc).__name__)
            print(exc)


if __name__ == "__main__":
    main()    