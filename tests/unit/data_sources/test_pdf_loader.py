import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_sources.file.pdf_loader import PDFLoader
from app.exceptions import GMAOError


FILES = [
    "tests/data/pdf/valid.pdf",
    "tests/data/pdf/empty.pdf",
    "tests/data/pdf/missing.pdf",
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

            loader = PDFLoader(Path(filename))

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