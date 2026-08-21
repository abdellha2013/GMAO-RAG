
"""
tests/test_html_loader_manual.py
================================

Test manuel du HTMLLoader.

Ce script teste :

- HTML valide
- HTML Unicode
- HTML complexe
- HTML vide
- HTML inexistant

Et affiche :

- nom
- type
- chemin
- MIME
- taille
- dates
- contenu
- métadonnées
- propriétés du Document
"""

from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.data_sources.file.html_loader import HTMLLoader
from app.exceptions import GMAOError


# ==========================================================
# Configuration
# ==========================================================




FILES = [
    "tests/data/html/valid.html",
    "tests/data/html/unicode.html",
    "tests/data/html/complex.html",
    "tests/data/html/empty.html",
    "tests/data/html/missing.html",
]

# ==========================================================
# Display helpers
# ==========================================================

def print_separator() -> None:
    print("=" * 80)


def print_metadata(metadata: dict) -> None:
    print()
    print("Metadata")
    print("-" * 40)

    for key, value in metadata.items():
        print(f"{key:<20} : {value}")


def print_document(document) -> None:
    print()
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

    print_metadata(document.metadata)

    print()
    print("Content")
    print("-" * 40)

    print(document.content)


# ==========================================================
# Test one file
# ==========================================================

def test_file(path: Path) -> None:
    print()
    print_separator()
    print(f"Testing : {path}")
    print_separator()

    try:
        loader = HTMLLoader(path)

        document = loader.load()

        print_document(document)

    except GMAOError as exc:
        print("GMAO ERROR")
        print(type(exc).__name__)
        print(str(exc))

    except Exception as exc:
        print("UNEXPECTED ERROR")
        print(type(exc).__name__)
        print(str(exc))


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    print_separator()
    print("HTML LOADER MANUAL TEST")
    print_separator()

    for file_path in FILES:
        test_file(file_path)

    print()
    print_separator()
    print("TESTING FINISHED")
    print_separator()


if __name__ == "__main__":
    main()
