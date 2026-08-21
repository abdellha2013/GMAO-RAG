"""
Test manuel du TXTLoader.

Exécuter :

    python tests/unit/data_sources/test_txt_loader_manual.py
    
"""

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data_sources.file.txt_loader import TXTLoader
from app.exceptions import GMAOError

TEST_FILES = [
    "valid_utf8.txt",
    "unicode.txt",
    "multiline.txt",
    "empty.txt",
    "missing.txt",          # N'existe pas
]


def test_file(file_name: str) -> None:
    """
    Charge un fichier et affiche toutes ses informations.
    """

    path = Path("tests/data/txt") / file_name

    print("=" * 80)
    print(f"Testing : {path}")
    print("=" * 80)

    try:

        loader = TXTLoader(path)

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


try :
    from app.exceptions import GMAOError
    if __name__ == "__main__":
        main()
except ImportError as exc:
    print(f"Erreur d'importation : {exc}")
    print("Assurez-vous que le projet est correctement installé et que les dépendances sont satisfaites.")