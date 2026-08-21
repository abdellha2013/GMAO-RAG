"""
tests/integration/test_pipeline_load_to_chunk.py
==================================================

Test manuel du pipeline complet, de la source brute jusqu'au chunking :

    DataSourceOrchestrator.load()   ->  SourceDocument
    ParserOrchestrator.parse()      ->  ParsedDocument
    ChunkerOrchestrator.chunk()     ->  list[Chunk]

Ce script est le prolongement de
``tests/unit/data_sources/tset_piplaine_loaging_data.py`` (qui ne
teste que la couche Data Source) : il enchaîne les TROIS étages
réellement utilisés par le projet, avec les modèles et orchestrateurs
existants (aucune nouvelle logique métier n'est introduite ici).

Sources supportées
-------------------
    1. "file"      -> chemin d'un fichier (str). Formats couverts par
                       les stratégies existantes : txt, md, html,
                       json, csv, xlsx, docx, pdf.
    2. "database"   -> configuration JSON (doit contenir "driver").
                       Seul "mysql" est actuellement implémenté.
    3. "api"        -> volontairement NON implémenté côté
                       DataSourceOrchestrator. Le test vérifie que
                       l'erreur levée (UnsupportedSourceError) est
                       propre et n'interrompt pas le reste du script.

Usage
-----
    # Fichier
    python tests/integration/test_pipeline_load_to_chunk.py file tests/data/markdown/valid.md

    # Base de données
    python tests/integration/test_pipeline_load_to_chunk.py database '{
        "driver": "mysql",
        "host": "localhost",
        "database": "gmao",
        "user": "root",
        "password": "secret",
        "table": "interventions"
    }'

    # API (non implémenté -> erreur volontaire et propre)
    python tests/integration/test_pipeline_load_to_chunk.py api "irrelevant"

Sans argument, le script exécute un jeu de cas par défaut couvrant
les trois types de source et plusieurs formats de fichier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ==========================================================
# Project root
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==========================================================
# Environment (.env optionnel, utilisé pour la config MySQL)
# ==========================================================

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    # python-dotenv est optionnel pour ce script : sans lui, les
    # variables d'environnement doivent simplement déjà être définies
    # (ou le cas "database" par défaut échouera proprement).
    pass

# ==========================================================
# Application imports
# ==========================================================

from app.data_sources import DataSourceOrchestrator
from app.exceptions import GMAOError
from app.models.chunk import Chunk
from app.models.document import SourceDocument
from app.models.parsing import ParsedDocument

from app.parser import (
    DatabaseParser,
    HTMLParser,
    MarkdownParser,
    ParserOrchestrator,
    ParserRegistry,
    StructuredParser,
    TextParser,
)

from app.chunker import build_default_orchestrator


# ==========================================================
# Construction des orchestrateurs (modèles déjà existants,
# aucune stratégie n'est réimplémentée ici)
# ==========================================================


def build_parser_orchestrator() -> ParserOrchestrator:
    """
    Construit un ParserOrchestrator avec toutes les stratégies de
    parsing connues du projet déjà enregistrées.

    Il n'existe pas encore de ``build_default_registry`` côté
    ``app.parser`` (contrairement à ``app.chunker``) : on reproduit
    donc ici l'enregistrement, à l'identique de ce que ferait une
    telle fonction utilitaire.
    """
    registry = ParserRegistry()

    for strategy in (
        TextParser,
        MarkdownParser,
        HTMLParser,
        StructuredParser,
        DatabaseParser,
    ):
        registry.register(strategy)

    return ParserOrchestrator(registry=registry)


DATA_SOURCE_ORCHESTRATOR = DataSourceOrchestrator()
PARSER_ORCHESTRATOR = build_parser_orchestrator()
CHUNKER_ORCHESTRATOR = build_default_orchestrator(chunk_size=500, chunk_overlap=50)


# ==========================================================
# Cas par défaut (si aucun argument n'est fourni)
# ==========================================================

DEFAULT_CASES: list[tuple[str, str]] = [
    # --- file : un représentant par famille de stratégie ---
    # ("file", "tests/data/txt/multiline.txt"),          # TextParser + RecursiveChunker
    # ("file", "tests/data/markdown/valid.md"),           # MarkdownParser + MarkdownChunker
    # ("file", "tests/data/html/valid.html"),              # HTMLParser + RecursiveChunker
    # ("file", "tests/data/json/valid_object.json"),       # StructuredParser + StructuredChunker
    # ("file", "tests/data/csv/valid_comma.csv"),          # StructuredParser + StructuredChunker
    # --- database : driver mysql (nécessite un serveur accessible) ---
    (
        "database",
        json.dumps(
            {
                "driver": "mysql",
                "host": "127.0.0.1",
                "database": "gmao_rag_test",
                "user": "root",
                "password": "Zzdv6401",
                "table": "equipements",
            }
        ),
    ),
    # --- api : volontairement non implémenté ---
    # ("api", "irrelevant"),
]


# ==========================================================
# Parsing de l'argument brut selon le type de source
# ==========================================================


def parse_data(kind: str, raw_data: str) -> str | dict[str, Any]:
    """
    Convertit l'argument brut en la valeur attendue par
    ``DataSourceOrchestrator.load``.
    """
    if kind == "database":
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Pour 'database', la donnée doit être un JSON valide, "
                f"ex: '{{\"driver\": \"mysql\", \"host\": \"localhost\", ...}}'. "
                f"Erreur : {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Pour 'database', le JSON fourni doit représenter un "
                "objet (dictionnaire de configuration)."
            )

        return parsed

    return raw_data


# ==========================================================
# Affichage de chaque étage du pipeline
# ==========================================================


def print_source_document(document: SourceDocument) -> None:
    print("\n[1/3] SourceDocument (Data Source Layer)")
    print("-" * 60)
    print(f"source_name : {document.source_name}")
    print(f"source_type : {document.source_type}")
    print(f"mime_type   : {document.mime_type}")
    print(f"size        : {document.size} bytes")
    print(f"is_empty    : {document.is_empty}")
    print(f"content_len : {document.content_length}")


def print_parsed_document(document: ParsedDocument) -> None:
    print("\n[2/3] ParsedDocument (Parser Layer)")
    print("-" * 60)
    print(f"source_name : {document.source_name}")
    print(f"source_type : {document.source_type}")
    print(f"content_len : {len(document.content)}")
    # preview = document.content.strip().replace("\n", " ")[:200]
    #the all data with the same forme 
    preview = document.content
    print(f"preview     : \n{preview}")


def print_chunks(chunks: list[Chunk]) -> None:
    print("\n[3/3] Chunks (Chunker Layer)")
    print("-" * 60)
    print(f"total_chunks : {len(chunks)}")

    for chunk in chunks:
        preview = chunk.content
        print(
            f"  - #{chunk.chunk_index} \n"
            f"(len={len(chunk.content)}, \n"
            f"total={chunk.total_chunks}) \n"
            f"preview :\n{preview}"
        )

    # if len(chunks) > 3:
    #     print(f"  ... ({len(chunks) - 3} chunk(s) supplémentaire(s))")


# ==========================================================
# Exécution d'un cas : load -> parse -> chunk
# ==========================================================


def run_pipeline(kind: str, raw_data: str) -> None:
    print("=" * 80)
    print(f"Type : {kind} | Donnée : {raw_data}")
    print("=" * 80)

    try:
        data = parse_data(kind, raw_data)

        # ---- 1. Data Source Layer ----
        source_document = DATA_SOURCE_ORCHESTRATOR.load(data, kind=kind)
        print_source_document(source_document)

        # ---- 2. Parser Layer ----
        parsed_document = PARSER_ORCHESTRATOR.parse(source_document)
        print_parsed_document(parsed_document)

        # ---- 3. Chunker Layer ----
        chunks = CHUNKER_ORCHESTRATOR.chunk(parsed_document)
        print_chunks(chunks)

        print("\nSUCCÈS : pipeline complet exécuté jusqu'au chunking.")

    except GMAOError as exc:
        # Cas attendu pour "api" (UnsupportedSourceError) et pour
        # toute erreur métier propre levée par un étage du pipeline.
        print(f"\nERREUR GMAO ({type(exc).__name__}) : {exc}")

    except ValueError as exc:
        print(f"\nERREUR D'ENTRÉE : {exc}")

    except Exception as exc:  # garde-fou : ne doit normalement jamais arriver
        print(f"\nERREUR INATTENDUE ({type(exc).__name__}) : {exc}")

    print()


# ==========================================================
# Entrée principale
# ==========================================================


def guess_kind(raw_data: str) -> str:
    """
    Déduit le ``kind`` quand un seul argument est fourni en ligne de
    commande, avec la même logique que
    ``DataSourceOrchestrator._resolve_kind`` :

    - littéral "api"                          -> "api"
    - JSON représentant un objet avec "driver" -> "database"
    - tout le reste (chemin de fichier, etc.)  -> "file"
    """
    if raw_data.strip().lower() == "api":
        return "api"

    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError:
        return "file"

    if isinstance(parsed, dict) and "driver" in parsed:
        return "database"

    return "file"


def main() -> None:
    # 2 arguments explicites : <kind> <data>
    # if len(sys.argv) >= 3:
    #     kind, raw_data = sys.argv[1], sys.argv[2]
    #     run_pipeline(kind, raw_data)
    #     return

    # # 1 argument : kind déduit automatiquement (fichier, JSON avec
    # # "driver", ou littéral "api")
    # if len(sys.argv) == 2:
    #     raw_data = sys.argv[1]
    #     kind = guess_kind(raw_data)
    #     run_pipeline(kind, raw_data)
    #     return

    # aucun argument : jeu de cas par défaut
    for kind, raw_data in DEFAULT_CASES:
        run_pipeline(kind, raw_data)

try :
    from app.exceptions import GMAOError

    if __name__ == "__main__":
        main()
except ImportError as exc:
    print(f"Erreur d'importation : {exc}")
    print("Assurez-vous que le projet est correctement installé et que les dépendances sont satisfaites.")