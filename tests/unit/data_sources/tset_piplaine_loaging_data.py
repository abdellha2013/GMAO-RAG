"""
Test manuel du DataSourceOrchestrator.

Ce script prend en entrée :

    1. le type de source  : "file", "database" ou "api"
    2. la donnée associée :
        - "file"     -> chemin du fichier (str)
        - "database" -> configuration JSON
                         (doit contenir au minimum "driver")
        - "api"      -> n'importe quelle valeur (non implémenté,
                         sert à vérifier que l'erreur est propre)

Il exécute le VRAI point d'entrée unique du projet
(``DataSourceOrchestrator.load``) et affiche le ``SourceDocument``
résultant, prêt à être "envoyé" à l'étage suivant du pipeline
(Parser). La sérialisation JSON en fin de sortie montre exactement
la forme du payload transmis.

Usage
-----
    # Fichier
    python tests/integration/test_data_source_manual.py file tests/data/txt/multiline.txt

    # Base de données
    python tests/integration/test_data_source_manual.py database '{
        "driver": "mysql",
        "host": "localhost",
        "database": "gmao",
        "user": "root",
        "password": "secret",
        "table": "interventions",
        "max_rows": 100
    }'

    # API (non implémenté -> erreur volontaire et propre)
    python tests/integration/test_data_source_manual.py api "irrelevant"

Sans argument, le script exécute un jeu d'exemples par défaut
(un fichier valide, une DB avec un driver inconnu, une API).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import os 
print(f"the currect working directory is : {os.getcwd()}")
from app.data_sources import DataSourceOrchestrator
from app.exceptions import GMAOError
from app.models.document import SourceDocument

# ==========================================================
# Cas par défaut (si aucun argument n'est fourni)
# ==========================================================

DEFAULT_CASES: list[tuple[str, str]] = [
    ("file", "tests/data/txt/multiline.txt"),
    ("file", "tests/data/json/valid_object.json"),
    (
        "database",
        json.dumps(
            {
                "driver": "mysql",  # driver volontairement non supporté
                "host": "127.0.0.1",
                "database": "gmao_rag_test",
                "user": "root",
                "password": "Zzdv6401",
                "table": "equipements",
            }
        ),
    ),
    ("api", "irrelevant"),
]


# ==========================================================
# Parsing de la donnée d'entrée selon le type de source
# ==========================================================


def parse_data(kind: str, raw_data: str) -> str | dict[str, Any]:
    """
    Convertit l'argument brut en la valeur attendue par
    ``DataSourceOrchestrator.load``.

    - "file" -> chemin brut (str), interprété par le loader.
    - "database" -> dictionnaire de configuration (JSON attendu).
    - "api" -> valeur brute, transmise telle quelle.
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
                "Pour 'database', le JSON fourni doit représenter un objet "
                "(dictionnaire de configuration)."
            )

        return parsed

    return raw_data


# ==========================================================
# Envoi / affichage du SourceDocument
# ==========================================================


def send_source_document(document: SourceDocument) -> None:
    """
    Simule l'envoi du SourceDocument à l'étage suivant du pipeline.

    En attendant un vrai consommateur (ex: ParserOrchestrator), on se
    contente d'afficher le document et sa forme sérialisée, qui est
    exactement le payload qu'un envoi réel transmettrait.
    """
    

    print("SourceDocument")
    print("-" * 40)
    print(f"source_name : {document.source_name}")
    print(f"source_type : {document.source_type}")
    print(f"mime_type   : {document.mime_type}")
    print(f"size        : {document.size} bytes")
    print(f"is_empty    : {document.is_empty}")
    print(f"content_len : {document.content_length}")

    print("\nPayload envoyé (JSON)")
    print("-" * 40)
    print(json.dumps(document.to_dict(), indent=2, ensure_ascii=False, default=str))
    print("the content is :")
    print("-" * 40)
    print(document.content)


# ==========================================================
# Exécution pour un cas
# ==========================================================


def run_case(kind: str, raw_data: str, orchestrator: DataSourceOrchestrator) -> None:
    print("=" * 80)
    print(f"Type : {kind} | Donnée : {raw_data}")
    print("=" * 80)

    try:
        data = parse_data(kind, raw_data)
        document = orchestrator.load(data, kind=kind)

        print("SUCCÈS\n")
        send_source_document(document)

    except GMAOError as exc:
        print("ERREUR GMAO")
        print(f"{type(exc).__name__}: {exc}")

    except ValueError as exc:
        print("ERREUR D'ENTRÉE")
        print(exc)

    except Exception as exc:  # garde-fou : ne doit normalement jamais arriver
        print("ERREUR INATTENDUE")
        print(f"{type(exc).__name__}: {exc}")

    print()


# ==========================================================
# Entrée principale
# ==========================================================


def main() -> None:
    orchestrator = DataSourceOrchestrator()

    if len(sys.argv) >= 3:
        kind, raw_data = sys.argv[1], sys.argv[2]
        run_case(kind, raw_data, orchestrator) 
        return

    if len(sys.argv) == 2:
        print("Usage : python test_data_source_manual.py <file|database|api> <data>")
        sys.exit(1)

    for kind, raw_data in DEFAULT_CASES:
        run_case(kind, raw_data, orchestrator)


if __name__ == "__main__":
    main()