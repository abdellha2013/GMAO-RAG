"""Manual end-to-end test: TXT file -> storage backends.

Prerequisites in ``.env``:

    STORAGE_TEST_DOCUMENT_ID=<existing document.id_document>
    GMAO_DB_HOST, GMAO_DB_PORT, GMAO_DB_NAME, GMAO_DB_USER, GMAO_DB_PASSWORD
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME

Run with:

    .venv/bin/python tests/unit/storage/manual_test_file_to_storage.py

The records created by this script are deleted at the end unless
``STORAGE_TEST_KEEP_DATA=1`` is set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.chunker import build_default_orchestrator as build_chunker
from app.data_sources import DataSourceOrchestrator
from app.embedding import build_default_orchestrator as build_embedder
from app.exceptions import GMAOError, PartialStorageError
from app.parser import build_default_orchestrator as build_parser
from app.storage import build_default_orchestrator as build_storage


def document_id_from_env() -> int:
    value = os.getenv("STORAGE_TEST_DOCUMENT_ID", "").strip()
    if not value:
        raise RuntimeError(
            "STORAGE_TEST_DOCUMENT_ID est requis dans .env et doit référencer "
            "un document existant dans la base configurée."
        )
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError("STORAGE_TEST_DOCUMENT_ID doit être un entier.") from exc


def main() -> None:
    source_path = PROJECT_ROOT / "tests" / "data" / "docx" / "sample-5pages.docx"
    document_id = document_id_from_env()
    created_ids: tuple[int, ...] = ()

    try:
        source = DataSourceOrchestrator().load(source_path)
        parsed = build_parser().parse(source)
        chunks = build_chunker(chunk_size=250, chunk_overlap=30).chunk(parsed)
        for chunk in chunks:
            chunk.metadata["id_document"] = document_id
        embeddings = build_embedder(batch_size=8, device="auto").embed(chunks)
        storage = build_storage(raise_on_partial_failure=False)
        report = storage.save(chunks, embeddings)
        created_ids = tuple(
            chunk.metadata["id_chunk"]
            for chunk in chunks
            if isinstance(chunk.metadata.get("id_chunk"), int)
        )

        if report.has_failures:
            raise PartialStorageError(
                details={
                    "failures": list(report.failures),
                    "succeeded_strategies": [
                        outcome.strategy_name for outcome in report.outcomes if outcome.success
                    ],
                }
            )

        print("Pipeline réussi.")
        print(f"Fichier chargé : {source_path.name}")
        print(f"Chunks stockés : {created_ids}")
    except GMAOError as exc:
        print(f"Erreur métier : {exc}")
        print(f"Détails : {exc.to_dict()}")
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        print(f"Configuration invalide : {exc}")
        raise SystemExit(2) from exc
    # finally:
        # if created_ids and os.getenv("STORAGE_TEST_KEEP_DATA") != "1":
        #     try:
        #         build_storage(raise_on_partial_failure=False).delete(created_ids)
        #         print(f"Nettoyage réussi : {created_ids}")
        #     except GMAOError as exc:
        #         print(f"Échec du nettoyage : {exc.to_dict()}")


try:
    if __name__ == "__main__":
        main()
except ValueError as exc:
    print("the exception is :",exc)
