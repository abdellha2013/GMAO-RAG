"""Manual end-to-end test: MySQL panne record -> storage backends.

Prerequisites in ``.env``:

    MYSQL_DSN, or GMAO_DB_HOST, GMAO_DB_PORT, GMAO_DB_NAME,
    GMAO_DB_USER, GMAO_DB_PASSWORD
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME

Run with:

    .venv/bin/python tests/unit/storage/manual_test_database_to_storage.py

Each ``panne`` row is loaded separately through the MySQL data-source loader.
Its chunks are associated with its own ``panne.id_panne`` in ``panne_chunk``.
Set ``STORAGE_TEST_KEEP_DATA=1`` to keep the chunks after the test.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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


def _env_value(name: str) -> str:
    """Return a required, cleaned environment variable."""
    value = os.getenv(name, "").split("#", maxsplit=1)[0].strip()
    if not value:
        raise RuntimeError(f"{name} est requis dans .env.")
    return value


def _mysql_dsn() -> str:
    """Build the DSN using the same environment settings as MySQL storage."""
    dsn = os.getenv("MYSQL_DSN", "").strip()
    if dsn:
        return dsn
    return (
        f"mysql+pymysql://{_env_value('GMAO_DB_USER')}:"
        f"{os.getenv('GMAO_DB_PASSWORD', '')}@{_env_value('GMAO_DB_HOST')}:"
        f"{_env_value('GMAO_DB_PORT')}/{_env_value('GMAO_DB_NAME')}"
    )


def panne_ids() -> tuple[int, ...]:
    """Return every panne identifier before processing rows one by one."""
    engine = create_engine(_mysql_dsn())
    try:
        with engine.connect() as connection:
            return tuple(
                int(row.id_panne)
                for row in connection.execute(text("SELECT id_panne FROM panne ORDER BY id_panne"))
            )
    finally:
        engine.dispose()

def main() -> None:
    all_panne_ids = panne_ids()
    created_ids: list[int] = []

    database_config = {
        "driver": "mysql",
        "host": _env_value("GMAO_DB_HOST"),
        "port": int(_env_value("GMAO_DB_PORT")),
        "database": _env_value("GMAO_DB_NAME"),
        "user": _env_value("GMAO_DB_USER"),
        "password": os.getenv("GMAO_DB_PASSWORD", ""),
    }

    try:
        if not all_panne_ids:
            print("Aucune panne à indexer.")
            return

        loader = DataSourceOrchestrator()
        parser = build_parser()
        chunker = build_chunker(chunk_size=250, chunk_overlap=30)
        embedder = build_embedder(batch_size=8, device="auto")
        storage = build_storage(raise_on_partial_failure=False)

        for panne_id in all_panne_ids:
            database_config["query"] = (
                "SELECT id_panne, titre, description, gravite, date_detection, "
                "cause, solution, symptomes, statut_indexation, id_equipement, id_ot "
                "FROM panne WHERE id_panne = :panne_id"
            )
            database_config["params"] = {"panne_id": panne_id}
            source = loader.load(database_config)
            parsed = parser.parse(source)
            chunks = chunker.chunk(parsed)
            for chunk in chunks:
                chunk.metadata["id_panne"] = panne_id

            embeddings = embedder.embed(chunks)
            report = storage.save(chunks, embeddings)
            new_ids = [
                chunk.metadata["id_chunk"]
                for chunk in chunks
                if isinstance(chunk.metadata.get("id_chunk"), int)
            ]
            created_ids.extend(new_ids)

            if report.has_failures:
                raise PartialStorageError(
                    details={
                        "panne_id": panne_id,
                        "failures": list(report.failures),
                        "succeeded_strategies": [
                            outcome.strategy_name for outcome in report.outcomes if outcome.success
                        ],
                    }
                )
            print(f"Panne MySQL indexée : {panne_id} ({len(new_ids)} chunk(s))")

        print("Pipeline réussi.")
        print(f"Chunks stockés : {created_ids}")
    except GMAOError as exc:
        print(f"Erreur métier : {exc}")
        print(f"Détails : {exc.to_dict()}")
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        print(f"Configuration invalide : {exc}")
        raise SystemExit(2) from exc
    finally:
        if created_ids and os.getenv("STORAGE_TEST_KEEP_DATA") != "1":
            try:
                build_storage(raise_on_partial_failure=False).delete(created_ids)
                print(f"Nettoyage réussi : {created_ids}")
            except GMAOError as exc:
                print(f"Échec du nettoyage : {exc.to_dict()}")


if __name__ == "__main__":
    main()
