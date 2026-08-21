"""Manual retrieval smoke test against the configured MySQL and Qdrant data.

Run from the repository root::

    .venv/bin/python tests/unit/retrieval/manual_test_retrieval.py \
        "Pourquoi le moteur vibre-t-il ?" --equipment-id 42

This script intentionally uses the public RetrievalOrchestrator API. It does
not create, update or delete data in either backend.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Sequence

from app.exceptions import GMAOError
from app.models.retrieval import RetrievalFilter
from app.retrieval import build_default_orchestrator


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the query and optional GMAO filters supplied on the command line."""
    parser = argparse.ArgumentParser(description="Run a read-only GMAO-RAG retrieval query.")
    parser.add_argument("query", help="Natural-language question to search for.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of results (default: 5).")
    parser.add_argument("--equipment-id", type=int, dest="id_equipement")
    parser.add_argument("--document-id", type=int, dest="id_document")
    parser.add_argument("--panne-id", type=int, dest="id_panne")
    parser.add_argument("--source-type", help="Optional source type, for example pdf or txt.")
    parser.add_argument("--hybrid", action="store_true", help="Use the hybrid Qdrant + MySQL strategy.")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute one query and print an inspectable, read-only report."""
    options = parse_arguments(arguments)
    filters = RetrievalFilter(
        id_document=options.id_document,
        id_panne=options.id_panne,
        id_equipement=options.id_equipement,
        source_type=options.source_type,
    )

    try:
        report = build_default_orchestrator().retrieve(
            options.query,
            top_k=options.top_k,
            filters=filters,
            strategy_name="hybrid" if options.hybrid else None,
        )
    except GMAOError as error:
        print(f"Retrieval failed: {error}")
        print(error.to_dict())
        return 1

    print(f"Query: {report.query!r}")
    print(f"Strategy: {report.strategy_name}")
    print(f"Candidates before threshold: {report.total_candidates}")
    print(f"Results: {len(report.results)}")

    for result in report.results:
        print("-" * 80)
        print(f"#{result.rank}  score={result.score:.4f}  chunk_id={result.chunk_id}")
        print(
            "source="
            f"{result.source_name!r} ({result.source_type}), "
            f"document={result.id_document}, panne={result.id_panne}, "
            f"equipment={result.id_equipement}"
        )
        print(result.content)
        print(f"metadata={result.metadata}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
