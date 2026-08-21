"""Tests for StorageOrchestrator.delete(), failure handling and PartialStorageError.

Covers requirements #2, #5 (exception detail preservation) and #6
(PartialStorageError raised at the right place, report still exploitable).
"""
from __future__ import annotations

import pytest

from app.exceptions import PartialStorageError, StorageWriteError
from app.models.chunk import Chunk
from app.models.embedding import Embedding
from app.storage.base import StorageOutcome, StorageStrategy
from app.storage.orchestrator import StorageOrchestrator
from app.storage.registry import StorageRegistry


class _OkStrategy(StorageStrategy):
    name = "ok"

    def __init__(self, **_options):
        pass

    def save(self, chunks, embeddings):
        return StorageOutcome(self.name, tuple(range(len(chunks))))

    def delete(self, chunk_ids):
        return StorageOutcome(self.name, tuple(chunk_ids))

    def supports(self, chunks, embeddings):
        return True


class _FailingStrategy(StorageStrategy):
    name = "failing"

    def __init__(self, **_options):
        pass

    def save(self, chunks, embeddings):
        raise StorageWriteError(message="boom", details={"reason": "disk full"})

    def delete(self, chunk_ids):
        raise StorageWriteError(message="boom-delete", details={"reason": "locked"})

    def supports(self, chunks, embeddings):
        return True


def _registry():
    registry = StorageRegistry()
    registry.register(_OkStrategy)
    registry.register(_FailingStrategy)
    return registry


def test_delete_returns_report_for_each_strategy():
    orchestrator = StorageOrchestrator(
        _registry(),
        strategy_sequence=("ok",),
        raise_on_partial_failure=True,
    )
    report = orchestrator.delete([1, 2, 3])
    assert report.is_full_success
    assert report.outcomes[0].saved_ids == (1, 2, 3)


def test_delete_stop_on_failure_true_raises_original_error():
    orchestrator = StorageOrchestrator(
        _registry(),
        strategy_sequence=("failing", "ok"),
        stop_on_failure=True,
    )
    with pytest.raises(StorageWriteError):
        orchestrator.delete([1])


def test_delete_stop_on_failure_false_collects_failure_and_continues():
    orchestrator = StorageOrchestrator(
        _registry(),
        strategy_sequence=("failing", "ok"),
        stop_on_failure=False,
        raise_on_partial_failure=False,
    )
    report = orchestrator.delete([1])
    assert report.has_failures
    assert len(report.outcomes) == 2  # both strategies ran, "ok" was not skipped
    assert report.outcomes[1].success


def test_failure_details_are_preserved_not_replaced_by_generic_message():
    orchestrator = StorageOrchestrator(
        _registry(),
        strategy_sequence=("failing",),
        stop_on_failure=False,
        raise_on_partial_failure=False,
    )
    chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt")
    embedding = Embedding(chunk_id="doc.txt:0", vector=(0.1,), model_name="test", dimension=1)
    report = orchestrator.save([chunk], [embedding])
    failure = report.failures[0]
    assert failure["message"] == "boom"
    assert failure["error_code"] == "STORAGE_WRITE_ERROR"
    assert failure["details"] == {"reason": "disk full"}


def test_partial_storage_error_raised_with_serializable_details():
    orchestrator = StorageOrchestrator(
        _registry(),
        strategy_sequence=("ok", "failing"),
        stop_on_failure=False,
        raise_on_partial_failure=True,
    )
    chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt")
    embedding = Embedding(chunk_id="doc.txt:0", vector=(0.1,), model_name="test", dimension=1)

    with pytest.raises(PartialStorageError) as excinfo:
        orchestrator.save([chunk], [embedding])

    assert excinfo.value.details["succeeded_strategies"] == ["ok"]
    assert excinfo.value.details["failures"][0]["error_code"] == "STORAGE_WRITE_ERROR"
    assert not hasattr(excinfo.value, "report")


def test_no_partial_error_when_disabled():
    orchestrator = StorageOrchestrator(
        _registry(),
        strategy_sequence=("ok", "failing"),
        stop_on_failure=False,
        raise_on_partial_failure=False,
    )
    chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt")
    embedding = Embedding(chunk_id="doc.txt:0", vector=(0.1,), model_name="test", dimension=1)
    report = orchestrator.save([chunk], [embedding])
    assert report.has_failures
