"""Regression tests for storage partial-failure reporting."""
from __future__ import annotations

import pytest

from app.exceptions import PartialStorageError
from app.models.chunk import Chunk
from app.models.embedding import Embedding
from app.storage.base import StorageOutcome, StorageStrategy
from app.storage.orchestrator import StorageOrchestrator
from app.storage.registry import StorageRegistry


class SuccessfulStorage(StorageStrategy):
    name = "mysql"

    def supports(self, chunks, embeddings): return True
    def save(self, chunks, embeddings): return StorageOutcome(self.name, saved_ids=(1,))
    def delete(self, chunk_ids): return StorageOutcome(self.name, saved_ids=tuple(chunk_ids))


class FailingStorage(StorageStrategy):
    name = "qdrant"

    def supports(self, chunks, embeddings): return True
    def save(self, chunks, embeddings): return StorageOutcome(self.name, failures=({"error_code": "STORAGE_WRITE_ERROR"},))
    def delete(self, chunk_ids): return StorageOutcome(self.name, failures=({"error_code": "STORAGE_WRITE_ERROR"},))


def test_partial_storage_error_uses_serializable_details() -> None:
    registry = StorageRegistry()
    registry.register(SuccessfulStorage)
    registry.register(FailingStorage)
    chunk = Chunk(content="content", chunk_index=0, source_name="source", source_type="txt")
    embedding = Embedding(chunk_id="source:0", vector=(0.1,), model_name="test", dimension=1)

    with pytest.raises(PartialStorageError) as excinfo:
        StorageOrchestrator(registry).save([chunk], [embedding])

    error = excinfo.value
    assert error.details["failures"] == [{"error_code": "STORAGE_WRITE_ERROR"}]
    assert error.details["succeeded_strategies"] == ["mysql"]
    assert not hasattr(error, "report")
