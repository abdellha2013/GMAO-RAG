"""Tests for requirement #3: Qdrant/MySQL responsibility separation.

QdrantStorage must never write to MySQL itself. Propagating a successful
Qdrant write into ``chunk_rag.statut_embedding`` is the orchestrator's job,
via ``MySQLStorage.mark_indexed()``.
"""
from __future__ import annotations

from app.models.chunk import Chunk
from app.models.embedding import Embedding
from app.storage.base import StorageOutcome, StorageStrategy
from app.storage.orchestrator import StorageOrchestrator
from app.storage.registry import StorageRegistry


class _FakeMySQL(StorageStrategy):
    name = "mysql"
    mark_indexed_calls: list = []

    def __init__(self, **_options):
        pass

    def supports(self, chunks, embeddings):
        return True

    def save(self, chunks, embeddings):
        for chunk in chunks:
            chunk.metadata["id_chunk"] = chunk.chunk_index + 1
        return StorageOutcome(self.name, tuple(c.metadata["id_chunk"] for c in chunks))

    def delete(self, chunk_ids):
        return StorageOutcome(self.name, tuple(chunk_ids))

    def mark_indexed(self, chunk_ids):
        _FakeMySQL.mark_indexed_calls.append(tuple(chunk_ids))


class _FakeQdrant(StorageStrategy):
    name = "qdrant"

    def __init__(self, **_options):
        pass

    def supports(self, chunks, embeddings):
        return True

    def save(self, chunks, embeddings):
        return StorageOutcome(self.name, tuple(c.metadata["id_chunk"] for c in chunks))

    def delete(self, chunk_ids):
        return StorageOutcome(self.name, tuple(chunk_ids))


def test_mark_indexed_called_after_successful_qdrant_save():
    _FakeMySQL.mark_indexed_calls = []
    registry = StorageRegistry()
    registry.register(_FakeMySQL)
    registry.register(_FakeQdrant)

    orchestrator = StorageOrchestrator(registry, strategy_sequence=("mysql", "qdrant"))
    chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt")
    embedding = Embedding(chunk_id="doc.txt:0", vector=(0.1,), model_name="test", dimension=1)

    report = orchestrator.save([chunk], [embedding])

    assert report.is_full_success
    assert _FakeMySQL.mark_indexed_calls == [(1,)]


def test_qdrant_storage_module_has_no_sqlalchemy_dependency():
    """QdrantStorage must not import sqlalchemy or know about the MySQL schema."""
    import ast
    import inspect

    from app.storage.strategies import qdrant_storage

    source = inspect.getsource(qdrant_storage)
    tree = ast.parse(source)
    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "sqlalchemy" not in imported_modules
    assert "chunk_rag" not in source
