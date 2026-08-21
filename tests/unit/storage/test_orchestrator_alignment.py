"""Tests for the critical chunk/embedding alignment fix.

Covers requirement #1 (PROMPT_CORRECTION_STORAGE.md): the alignment check
in ``StorageOrchestrator.save()`` must not compare the raw
``chunk.chunk_id`` field, since ``RecursiveChunker`` always leaves it at
``None``. It must compare the same logical identifier that
``Embedding.chunk_id`` already falls back to.
"""
from __future__ import annotations

import pytest

from app.exceptions import StorageAlignmentError
from app.models.chunk import Chunk
from app.models.embedding import Embedding
from app.storage.base import StorageStrategy
from app.storage.orchestrator import StorageOrchestrator
from app.storage.registry import StorageRegistry


class _RecordingStrategy(StorageStrategy):
    """Minimal fake strategy: records the batches it was asked to save."""

    name = "fake"

    def __init__(self, **_options):
        self.calls = []

    def save(self, chunks, embeddings):
        from app.storage.base import StorageOutcome

        self.calls.append((chunks, embeddings))
        return StorageOutcome(self.name, tuple(range(len(chunks))))

    def delete(self, chunk_ids):
        from app.storage.base import StorageOutcome

        return StorageOutcome(self.name, tuple(chunk_ids))

    def supports(self, chunks, embeddings):
        return True


def _orchestrator():
    registry = StorageRegistry()
    registry.register(_RecordingStrategy)
    return StorageOrchestrator(registry, strategy_sequence=("fake",))


def test_alignment_ok_for_recursive_chunker_batch():
    """RecursiveChunker output (chunk_id=None) must align with its embedding."""
    chunk = Chunk(content="hello", chunk_index=0, source_name="doc.txt", source_type="txt", chunk_id=None)
    embedding = Embedding(chunk_id="doc.txt:0", vector=(0.1, 0.2), model_name="test", dimension=2)
    assert embedding.chunk_id == "doc.txt:0"

    orchestrator = _orchestrator()
    report = orchestrator.save([chunk], [embedding])
    assert report.is_full_success


def test_alignment_ok_for_mixed_batch():
    """A batch mixing RecursiveChunker (chunk_id=None) and MarkdownChunker
    (chunk_id explicitly set) chunks must both align correctly."""
    recursive_chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt", chunk_id=None)
    markdown_chunk = Chunk(content="b", chunk_index=1, source_name="doc.md", source_type="markdown", chunk_id="doc.md#section-1")

    recursive_embedding = Embedding(chunk_id="doc.txt:0", vector=(0.1,), model_name="test", dimension=1)
    markdown_embedding = Embedding(
        chunk_id="doc.md#section-1", vector=(0.2,), model_name="test", dimension=1
    )

    orchestrator = _orchestrator()
    report = orchestrator.save(
        [recursive_chunk, markdown_chunk],
        [recursive_embedding, markdown_embedding],
    )
    assert report.is_full_success


def test_alignment_fails_on_real_mismatch():
    """A genuine mismatch (wrong embedding for a chunk) must still raise."""
    chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt", chunk_id=None)
    wrong_embedding = Embedding(chunk_id="other.txt:9", vector=(0.1,), model_name="test", dimension=1)

    orchestrator = _orchestrator()
    with pytest.raises(StorageAlignmentError):
        orchestrator.save([chunk], [wrong_embedding])


def test_alignment_fails_on_length_mismatch():
    chunk = Chunk(content="a", chunk_index=0, source_name="doc.txt", source_type="txt")
    orchestrator = _orchestrator()
    with pytest.raises(StorageAlignmentError):
        orchestrator.save([chunk], [])
