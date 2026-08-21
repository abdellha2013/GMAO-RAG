"""Unit and opt-in integration tests for the embedding layer."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow ``python tests/unit/embedding/test_embedding.py`` as well as pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding import (
    EmbeddingOrchestrator,
    EmbeddingRegistry,
    SentenceTransformerEmbedding,
    build_default_orchestrator,
)
from app.embedding.base import EmbeddingStrategy
from app.exceptions import (
    EmbeddingModelError,
    EmbeddingStrategyNotRegisteredError,
    EmbeddingValidationError,
)
from app.models.chunk import Chunk
from app.models.embedding import Embedding


class FakeSentenceTransformer:
    """Small in-memory substitute that keeps unit tests offline."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs):
        self.calls.append((texts, kwargs))
        return [[float(index), 0.5, 1.0] for index, _ in enumerate(texts)]


class FakeEmbeddingStrategy(EmbeddingStrategy):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimension(self) -> int:
        return 2

    def supports(self, chunks) -> bool:
        return bool(chunks) and all(isinstance(chunk, Chunk) for chunk in chunks)

    def embed(self, chunks) -> list[Embedding]:
        return [
            Embedding(
                chunk_id=chunk.chunk_id or f"{chunk.source_name}:{chunk.chunk_index}",
                vector=(0.6, 0.8),
                model_name=self.model_name,
                dimension=self.dimension,
            )
            for chunk in chunks
        ]


class EmbeddingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        SentenceTransformerEmbedding.clear_model_cache()
        self.chunk = Chunk(
            chunk_id="manual:0",
            chunk_index=0,
            source_name="manual.txt",
            source_type="txt",
            content="Le moteur présente une vibration anormale.",
            metadata={"equipment": "motor"},
        )

    def _strategy_with_fake_model(self) -> tuple[SentenceTransformerEmbedding, FakeSentenceTransformer]:
        strategy = SentenceTransformerEmbedding(batch_size=2)
        model = FakeSentenceTransformer()
        strategy._model_cache[
            (strategy.model_name, strategy.model_revision, strategy.device)
        ] = model
        return strategy, model

    def test_one_chunk_produces_one_embedding(self) -> None:
        strategy, _ = self._strategy_with_fake_model()

        embeddings = strategy.embed([self.chunk])

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].chunk_id, "manual:0")
        self.assertEqual(embeddings[0].dimension, 3)
        self.assertEqual(embeddings[0].vector, (0.0, 0.5, 1.0))
        self.assertEqual(embeddings[0].metadata["embedding_model"], strategy.model_name)

    def test_multiple_chunks_are_encoded_in_one_batch(self) -> None:
        strategy, model = self._strategy_with_fake_model()
        second = Chunk(
            chunk_index=1,
            source_name="manual.txt",
            source_type="txt",
            content="Le stock de pièces détachées est insuffisant.",
        )

        embeddings = strategy.embed([self.chunk, second])

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0][0],
            [
                "passage: Le moteur présente une vibration anormale.",
                "passage: Le stock de pièces détachées est insuffisant.",
            ],
        )
        self.assertEqual(model.calls[0][1]["batch_size"], 2)
        self.assertTrue(model.calls[0][1]["normalize_embeddings"])

    def test_query_uses_same_cached_model_and_e5_query_prefix(self) -> None:
        strategy, model = self._strategy_with_fake_model()

        passages = strategy.embed([self.chunk])
        query = strategy.embed_query("vibration moteur")

        self.assertEqual(query, (0.0, 0.5, 1.0))
        self.assertEqual(len(query), passages[0].dimension)
        self.assertEqual(
            model.calls[1][0],
            ["query: vibration moteur"],
        )
        self.assertIs(
            strategy._model_cache[
                (strategy.model_name, strategy.model_revision, strategy.device)
            ],
            model,
        )

    def test_empty_chunks_raise_clear_validation_error(self) -> None:
        orchestrator = build_default_orchestrator()

        with self.assertRaises(EmbeddingValidationError):
            orchestrator.embed([])

    def test_invalid_chunk_raises_validation_error(self) -> None:
        orchestrator = build_default_orchestrator()

        with self.assertRaises(EmbeddingValidationError):
            orchestrator.embed([object()])

    def test_missing_sentence_transformers_is_a_dedicated_model_error(self) -> None:
        strategy = SentenceTransformerEmbedding()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(EmbeddingModelError):
                strategy.embed([self.chunk])

    def test_registry_operations_and_unknown_strategy(self) -> None:
        registry = EmbeddingRegistry()
        registry.register(FakeEmbeddingStrategy)

        self.assertTrue(registry.has("FAKE"))
        self.assertIn("fake", registry)
        self.assertIs(registry.get("fake"), FakeEmbeddingStrategy)
        self.assertEqual(registry.supported_strategies(), ("fake",))

        registry.unregister("fake")
        self.assertFalse(registry.has("fake"))
        with self.assertRaises(EmbeddingStrategyNotRegisteredError):
            registry.get("fake")

        registry.register(FakeEmbeddingStrategy)
        registry.clear()
        self.assertEqual(len(registry), 0)

    def test_orchestrator_returns_one_embedding_per_chunk(self) -> None:
        registry = EmbeddingRegistry()
        registry.register(FakeEmbeddingStrategy)
        orchestrator = EmbeddingOrchestrator(registry, strategy_name="fake")

        embeddings = orchestrator.embed([self.chunk])

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].model_name, "fake-model")


@unittest.skipUnless(
    os.environ.get("RUN_EMBEDDING_INTEGRATION") == "1",
    "Set RUN_EMBEDDING_INTEGRATION=1 to download/use the local model.",
)
class EmbeddingIntegrationTestCase(unittest.TestCase):
    """Opt-in tests that verify the real local model and full pipeline."""

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        return numerator / denominator

    def test_technical_sentences_have_semantic_similarity(self) -> None:
        chunks = [
            Chunk("Le moteur présente une vibration anormale.", 0, "test", "txt"),
            Chunk("Une vibration excessive est détectée sur le moteur.", 1, "test", "txt"),
            Chunk("Le stock de pièces détachées est insuffisant.", 2, "test", "txt"),
        ]

        embeddings = build_default_orchestrator(batch_size=3).embed(chunks)
        close_similarity = self._cosine(embeddings[0].vector, embeddings[1].vector)
        distant_similarity = self._cosine(embeddings[0].vector, embeddings[2].vector)

        self.assertGreater(close_similarity, distant_similarity)

    def test_full_file_to_embedding_pipeline(self) -> None:
        from app.chunker import build_default_orchestrator as build_chunker
        from app.data_sources import DataSourceOrchestrator
        from app.parser import ParserOrchestrator, ParserRegistry, TextParser

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maintenance.txt"
            path.write_text("Le moteur présente une vibration anormale.", encoding="utf-8")

            source = DataSourceOrchestrator().load(path)
            parser_registry = ParserRegistry()
            parser_registry.register(TextParser)
            parsed = ParserOrchestrator(parser_registry).parse(source)
            chunks = build_chunker(chunk_size=100, chunk_overlap=10).chunk(parsed)
            embeddings = build_default_orchestrator(batch_size=8).embed(chunks)

        self.assertEqual(len(embeddings), len(chunks))
        self.assertTrue(embeddings[0].vector)


if __name__ == "__main__":
    unittest.main()
