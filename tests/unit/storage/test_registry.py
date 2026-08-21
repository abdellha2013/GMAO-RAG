"""Tests for StorageRegistry: no instantiation on register/unregister.

Covers requirement #4: ``build_default_registry()`` (and therefore
``MySQLStorage``/``QdrantStorage`` registration) must not raise, connect to
a database, or otherwise require environment configuration, even when no
DB-related environment variable is set.
"""
from __future__ import annotations

import os

import pytest

from app.exceptions import StorageStrategyNotRegisteredError, StorageValidationError
from app.storage.base import StorageStrategy
from app.storage.registry import StorageRegistry


class _NeverInstantiate(StorageStrategy):
    """A strategy whose constructor blows up if it is ever called."""

    name = "boom"

    def __init__(self, **_options):
        raise AssertionError("StorageRegistry must never instantiate a strategy to register it.")

    def supports(self, chunks, embeddings):
        return True

    def save(self, chunks, embeddings):
        raise NotImplementedError

    def delete(self, chunk_ids):
        raise NotImplementedError


def test_register_does_not_instantiate():
    registry = StorageRegistry()
    registry.register(_NeverInstantiate)  # must not raise AssertionError
    assert registry.has("boom")


def test_unregister_does_not_instantiate():
    registry = StorageRegistry()
    registry.register(_NeverInstantiate)
    registry.unregister("boom")  # must not raise AssertionError
    assert not registry.has("boom")


def test_unregister_unknown_raises():
    registry = StorageRegistry()
    with pytest.raises(StorageStrategyNotRegisteredError):
        registry.unregister("does-not-exist")


def test_register_duplicate_raises():
    registry = StorageRegistry()
    registry.register(_NeverInstantiate)
    with pytest.raises(StorageValidationError):
        registry.register(_NeverInstantiate)


def test_build_default_registry_without_db_env(monkeypatch):
    """The real regression test from the correction prompt: importing and
    building the default registry must succeed with a completely empty
    DB-related environment."""
    for key in ("MYSQL_DSN", "GMAO_DB_HOST", "GMAO_DB_USER", "GMAO_DB_NAME", "GMAO_DB_PASSWORD", "GMAO_DB_PORT"):
        monkeypatch.delenv(key, raising=False)

    from app.storage import build_default_registry

    registry = build_default_registry()
    assert registry.supported_strategies() == ("mysql", "qdrant")
