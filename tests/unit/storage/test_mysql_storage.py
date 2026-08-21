"""Tests for requirement #5 (error categorization) and #9 (FK-safe delete)
of MySQLStorage, using a fake SQLAlchemy engine so no real database or
network connection is required.
"""
from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from app.exceptions import StorageConnectionError, StorageWriteError
from app.storage.strategies.mysql_storage import MySQLStorage


class _FakeConnection:
    def __init__(self, recorder, fail_with=None):
        self._recorder = recorder
        self._fail_with = fail_with
        self._lastrowid_counter = 0

    def execute(self, statement, params=None):
        sql = str(statement)
        self._recorder.append((sql, params))
        if self._fail_with is not None:
            raise self._fail_with
        self._lastrowid_counter += 1

        class _Result:
            lastrowid = self._lastrowid_counter

        return _Result()


class _FakeEngineContext:
    def __init__(self, recorder, fail_with=None):
        self._connection = _FakeConnection(recorder, fail_with)

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self, recorder, fail_with=None):
        self._recorder = recorder
        self._fail_with = fail_with

    def begin(self):
        return _FakeEngineContext(self._recorder, self._fail_with)


def _make_storage(monkeypatch, recorder, fail_with=None):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pwd@localhost/db")
    monkeypatch.setattr(
        "app.storage.strategies.mysql_storage.create_engine",
        lambda dsn: _FakeEngine(recorder, fail_with),
    )
    return MySQLStorage()


def test_operational_error_becomes_storage_connection_error(monkeypatch):
    recorder = []
    original = OperationalError("SELECT 1", {}, Exception("Connection refused"))
    storage = _make_storage(monkeypatch, recorder, fail_with=original)

    with pytest.raises(StorageConnectionError):
        storage.mark_indexed([1])


def test_integrity_error_becomes_storage_write_error(monkeypatch):
    recorder = []
    original = IntegrityError("INSERT ...", {}, Exception("Duplicate entry"))
    storage = _make_storage(monkeypatch, recorder, fail_with=original)

    with pytest.raises(StorageWriteError):
        storage.mark_indexed([1])


def test_delete_removes_child_rows_before_chunk_rag(monkeypatch):
    recorder = []
    storage = _make_storage(monkeypatch, recorder, fail_with=None)

    storage.delete([42])

    executed_tables_in_order = []
    for sql, _params in recorder:
        if "document_chunk" in sql:
            executed_tables_in_order.append("document_chunk")
        elif "panne_chunk" in sql:
            executed_tables_in_order.append("panne_chunk")
        elif "chunk_rag" in sql:
            executed_tables_in_order.append("chunk_rag")

    assert executed_tables_in_order[-1] == "chunk_rag"
    assert "document_chunk" in executed_tables_in_order
    assert "panne_chunk" in executed_tables_in_order
    assert executed_tables_in_order.index("document_chunk") < executed_tables_in_order.index("chunk_rag")
    assert executed_tables_in_order.index("panne_chunk") < executed_tables_in_order.index("chunk_rag")
