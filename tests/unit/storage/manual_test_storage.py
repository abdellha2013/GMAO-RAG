"""Test manuel — app.storage

Script autonome (pas pytest) à exécuter directement depuis la racine du
projet réel :

    python manual_test_storage.py

Il couvre toutes les fonctionnalités et cas limites du module :
registry, alignement chunk/embedding, orchestrateur (save/delete,
stop_on_failure, raise_on_partial_failure, propagation mark_indexed),
et la logique interne des deux stratégies concrètes (MySQLStorage,
QdrantStorage) via des connexions/clients simulés — aucune vraie base
MySQL ni instance Qdrant n'est nécessaire pour l'exécuter.

Chaque cas imprime [OK]/[FAIL]. Le script se termine avec un code de
sortie non nul si au moins un cas échoue, pour pouvoir être branché sur
un pipeline CI même sans pytest.
"""
from __future__ import annotations
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import sys
import traceback
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch

from app.exceptions import (
    InvalidStorageStrategyError,
    PartialStorageError,
    StorageAlignmentError,
    StorageConnectionError,
    StorageStrategyNotRegisteredError,
    StorageValidationError,
    StorageWriteError,
)
from app.models.chunk import Chunk
from app.models.embedding import Embedding
from app.storage.base import StorageOutcome, StorageReport, StorageStrategy
from app.storage.orchestrator import StorageOrchestrator
from app.storage.registry import StorageRegistry
from app.storage.strategies.mysql_storage import MySQLStorage
from app.storage.strategies.qdrant_storage import QdrantStorage

# ---------------------------------------------------------------------
# Mini-runner (pas de dépendance à pytest)
# ---------------------------------------------------------------------

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"[OK]   {name}")
    except AssertionError as exc:
        _RESULTS.append((name, False, str(exc)))
        print(f"[FAIL] {name} — {exc}")
    except Exception as exc:  # noqa: BLE001 — on veut capturer et reporter, pas planter
        _RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


def make_chunk(
    *,
    chunk_id: str | None = None,
    chunk_index: int = 0,
    source_name: str = "doc.txt",
    source_type: str = "txt",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        content="contenu de test",
        chunk_index=chunk_index,
        source_name=source_name,
        source_type=source_type,
        chunk_id=chunk_id,
        metadata=dict(metadata or {}),
    )


def make_embedding(
    *,
    chunk_id: str,
    chunk_index: int = 0,
    source_name: str = "doc.txt",
) -> Embedding:
    # Conservés pour la lisibilité des scénarios d'alignement : l'identité
    # effective est désormais déjà portée par ``chunk_id``.
    del chunk_index, source_name
    return Embedding(
        vector=(0.1, 0.2, 0.3),
        chunk_id=chunk_id,
        model_name="manual-test",
        dimension=3,
    )


# ---------------------------------------------------------------------
# Stratégies factices pour tester l'orchestrateur sans DB/Qdrant réels
# ---------------------------------------------------------------------

class _RecordingStrategy(StorageStrategy):
    """Stratégie de test : succès configurable, enregistre les appels."""

    name = "stub"
    calls: list[str] = []
    fail_on: set[str] = set()

    def __init__(self, **_options: Any) -> None:
        pass

    def supports(self, chunks, embeddings) -> bool:
        return True

    def save(self, chunks, embeddings) -> StorageOutcome:
        self.calls.append(f"{self.name}.save")
        if "save" in self.fail_on:
            raise StorageWriteError(message=f"{self.name} save failed", details={"n": len(chunks)})
        return StorageOutcome(self.name, tuple(range(len(chunks))))

    def delete(self, chunk_ids) -> StorageOutcome:
        self.calls.append(f"{self.name}.delete")
        if "delete" in self.fail_on:
            raise StorageWriteError(message=f"{self.name} delete failed")
        return StorageOutcome(self.name, tuple(chunk_ids))


def make_strategy_class(name: str, *, fail_on: set[str] = frozenset()) -> type[_RecordingStrategy]:
    """Construit une sous-classe nommée et configurée de _RecordingStrategy."""
    return type(
        f"Strategy_{name}",
        (_RecordingStrategy,),
        {"name": name, "calls": [], "fail_on": set(fail_on)},
    )


# =======================================================================
# 1. StorageRegistry
# =======================================================================

def test_registry_register_get_has_supported():
    registry = StorageRegistry()
    ok_cls = make_strategy_class("ok")
    registry.register(ok_cls)
    assert registry.get("ok") is ok_cls
    assert registry.has("OK") is True  # normalisation
    assert registry.has("unknown") is False
    assert registry.supported_strategies() == ("ok",)


def test_registry_register_does_not_instantiate():
    calls = {"init": 0}

    class ExplodingInit(StorageStrategy):
        name = "exploding"

        def __init__(self, **_options: Any) -> None:
            calls["init"] += 1
            raise RuntimeError("ne doit jamais être appelé par register()")

        def supports(self, chunks, embeddings):
            return True

        def save(self, chunks, embeddings):
            return StorageOutcome(self.name)

        def delete(self, chunk_ids):
            return StorageOutcome(self.name)

    registry = StorageRegistry()
    registry.register(ExplodingInit)  # ne doit pas lever
    assert calls["init"] == 0, "register() a instancié la classe alors qu'il ne devrait pas"


def test_registry_duplicate_name_rejected():
    registry = StorageRegistry()
    a = make_strategy_class("dup")
    b = make_strategy_class("dup")
    registry.register(a)
    try:
        registry.register(b)
        raise AssertionError("un second enregistrement sous le même nom aurait dû échouer")
    except StorageValidationError:
        pass


def test_registry_invalid_class_rejected():
    registry = StorageRegistry()
    try:
        registry.register(object)  # type: ignore[arg-type]
        raise AssertionError("une classe qui n'hérite pas de StorageStrategy aurait dû être rejetée")
    except InvalidStorageStrategyError:
        pass


def test_registry_get_unknown_raises():
    registry = StorageRegistry()
    try:
        registry.get("does-not-exist")
        raise AssertionError("get() sur un nom inconnu aurait dû lever")
    except StorageStrategyNotRegisteredError:
        pass


def test_registry_unregister_does_not_instantiate():
    calls = {"init": 0}

    class ExplodingInit(StorageStrategy):
        name = "exploding2"

        def __init__(self, **_options: Any) -> None:
            calls["init"] += 1
            raise RuntimeError("ne doit jamais être appelé par unregister()")

        def supports(self, chunks, embeddings):
            return True

        def save(self, chunks, embeddings):
            return StorageOutcome(self.name)

        def delete(self, chunk_ids):
            return StorageOutcome(self.name)

    registry = StorageRegistry()
    registry.register(ExplodingInit)
    registry.unregister("exploding2")
    assert calls["init"] == 0
    assert registry.has("exploding2") is False


def test_registry_unregister_unknown_raises():
    registry = StorageRegistry()
    try:
        registry.unregister("nope")
        raise AssertionError("unregister() sur un nom inconnu aurait dû lever")
    except StorageStrategyNotRegisteredError:
        pass


def test_registry_clear():
    registry = StorageRegistry()
    registry.register(make_strategy_class("a"))
    registry.register(make_strategy_class("b"))
    registry.clear()
    assert registry.supported_strategies() == ()


def test_registry_build_default_does_not_require_env():
    # Reproduit le test de non-régression : construire le registre par
    # défaut ne doit jamais instancier MySQLStorage/QdrantStorage, donc
    # ne doit jamais échouer faute de variables d'environnement.
    from app.storage import build_default_registry

    registry = build_default_registry()
    assert set(registry.supported_strategies()) == {"mysql", "qdrant"}


# =======================================================================
# 2. StorageOutcome / StorageReport
# =======================================================================

def test_outcome_success_property():
    ok = StorageOutcome("mysql", saved_ids=(1, 2))
    ko = StorageOutcome("mysql", failures=({"message": "x"},))
    assert ok.success is True
    assert ko.success is False


def test_report_aggregation_properties():
    ok = StorageOutcome("mysql", saved_ids=(1,))
    ko = StorageOutcome("qdrant", failures=({"message": "boom"},))
    report_all_ok = StorageReport((ok,))
    report_mixed = StorageReport((ok, ko))
    report_empty = StorageReport(())

    assert report_all_ok.has_failures is False
    assert report_all_ok.is_full_success is True
    assert report_mixed.has_failures is True
    assert report_mixed.is_full_success is False
    assert report_mixed.failures == ({"message": "boom"},)
    assert report_empty.is_full_success is False  # aucun outcome != succès


# =======================================================================
# 3. Alignement chunk/embedding (bug critique corrigé)
# =======================================================================

def _registry_with_stub(fail_on: set[str] = frozenset()) -> StorageRegistry:
    registry = StorageRegistry()
    registry.register(make_strategy_class("stub", fail_on=fail_on))
    return registry


def test_alignment_recursive_chunker_chunk_id_none_matches():
    # RecursiveChunker laisse chunk_id=None ; Embedding.chunk_id retombe
    # sur "source:index". Ça doit maintenant s'aligner correctement.
    chunk = make_chunk(chunk_id=None, chunk_index=0, source_name="manuel.pdf")
    embedding = make_embedding(chunk_id="manuel.pdf:0", chunk_index=0, source_name="manuel.pdf")

    orchestrator = StorageOrchestrator(_registry_with_stub(), strategy_sequence=("stub",))
    report = orchestrator.save([chunk], [embedding])
    assert report.is_full_success


def test_alignment_markdown_chunker_explicit_chunk_id_matches():
    chunk = make_chunk(chunk_id="manuel.md:0", chunk_index=0, source_name="manuel.md")
    embedding = make_embedding(chunk_id="manuel.md:0", chunk_index=0, source_name="manuel.md")

    orchestrator = StorageOrchestrator(_registry_with_stub(), strategy_sequence=("stub",))
    report = orchestrator.save([chunk], [embedding])
    assert report.is_full_success


def test_alignment_mixed_batch_recursive_and_markdown():
    chunks = [
        make_chunk(chunk_id=None, chunk_index=0, source_name="a.txt"),
        make_chunk(chunk_id="b.md:0", chunk_index=0, source_name="b.md"),
    ]
    embeddings = [
        make_embedding(chunk_id="a.txt:0", chunk_index=0, source_name="a.txt"),
        make_embedding(chunk_id="b.md:0", chunk_index=0, source_name="b.md"),
    ]
    orchestrator = StorageOrchestrator(_registry_with_stub(), strategy_sequence=("stub",))
    report = orchestrator.save(chunks, embeddings)
    assert report.is_full_success


def test_alignment_length_mismatch_raises():
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    orchestrator = StorageOrchestrator(_registry_with_stub(), strategy_sequence=("stub",))
    try:
        orchestrator.save([chunk], [])
        raise AssertionError("un batch de longueurs différentes aurait dû lever StorageAlignmentError")
    except StorageAlignmentError as exc:
        assert exc.details.get("chunk_count") == 1
        assert exc.details.get("embedding_count") == 0


def test_alignment_identity_mismatch_raises():
    # Même longueur, mais l'embedding ne correspond pas au bon chunk.
    chunk = make_chunk(chunk_id=None, chunk_index=0, source_name="a.txt")
    wrong_embedding = make_embedding(chunk_id="b.txt:0", chunk_index=0, source_name="b.txt")
    orchestrator = StorageOrchestrator(_registry_with_stub(), strategy_sequence=("stub",))
    try:
        orchestrator.save([chunk], [wrong_embedding])
        raise AssertionError("un désalignement réel aurait dû lever StorageAlignmentError")
    except StorageAlignmentError as exc:
        assert exc.details.get("chunk_id") == "a.txt:0"
        assert exc.details.get("embedding_chunk_id") == "b.txt:0"


# =======================================================================
# 4. StorageOrchestrator.save() — séquencement et politiques d'échec
# =======================================================================

def test_save_happy_path_multiple_strategies():
    registry = StorageRegistry()
    registry.register(make_strategy_class("a"))
    registry.register(make_strategy_class("b"))
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(registry, strategy_sequence=("a", "b"))
    report = orchestrator.save([chunk], [embedding])
    assert report.is_full_success
    assert [o.strategy_name for o in report.outcomes] == ["a", "b"]


def test_save_stop_on_failure_true_raises_and_stops():
    registry = StorageRegistry()
    registry.register(make_strategy_class("failing", fail_on={"save"}))
    ok_cls = make_strategy_class("ok")
    registry.register(ok_cls)
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(
        registry, strategy_sequence=("failing", "ok"), stop_on_failure=True
    )
    try:
        orchestrator.save([chunk], [embedding])
        raise AssertionError("stop_on_failure=True aurait dû laisser remonter StorageWriteError")
    except StorageWriteError:
        pass
    assert ok_cls.calls == [], "la stratégie suivante n'aurait pas dû être exécutée"


def test_save_stop_on_failure_false_continues_and_collects():
    registry = StorageRegistry()
    registry.register(make_strategy_class("failing", fail_on={"save"}))
    ok_cls = make_strategy_class("ok")
    registry.register(ok_cls)
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(
        registry,
        strategy_sequence=("failing", "ok"),
        stop_on_failure=False,
        raise_on_partial_failure=False,
    )
    report = orchestrator.save([chunk], [embedding])
    assert report.has_failures
    assert len(report.outcomes) == 2
    assert report.outcomes[1].success
    assert ok_cls.calls == ["ok.save"], "la stratégie suivante aurait dû être exécutée malgré l'échec"


def test_save_failure_details_preserved():
    registry = StorageRegistry()
    registry.register(make_strategy_class("failing", fail_on={"save"}))
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(
        registry, strategy_sequence=("failing",), raise_on_partial_failure=False
    )
    report = orchestrator.save([chunk], [embedding])
    failure = report.failures[0]
    assert failure["message"] == "failing save failed"
    assert failure["error_code"] == "STORAGE_WRITE_ERROR"
    assert failure["details"] == {"n": 1}


def test_save_raises_partial_storage_error_by_default():
    registry = StorageRegistry()
    registry.register(make_strategy_class("ok"))
    registry.register(make_strategy_class("failing", fail_on={"save"}))
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(registry, strategy_sequence=("ok", "failing"))
    try:
        orchestrator.save([chunk], [embedding])
        raise AssertionError("PartialStorageError attendue (comportement par défaut)")
    except PartialStorageError as exc:
        assert "failures" in exc.details
        assert len(exc.details["failures"]) == 1


def test_save_no_partial_error_when_disabled():
    registry = StorageRegistry()
    registry.register(make_strategy_class("ok"))
    registry.register(make_strategy_class("failing", fail_on={"save"}))
    chunk = make_chunk(chunk_id=None, source_name="a.txt")
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(
        registry,
        strategy_sequence=("ok", "failing"),
        raise_on_partial_failure=False,
    )
    report = orchestrator.save([chunk], [embedding])  # ne doit pas lever
    assert report.has_failures


# =======================================================================
# 5. Propagation mark_indexed (MySQL <- Qdrant)
# =======================================================================

def test_mark_indexed_called_after_qdrant_success():
    marked: list[Any] = []

    mysql_cls = make_strategy_class("mysql")

    def mark_indexed(self, chunk_ids):
        marked.extend(chunk_ids)

    mysql_cls.mark_indexed = mark_indexed  # type: ignore[attr-defined]

    qdrant_cls = make_strategy_class("qdrant")

    registry = StorageRegistry()
    registry.register(mysql_cls)
    registry.register(qdrant_cls)

    chunk = make_chunk(chunk_id=None, source_name="a.txt", metadata={"id_chunk": 42})
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(registry, strategy_sequence=("mysql", "qdrant"))
    orchestrator.save([chunk], [embedding])
    assert marked == [42], "mark_indexed aurait dû être appelée avec l'id_chunk du batch"


def test_mark_indexed_not_called_when_qdrant_fails():
    marked: list[Any] = []
    mysql_cls = make_strategy_class("mysql")

    def mark_indexed(self, chunk_ids):
        marked.extend(chunk_ids)

    mysql_cls.mark_indexed = mark_indexed  # type: ignore[attr-defined]
    qdrant_cls = make_strategy_class("qdrant", fail_on={"save"})

    registry = StorageRegistry()
    registry.register(mysql_cls)
    registry.register(qdrant_cls)

    chunk = make_chunk(chunk_id=None, source_name="a.txt", metadata={"id_chunk": 42})
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(
        registry, strategy_sequence=("mysql", "qdrant"), raise_on_partial_failure=False
    )
    orchestrator.save([chunk], [embedding])
    assert marked == [], "mark_indexed ne doit pas être appelée si Qdrant échoue"


def test_mark_indexed_skipped_when_mysql_not_registered():
    qdrant_cls = make_strategy_class("qdrant")
    registry = StorageRegistry()
    registry.register(qdrant_cls)

    chunk = make_chunk(chunk_id=None, source_name="a.txt", metadata={"id_chunk": 42})
    embedding = make_embedding(chunk_id="a.txt:0")

    orchestrator = StorageOrchestrator(registry, strategy_sequence=("qdrant",))
    report = orchestrator.save([chunk], [embedding])  # ne doit pas lever
    assert report.is_full_success


# =======================================================================
# 6. StorageOrchestrator.delete()
# =======================================================================

def test_delete_happy_path():
    registry = StorageRegistry()
    registry.register(make_strategy_class("stub"))
    orchestrator = StorageOrchestrator(registry, strategy_sequence=("stub",))
    report = orchestrator.delete([1, 2, 3])
    assert report.is_full_success
    assert report.outcomes[0].saved_ids == (1, 2, 3)


def test_delete_stop_on_failure_true_raises():
    registry = StorageRegistry()
    registry.register(make_strategy_class("failing", fail_on={"delete"}))
    orchestrator = StorageOrchestrator(
        registry, strategy_sequence=("failing",), stop_on_failure=True
    )
    try:
        orchestrator.delete([1])
        raise AssertionError("stop_on_failure=True aurait dû laisser remonter l'erreur")
    except StorageWriteError:
        pass


def test_delete_stop_on_failure_false_collects():
    registry = StorageRegistry()
    registry.register(make_strategy_class("failing", fail_on={"delete"}))
    ok_cls = make_strategy_class("ok")
    registry.register(ok_cls)
    orchestrator = StorageOrchestrator(
        registry,
        strategy_sequence=("failing", "ok"),
        stop_on_failure=False,
        raise_on_partial_failure=False,
    )
    report = orchestrator.delete([1])
    assert report.has_failures
    assert ok_cls.calls == ["ok.delete"]


# =======================================================================
# 7. MySQLStorage — logique interne (connexion/moteur simulés)
# =======================================================================

def test_mysql_supports_true_with_id_document():
    with patch("app.storage.strategies.mysql_storage.create_engine"):
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")
    chunk = make_chunk(metadata={"id_document": 7})
    assert strategy.supports([chunk], []) is True


def test_mysql_supports_true_with_id_panne():
    with patch("app.storage.strategies.mysql_storage.create_engine"):
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")
    chunk = make_chunk(metadata={"id_panne": 3})
    assert strategy.supports([chunk], []) is True


def test_mysql_supports_false_without_parent_id():
    with patch("app.storage.strategies.mysql_storage.create_engine"):
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")
    chunk = make_chunk(metadata={})  # ni id_document ni id_panne
    assert strategy.supports([chunk], []) is False


def test_mysql_save_raises_validation_error_when_unsupported():
    with patch("app.storage.strategies.mysql_storage.create_engine"):
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")
    chunk = make_chunk(metadata={})
    embedding = make_embedding(chunk_id=chunk.chunk_id or "doc.txt:0")
    try:
        strategy.save([chunk], [embedding])
        raise AssertionError("save() sans id_document/id_panne aurait dû lever StorageValidationError")
    except StorageValidationError:
        pass


def test_mysql_missing_dsn_raises_validation_error():
    with patch.dict("os.environ", {}, clear=True):
        with patch("app.storage.strategies.mysql_storage.load_dotenv"):
            try:
                MySQLStorage(dsn=None)
                raise AssertionError("l'absence totale de DSN aurait dû lever StorageValidationError")
            except StorageValidationError:
                pass


def test_mysql_dsn_priority_param_over_env():
    with patch("app.storage.strategies.mysql_storage.create_engine") as mocked_engine:
        with patch.dict("os.environ", {"MYSQL_DSN": "mysql+pymysql://env/db"}, clear=False):
            MySQLStorage(dsn="mysql+pymysql://explicit/db")
    used_dsn = mocked_engine.call_args[0][0]
    assert used_dsn == "mysql+pymysql://explicit/db"


def test_mysql_dsn_from_components_when_no_dsn_var():
    env = {
        "GMAO_DB_HOST": "dbhost",
        "GMAO_DB_USER": "gmao",
        "GMAO_DB_NAME": "gmaodb",
        "GMAO_DB_PASSWORD": "secret",
        "GMAO_DB_PORT": "3307",
    }
    with patch("app.storage.strategies.mysql_storage.create_engine") as mocked_engine:
        with patch.dict("os.environ", env, clear=True):
            with patch("app.storage.strategies.mysql_storage.load_dotenv"):
                MySQLStorage(dsn=None)
    used_dsn = mocked_engine.call_args[0][0]
    assert used_dsn == "mysql+pymysql://gmao:secret@dbhost:3307/gmaodb"


def test_mysql_save_operational_error_maps_to_connection_error():
    from sqlalchemy.exc import OperationalError

    with patch("app.storage.strategies.mysql_storage.create_engine") as mocked_engine:
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__.side_effect = OperationalError("stmt", {}, Exception("refused"))
        mocked_engine.return_value.begin.return_value = mock_conn_cm
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")

    chunk = make_chunk(metadata={"id_document": 1})
    embedding = make_embedding(chunk_id=chunk.chunk_id or "doc.txt:0")
    try:
        strategy.save([chunk], [embedding])
        raise AssertionError("une OperationalError aurait dû être catégorisée en StorageConnectionError")
    except StorageConnectionError:
        pass


def test_mysql_save_other_sqlalchemy_error_maps_to_write_error():
    from sqlalchemy.exc import IntegrityError

    with patch("app.storage.strategies.mysql_storage.create_engine") as mocked_engine:
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__.side_effect = IntegrityError("stmt", {}, Exception("dup key"))
        mocked_engine.return_value.begin.return_value = mock_conn_cm
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")

    chunk = make_chunk(metadata={"id_document": 1})
    embedding = make_embedding(chunk_id=chunk.chunk_id or "doc.txt:0")
    try:
        strategy.save([chunk], [embedding])
        raise AssertionError("une IntegrityError aurait dû être catégorisée en StorageWriteError")
    except StorageWriteError:
        pass


def test_mysql_delete_removes_child_tables_before_parent():
    executed: list[str] = []

    with patch("app.storage.strategies.mysql_storage.create_engine") as mocked_engine:
        mock_connection = MagicMock()

        def record_execute(statement, params=None):
            executed.append(str(statement))
            result = MagicMock()
            result.lastrowid = 1
            return result

        mock_connection.execute.side_effect = record_execute
        mock_conn_cm = MagicMock()
        mock_conn_cm.__enter__.return_value = mock_connection
        mocked_engine.return_value.begin.return_value = mock_conn_cm
        strategy = MySQLStorage(dsn="mysql+pymysql://u:p@h/db")

    strategy.delete([99])
    joined = " | ".join(executed)
    assert "document_chunk" in joined or "panne_chunk" in joined
    assert "chunk_rag" in joined
    # les tables filles doivent être supprimées avant chunk_rag
    last_child_index = max(
        joined.find("document_chunk"), joined.find("panne_chunk")
    )
    parent_index = joined.find("DELETE FROM chunk_rag")
    assert last_child_index < parent_index, "chunk_rag supprimé avant les tables filles"


# =======================================================================
# 8. QdrantStorage — logique interne (client simulé)
# =======================================================================

def test_qdrant_supports_requires_int_id_chunk_and_vector():
    with patch("app.storage.strategies.qdrant_storage.QdrantClient"):
        strategy = QdrantStorage(host="localhost", port=6333)

    chunk_ok = make_chunk(metadata={"id_chunk": 5})
    embedding_ok = make_embedding(chunk_id=chunk_ok.chunk_id or "doc.txt:0")
    assert strategy.supports([chunk_ok], [embedding_ok]) is True

    chunk_missing_id = make_chunk(metadata={})  # pas d'id_chunk -> mysql pas encore passé
    embedding = make_embedding(chunk_id=chunk_missing_id.chunk_id or "doc.txt:0")
    assert strategy.supports([chunk_missing_id], [embedding]) is False


def test_qdrant_save_raises_validation_error_when_unsupported():
    with patch("app.storage.strategies.qdrant_storage.QdrantClient"):
        strategy = QdrantStorage(host="localhost", port=6333)
    chunk = make_chunk(metadata={})
    embedding = make_embedding(chunk_id=chunk.chunk_id or "doc.txt:0")
    try:
        strategy.save([chunk], [embedding])
        raise AssertionError("save() sans id_chunk MySQL aurait dû lever StorageValidationError")
    except StorageValidationError:
        pass


def test_qdrant_save_upsert_failure_maps_to_write_error():
    from qdrant_client.http.exceptions import ResponseHandlingException

    with patch("app.storage.strategies.qdrant_storage.QdrantClient") as mocked_client_cls:
        mocked_client_cls.return_value.upsert.side_effect = ResponseHandlingException(
            Exception("upsert failed")
        )
        strategy = QdrantStorage(host="localhost", port=6333)

    chunk = make_chunk(metadata={"id_chunk": 5})
    embedding = make_embedding(chunk_id=chunk.chunk_id or "doc.txt:0")
    try:
        strategy.save([chunk], [embedding])
        raise AssertionError("un échec d'upsert Qdrant aurait dû être StorageWriteError, pas StorageConnectionError")
    except StorageWriteError:
        pass
    except StorageConnectionError:
        raise AssertionError(
            "régression : l'échec d'upsert est catégorisé en StorageConnectionError au lieu de StorageWriteError"
        )


def test_qdrant_client_construction_failure_maps_to_connection_error():
    from qdrant_client.http.exceptions import ResponseHandlingException

    with patch("app.storage.strategies.qdrant_storage.QdrantClient") as mocked_client_cls:
        mocked_client_cls.side_effect = ResponseHandlingException(Exception("unreachable"))
        try:
            QdrantStorage(host="unreachable-host", port=6333)
            raise AssertionError("un host Qdrant injoignable à la construction aurait dû lever StorageConnectionError")
        except StorageConnectionError:
            pass


def test_qdrant_no_longer_depends_on_mysql_schema():
    import app.storage.strategies.qdrant_storage as qdrant_module

    assert not hasattr(qdrant_module, "create_engine"), (
        "QdrantStorage ne doit plus importer sqlalchemy.create_engine "
        "(régression du couplage MySQL/Qdrant)"
    )


def test_qdrant_delete_maps_failure_to_write_error():
    from qdrant_client.http.exceptions import ResponseHandlingException

    with patch("app.storage.strategies.qdrant_storage.QdrantClient") as mocked_client_cls:
        mocked_client_cls.return_value.delete.side_effect = ResponseHandlingException(
            Exception("delete failed")
        )
        strategy = QdrantStorage(host="localhost", port=6333)
    try:
        strategy.delete([5])
        raise AssertionError("un échec de suppression Qdrant aurait dû lever StorageWriteError")
    except StorageWriteError:
        pass


# =======================================================================
# Exécution
# =======================================================================

ALL_TESTS = [
    # 1. Registry
    test_registry_register_get_has_supported,
    test_registry_register_does_not_instantiate,
    test_registry_duplicate_name_rejected,
    test_registry_invalid_class_rejected,
    test_registry_get_unknown_raises,
    test_registry_unregister_does_not_instantiate,
    test_registry_unregister_unknown_raises,
    test_registry_clear,
    test_registry_build_default_does_not_require_env,
    # 2. Outcome / Report
    test_outcome_success_property,
    test_report_aggregation_properties,
    # 3. Alignement
    test_alignment_recursive_chunker_chunk_id_none_matches,
    test_alignment_markdown_chunker_explicit_chunk_id_matches,
    test_alignment_mixed_batch_recursive_and_markdown,
    test_alignment_length_mismatch_raises,
    test_alignment_identity_mismatch_raises,
    # 4. save()
    test_save_happy_path_multiple_strategies,
    test_save_stop_on_failure_true_raises_and_stops,
    test_save_stop_on_failure_false_continues_and_collects,
    test_save_failure_details_preserved,
    test_save_raises_partial_storage_error_by_default,
    test_save_no_partial_error_when_disabled,
    # 5. mark_indexed
    test_mark_indexed_called_after_qdrant_success,
    test_mark_indexed_not_called_when_qdrant_fails,
    test_mark_indexed_skipped_when_mysql_not_registered,
    # 6. delete()
    test_delete_happy_path,
    test_delete_stop_on_failure_true_raises,
    test_delete_stop_on_failure_false_collects,
    # 7. MySQLStorage
    test_mysql_supports_true_with_id_document,
    test_mysql_supports_true_with_id_panne,
    test_mysql_supports_false_without_parent_id,
    test_mysql_save_raises_validation_error_when_unsupported,
    test_mysql_missing_dsn_raises_validation_error,
    test_mysql_dsn_priority_param_over_env,
    test_mysql_dsn_from_components_when_no_dsn_var,
    test_mysql_save_operational_error_maps_to_connection_error,
    test_mysql_save_other_sqlalchemy_error_maps_to_write_error,
    test_mysql_delete_removes_child_tables_before_parent,
    # 8. QdrantStorage
    test_qdrant_supports_requires_int_id_chunk_and_vector,
    test_qdrant_save_raises_validation_error_when_unsupported,
    test_qdrant_save_upsert_failure_maps_to_write_error,
    test_qdrant_client_construction_failure_maps_to_connection_error,
    test_qdrant_no_longer_depends_on_mysql_schema,
    test_qdrant_delete_maps_failure_to_write_error,
]


def main() -> int:
    print(f"=== Test manuel app.storage — {len(ALL_TESTS)} cas ===\n")
    for test_fn in ALL_TESTS:
        check(test_fn.__name__, test_fn)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = [name for name, ok, _ in _RESULTS if not ok]

    print(f"\n=== Résultat : {passed}/{len(_RESULTS)} cas passés ===")
    if failed:
        print(f"Échecs ({len(failed)}) :")
        for name in failed:
            detail = next(msg for n, ok, msg in _RESULTS if n == name and not ok)
            print(f"  - {name}: {detail}")
        return 1
    print("Tous les cas sont passés.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
