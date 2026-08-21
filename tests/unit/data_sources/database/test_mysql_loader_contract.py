import unittest

from app.data_sources.database.mysql_loader import MySQLLoader
from app.chunker.strategies.structured import StructuredChunker
from app.exceptions import DatabaseValidationError
from app.models.document import SourceDocument
from app.models.embedding import Embedding
from app.parser.strategies.database import DatabaseParser
from app.storage.strategies.mysql_storage import MySQLStorage


class TestMySQLLoaderContract(unittest.TestCase):
    def test_load_returns_source_document_with_mysql_metadata(self):
        loader = MySQLLoader(
            host="localhost",
            database="test_db",
            user="test_user",
            password="",
            query="SELECT 1 AS id, NULL AS name",
            max_rows=1,
            max_content_bytes=4096,
        )

        loader.execute_query = lambda sql, params=None: [
            {"id": 1, "name": None}
        ]

        document = loader.load()

        self.assertIsInstance(document, SourceDocument)
        self.assertEqual(document.source_type, "mysql")
        self.assertIsNone(document.source_path)
        self.assertEqual(document.mime_type, "application/x-mysql-resultset")
        self.assertEqual(document.metadata["query"], "SELECT 1 AS id, NULL AS name")
        self.assertEqual(document.metadata["query_mode"], "query")
        self.assertEqual(document.metadata["row_count"], 1)
        self.assertEqual(document.metadata["column_count"], 2)
        self.assertEqual(document.metadata["sql_operation"], "select")
        self.assertFalse(document.is_empty)
        self.assertEqual(document.size, len(document.content.encode("utf-8")))
        self.assertEqual(document.created_at, document.updated_at)

    def test_query_mode_honors_max_rows_limit(self):
        captured = {}

        loader = MySQLLoader(
            host="localhost",
            database="test_db",
            user="test_user",
            password="",
            query="SELECT id, name FROM equipment ORDER BY id",
            max_rows=2,
        )

        def fake_execute(sql, params=None):
            captured["sql"] = sql
            return [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
            ]

        loader.execute_query = fake_execute
        document = loader.load()

        self.assertIn("LIMIT 2", captured["sql"])
        self.assertTrue(document.metadata["limited"])
        self.assertEqual(document.metadata["row_count"], 2)
        self.assertNotIn("--- Row", document.content)
        self.assertIn("id: 1", document.content)
        self.assertIn("id: 2", document.content)

    def test_gmao_parent_metadata_reaches_chunk_and_is_supported_by_mysql_storage(self):
        loader = MySQLLoader(
            host="localhost",
            database="test_db",
            user="test_user",
            query="SELECT description FROM panne WHERE id_panne = 7",
            id_panne=7,
        )

        def fake_execute(sql, params=None):
            if "FROM panne WHERE id_panne = :parent_id" in sql:
                return [{"id_panne": 7, "id_equipement": 12}]
            return [{"description": "Vibration anormale"}]

        loader.execute_query = fake_execute
        document = loader.load()
        parsed = DatabaseParser().parse(document)
        chunks = StructuredChunker(chunk_size=500, chunk_overlap=0).chunk(parsed)

        self.assertEqual(chunks[0].metadata["id_panne"], 7)
        self.assertEqual(chunks[0].metadata["id_equipement"], 12)
        embedding = Embedding(
            chunk_id=chunks[0].chunk_id,
            vector=(0.1,),
            model_name="test",
            dimension=1,
        )
        storage = MySQLStorage.__new__(MySQLStorage)
        self.assertTrue(storage.supports(chunks, [embedding]))

    def test_unknown_gmao_parent_is_rejected_before_loading_content(self):
        loader = MySQLLoader(
            host="localhost",
            database="test_db",
            user="test_user",
            query="SELECT description FROM document WHERE id_document = 99",
            id_document=99,
        )
        loader.execute_query = lambda sql, params=None: []

        with self.assertRaises(DatabaseValidationError) as error:
            loader.load()

        self.assertEqual(error.exception.details["id_document"], 99)


if __name__ == "__main__":
    unittest.main()
