from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ==========================================================
# Project root
# ==========================================================


ROOT = Path(__file__).resolve().parents[4]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==========================================================
# Environment
# ==========================================================

try:
    from dotenv import load_dotenv

except ImportError as exc:
    raise RuntimeError(
        "The 'python-dotenv' package is required to run "
        "the MySQL integration tests."
    ) from exc


load_dotenv(ROOT / ".env")


# ==========================================================
# Application imports
# ==========================================================
from app.data_sources.database import load_database
from app.exceptions import GMAOError


# ==========================================================
# Configuration
# ==========================================================

def _env_value(*names: str, default: str = "") -> str:
    """Retourne une variable d'environnement sans bruit de copie/collage."""
    for name in names:
        value = os.getenv(name)
        if value is not None:
            cleaned = str(value).strip().strip('"\'').rstrip(";").strip()
            if cleaned:
                return cleaned
    return default


DB_HOST = _env_value("GMAO_DB_HOST", "host", default="localhost")

DB_PORT = int(
    _env_value("GMAO_DB_PORT", "port", default="3306")
)

DB_NAME = _env_value("GMAO_DB_NAME", "database")
DB_USER = _env_value("GMAO_DB_USER", "user")
DB_PASSWORD = _env_value("GMAO_DB_PASSWORD", "password")

TEST_TABLE = os.getenv(
    "GMAO_TEST_TABLE",
    "panne",
)

TEST_PRIMARY_KEY = os.getenv(
    "GMAO_TEST_PRIMARY_KEY",
    "id_panne",
)

MAX_ROWS = 5


# ==========================================================
# Helpers
# ==========================================================

def print_separator(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_document(document: Any) -> None:
    print()
    print("SourceDocument")
    print("-" * 40)

    print(f"Name       : {document.source_name}")
    print(f"Type       : {document.source_type}")
    print(f"Path       : {document.source_path}")
    print(f"MIME       : {document.mime_type}")
    print(f"Size       : {document.size} bytes")
    print(f"Created    : {document.created_at}")
    print(f"Modified   : {document.updated_at}")
    print(f"Is Empty   : {document.is_empty}")
    print(f"Length     : {document.content_length}")
    print(f"Extension  : {document.extension}")

    print()
    print("Metadata")
    print("-" * 40)

    for key, value in document.metadata.items():
        print(f"{key:20}: {value}")

    print()
    print("Content")
    print("-" * 40)
    print(document.content)


def run_test(
    name: str,
    callback,
    *,
    expect_error: bool = False,
) -> bool:
    """
    Exécute un scénario de test et affiche son résultat.
    """

    print_separator(name)

    try:
        result = callback()

        if expect_error:
            print("FAILED")
            print(
                "An error was expected, but the operation "
                "completed successfully."
            )
            return False

        print("SUCCESS")

        if result is not None:
            print_document(result)

        return True

    except GMAOError as exc:

        if expect_error:
            print("EXPECTED GMAO ERROR")
            print(type(exc).__name__)
            print(exc)
            return True

        print("GMAO ERROR")
        print(type(exc).__name__)
        print(exc)
        return False

    except Exception as exc:

        print("UNEXPECTED ERROR")
        print(type(exc).__name__)
        print(exc)
        return False


# ==========================================================
# Test 1 - Connection / table loading
# ==========================================================

def test_table_loading():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table=TEST_TABLE,
        max_rows=MAX_ROWS,
    )


# ==========================================================
# Test 2 - Custom query
# ==========================================================

def test_custom_query():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        query=f"SELECT * FROM `{TEST_TABLE}` LIMIT 3",
    )


# ==========================================================
# Test 3 - Query with parameters
# ==========================================================

def test_parameterized_query():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        query=(
            f"SELECT * FROM `{TEST_TABLE}` "
            f"WHERE `{TEST_PRIMARY_KEY}` = :id"
        ),
        params={"id": 1},
    )


# ==========================================================
# Test 4 - max_rows
# ==========================================================

def test_max_rows():
    document = load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table=TEST_TABLE,
        max_rows=2,
    )

    return document


# ==========================================================
# Test 5 - table + query (expects error)
# ==========================================================

def _table_and_query():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table=TEST_TABLE,
        query=f"SELECT * FROM `{TEST_TABLE}`",
    )


def test_table_and_query():
    import pytest
    with pytest.raises(GMAOError):
        _table_and_query()


# ==========================================================
# Test 6 - neither table nor query (expects error)
# ==========================================================

def _no_source():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def test_no_source():
    import pytest
    with pytest.raises(GMAOError):
        _no_source()


# ==========================================================
# Test 7 - invalid table (expects error)
# ==========================================================

def _invalid_table():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table="table-name-invalid!",
    )


def test_invalid_table():
    import pytest
    with pytest.raises(GMAOError):
        _invalid_table()


# ==========================================================
# Test 8 - missing table (expects error)
# ==========================================================

def _missing_table():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table="table_that_does_not_exist",
    )


def test_missing_table():
    import pytest
    with pytest.raises(GMAOError):
        _missing_table()


# ==========================================================
# Test 9 - invalid SQL (expects error)
# ==========================================================

def _invalid_query():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        query="SELECT THIS IS INVALID SQL",
    )


def test_invalid_query():
    import pytest
    with pytest.raises(GMAOError):
        _invalid_query()


# ==========================================================
# Test 10 - empty result
# ==========================================================

def test_empty_result():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        query=(
            f"SELECT * FROM `{TEST_TABLE}` "
            "WHERE 1 = 0"
        ),
    )


# ==========================================================
# Test 11 - unsupported driver (expects error)
# ==========================================================

def _unsupported_driver():
    return load_database(
        driver="postgresql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table=TEST_TABLE,
    )


def test_unsupported_driver():
    import pytest
    with pytest.raises(GMAOError):
        _unsupported_driver()


# ==========================================================
# Test 12 - invalid max_rows (expects error)
# ==========================================================

def _invalid_max_rows():
    return load_database(
        driver="mysql",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table=TEST_TABLE,
        max_rows=0,
    )


def test_invalid_max_rows():
    import pytest
    with pytest.raises(GMAOError):
        _invalid_max_rows()


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print_separator("MYSQL DATABASE LOADER TEST SUITE")

    print("Configuration")
    print("-" * 40)
    print(f"Host       : {DB_HOST}")
    print(f"Port       : {DB_PORT}")
    print(f"Database   : {DB_NAME}")
    print(f"User       : {DB_USER}")
    print(f"Table      : {TEST_TABLE}")
    print(f"Primary key: {TEST_PRIMARY_KEY}")

    results: list[tuple[str, bool]] = []

    # ------------------------------------------------------
    # Successful cases
    # ------------------------------------------------------

    results.append(
        (
            "Table loading",
            run_test(
                "TEST 01 - TABLE LOADING",
                test_table_loading,
            ),
        )
    )

    results.append(
        (
            "Custom query",
            run_test(
                "TEST 02 - CUSTOM QUERY",
                test_custom_query,
            ),
        )
    )

    results.append(
        (
            "Parameterized query",
            run_test(
                "TEST 03 - PARAMETERIZED QUERY",
                test_parameterized_query,
            ),
        )
    )

    results.append(
        (
            "max_rows",
            run_test(
                "TEST 04 - MAX ROWS",
                test_max_rows,
            ),
        )
    )

    # ------------------------------------------------------
    # Expected errors
    # ------------------------------------------------------

    results.append(
        (
            "table + query",
            run_test(
                "TEST 05 - TABLE + QUERY",
                _table_and_query,
                expect_error=True,
            ),
        )
    )

    results.append(
        (
            "no source",
            run_test(
                "TEST 06 - NO TABLE / NO QUERY",
                _no_source,
                expect_error=True,
            ),
        )
    )

    results.append(
        (
            "invalid table",
            run_test(
                "TEST 07 - INVALID TABLE NAME",
                _invalid_table,
                expect_error=True,
            ),
        )
    )

    results.append(
        (
            "missing table",
            run_test(
                "TEST 08 - MISSING TABLE",
                _missing_table,
                expect_error=True,
            ),
        )
    )

    results.append(
        (
            "invalid SQL",
            run_test(
                "TEST 09 - INVALID SQL",
                _invalid_query,
                expect_error=True,
            ),
        )
    )

    results.append(
        (
            "empty result",
            run_test(
                "TEST 10 - EMPTY RESULT",
                test_empty_result,
            ),
        )
    )

    results.append(
        (
            "unsupported driver",
            run_test(
                "TEST 11 - UNSUPPORTED DRIVER",
                _unsupported_driver,
                expect_error=True,
            ),
        )
    )

    results.append(
        (
            "invalid max_rows",
            run_test(
                "TEST 12 - INVALID MAX ROWS",
                _invalid_max_rows,
                expect_error=True,
            ),
        )
    )

    # ======================================================
    # Summary
    # ======================================================

    print_separator("TEST SUMMARY")

    passed = 0
    failed = 0

    for name, success in results:

        status = "PASS" if success else "FAIL"

        print(f"{status:6} | {name}")

        if success:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    print(f"Total  : {len(results)}")

    print()

    if failed:
        print("RESULT : TEST SUITE FAILED")
        sys.exit(1)

    print("RESULT : ALL TESTS PASSED")


try:
    from app.exceptions import GMAOError
    if __name__ == "__main__":
        main()
except Exception as exc:
        print("The exception is : \n",exc)
