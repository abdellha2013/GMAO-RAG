from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.data_sources.file.json_loader import JSONLoader
from app.exceptions import ParserValidationError
from app.models.document import SourceDocument
from app.parser.strategies.structured import StructuredParser


def test_parse_json_normalizes_pretty_printed_json() -> None:
    path = Path("tests/data/json/valid_object.json")
    content = path.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)

    document = SourceDocument(
        source_name=path.name,
        source_type="json",
        source_path=path,
        content=content,
        mime_type="application/json",
        size=len(content.encode("utf-8")),
        created_at=now,
        updated_at=now,
        metadata={"origin": "test"},
    )

    parsed = StructuredParser().parse(document)

    assert parsed.content == (
        '{\n'
        '  "active": true,\n'
        '  "hours": 125,\n'
        '  "machine": "CNC-01",\n'
        '  "status": "Running",\n'
        '  "temperature": 42.7\n'
        '}'
    )
    assert parsed.source_type == document.source_type
    assert parsed.mime_type == document.mime_type
    assert parsed.size == document.size
    assert parsed.parsed_at is not None


def test_parse_csv_normalizes_tabular_content() -> None:
    path = Path("tests/data/csv/valid_comma.csv")
    content = path.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)

    document = SourceDocument(
        source_name=path.name,
        source_type="csv",
        source_path=path,
        content=content,
        mime_type="text/csv",
        size=len(content.encode("utf-8")),
        created_at=now,
        updated_at=now,
        metadata={"origin": "test"},
    )

    parsed = StructuredParser().parse(document)

    assert parsed.content == (
        "id,name,machine,status,hours\n"
        "1,Ahmed,CNC-01,Running,125\n"
        "2,Sara,CNC-02,Maintenance,84\n"
        "3,Youssef,CNC-03,Stopped,210\n"
        "4,John,CNC-04,Running,58"
    )
    assert parsed.parsed_at is not None


def test_parse_json_loader_output_with_json_loader() -> None:
    document = JSONLoader(Path("tests/data/json/valid_object.json")).load()

    parsed = StructuredParser().parse(document)

    assert "machine: CNC-01" in parsed.content
    assert "status: Running" in parsed.content
    assert parsed.source_type == "json"
    assert parsed.parsed_at is not None


def test_parse_unsupported_source_type_raises_error() -> None:
    document = SourceDocument(
        source_name="invalid.xml",
        source_type="xml",
        content="<root></root>",
        mime_type="application/xml",
        size=16,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"origin": "test"},
    )

    with pytest.raises(ParserValidationError):
        StructuredParser().parse(document)
