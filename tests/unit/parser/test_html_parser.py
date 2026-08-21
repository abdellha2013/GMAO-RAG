from datetime import datetime, timezone
from pathlib import Path

from app.models.document import SourceDocument
from app.parser.strategies.html import HTMLParser


def test_parse_html_preserves_source_fields_and_normalizes_text() -> None:
    path = Path("tests/data/html/valid.html")
    content = path.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)

    document = SourceDocument(
        source_name=path.name,
        source_type="html",
        source_path=path,
        content=content,
        mime_type="text/html",
        size=len(content.encode("utf-8")),
        created_at=now,
        updated_at=now,
        metadata={"origin": "test"},
    )

    parsed = HTMLParser().parse(document)

    assert parsed.source_name == document.source_name
    assert parsed.source_type == document.source_type
    assert parsed.source_path == document.source_path
    assert parsed.mime_type == document.mime_type
    assert parsed.size == document.size
    assert parsed.created_at == document.created_at
    assert parsed.updated_at == document.updated_at
    assert parsed.parsed_at is not None
    assert parsed.content == (
        "GMAO Maintenance Report\n"
        "Maintenance Report\n"
        "Machine Information\n"
        "Machine: CNC-001\n"
        "Status: Running\n"
        "Operations\n"
        "Lubrication completed\n"
        "Coolant level checked\n"
        "Temperature normal\n"
        "Next maintenance: 2026-09-15"
    )
