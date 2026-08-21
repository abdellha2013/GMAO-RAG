from datetime import datetime, timezone
from pathlib import Path

from app.models.document import SourceDocument
from app.parser.strategies.markdown import MarkdownParser


def test_parse_preserves_source_fields_and_sets_parsed_at() -> None:
    path = Path("tests/data/markdown/valid.md")
    content = path.read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)

    document = SourceDocument(
        source_name=path.name,
        source_type="markdown",
        source_path=path,
        content=content,
        mime_type="text/markdown",
        size=len(content.encode("utf-8")),
        created_at=now,
        updated_at=now,
        metadata={"origin": "test"},
    )

    parsed = MarkdownParser().parse(document)

    assert parsed.content == content.strip()
    assert parsed.source_name == document.source_name
    assert parsed.source_type == document.source_type
    assert parsed.source_path == document.source_path
    assert parsed.mime_type == document.mime_type
    assert parsed.size == document.size
    assert parsed.created_at == document.created_at
    assert parsed.updated_at == document.updated_at
    assert parsed.parsed_at is not None
