import pytest
from datetime import datetime, timezone

from app.exceptions import ParserValidationError
from app.models.document import SourceDocument
from app.parser.registry import ParserRegistry
from app.parser.strategies.text import TextParser


@pytest.fixture
def valid_txt_document() -> SourceDocument:
    now = datetime.now(timezone.utc)
    return SourceDocument(
        source_name="notes.txt",
        source_type="txt",
        content="  Hello world  \n",
        mime_type="text/plain",
        size=42,
        created_at=now,
        updated_at=now,
        metadata={"origin": "test"},
    )


def test_supports_true_for_txt_source_type(valid_txt_document: SourceDocument) -> None:
    parser = TextParser()

    assert parser.supports(valid_txt_document)


def test_supports_false_for_text_plain_mime_type_without_txt_source_type() -> None:
    parser = TextParser()
    document = SourceDocument(
        source_name="notes",
        source_type="unknown",
        content="Hello world",
        mime_type="text/plain",
        size=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"origin": "test"},
    )

    assert not parser.supports(document)


def test_supports_false_for_unsupported_document() -> None:
    parser = TextParser()
    document = SourceDocument(
        source_name="notes.pdf",
        source_type="pdf",
        content="Hello world",
        mime_type="application/pdf",
        size=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"origin": "test"},
    )

    assert not parser.supports(document)


def test_parse_preserves_source_fields_and_sets_parsed_at(valid_txt_document: SourceDocument) -> None:
    parser = TextParser()
    parsed = parser.parse(valid_txt_document)

    assert parsed.content == "Hello world"
    assert parsed.size == valid_txt_document.size
    assert parsed.created_at == valid_txt_document.created_at
    assert parsed.updated_at == valid_txt_document.updated_at
    assert parsed.metadata == valid_txt_document.metadata
    assert parsed.parsed_at is not None


def test_parse_raises_validation_error_for_non_source_document() -> None:
    parser = TextParser()

    with pytest.raises(ParserValidationError):
        parser.parse("not a source document")  # type: ignore[arg-type]


def test_parse_raises_validation_error_for_empty_content() -> None:
    parser = TextParser()
    document = SourceDocument(
        source_name="notes.txt",
        source_type="txt",
        content="   ",
        mime_type="text/plain",
        size=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"origin": "test"},
    )

    with pytest.raises(ParserValidationError):
        parser.parse(document)


def test_parse_raises_validation_error_for_unsupported_document_type() -> None:
    parser = TextParser()
    document = SourceDocument(
        source_name="notes.pdf",
        source_type="pdf",
        content="Hello world",
        mime_type="application/pdf",
        size=1,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={"origin": "test"},
    )

    with pytest.raises(ParserValidationError):
        parser.parse(document)


def test_parser_registry_registers_text_parser_under_text_key() -> None:
    registry = ParserRegistry()
    registry.register(TextParser)

    assert "txt" in registry
    assert registry.get("txt") is TextParser
