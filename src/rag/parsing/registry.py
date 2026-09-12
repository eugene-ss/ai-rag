from __future__ import annotations

from rag.parsing.base import Parser
from rag.parsing.html import HtmlParser
from rag.parsing.markdown import MarkdownParser
from rag.parsing.pdf import PdfParser
from rag.parsing.text import TextParser
from rag.schemas import Document

_PARSERS: list[Parser] = [
    TextParser(),
    MarkdownParser(),
    HtmlParser(),
    PdfParser(),
]

REGISTRY: dict[str, Parser] = {}
for _p in _PARSERS:
    for mime in _p.mime_types:
        REGISTRY[mime] = _p


def get_parser(mime_type: str) -> Parser:
    try:
        return REGISTRY[mime_type]
    except KeyError as exc:
        msg = f"No parser for mime type: {mime_type}"
        raise KeyError(msg) from exc


def parse_document(document: Document) -> Document:
    parser = get_parser(document.mime_type)
    return parser.parse(document)


def register(mime_type: str, parser: Parser) -> None:
    REGISTRY[mime_type] = parser
