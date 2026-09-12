"""Document parsers by mime type."""

from rag.parsing.base import Parser
from rag.parsing.registry import get_parser, parse_document, register

__all__ = ["Parser", "get_parser", "parse_document", "register"]
