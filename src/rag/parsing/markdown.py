from __future__ import annotations

from rag.schemas import Document
from rag.utils.text import collapse_blank_lines


class MarkdownParser:
    mime_types: tuple[str, ...] = ("text/markdown",)

    def parse(self, document: Document) -> Document:
        if document.text is None:
            msg = f"No text available for {document.doc_id}"
            raise ValueError(msg)
        # Normalize markdown for chunking: collapse excessive blank lines.
        text = collapse_blank_lines(document.text)
        return document.model_copy(
            update={"text": text, "metadata": {**document.metadata, "parser": "markdown"}}
        )
