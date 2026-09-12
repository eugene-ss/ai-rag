from __future__ import annotations

import re

from rag.schemas import Document


class MarkdownParser:
    mime_types: tuple[str, ...] = ("text/markdown",)

    def parse(self, document: Document) -> Document:
        if document.text is None:
            msg = f"No text available for {document.doc_id}"
            raise ValueError(msg)
        # Normalize markdown for chunking: collapse excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", document.text.strip())
        return document.model_copy(
            update={"text": text, "metadata": {**document.metadata, "parser": "markdown"}}
        )
