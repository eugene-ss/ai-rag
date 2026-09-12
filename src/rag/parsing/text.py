from __future__ import annotations

from rag.schemas import Document


class TextParser:
    mime_types: tuple[str, ...] = ("text/plain", "application/json")

    def parse(self, document: Document) -> Document:
        if document.text is None:
            msg = f"No text available for {document.doc_id}"
            raise ValueError(msg)
        return document.model_copy(
            update={
                "text": document.text.strip(),
                "metadata": {**document.metadata, "parser": "text"},
            }
        )
