from __future__ import annotations

import re

from rag.schemas import Document


class HtmlParser:
    """Minimal HTML stripper (no external deps)."""

    mime_types: tuple[str, ...] = ("text/html",)

    def parse(self, document: Document) -> Document:
        if document.text is None:
            msg = f"No text available for {document.doc_id}"
            raise ValueError(msg)
        text = re.sub(r"<script[\s\S]*?</script>", " ", document.text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return document.model_copy(
            update={"text": text, "metadata": {**document.metadata, "parser": "html"}}
        )
