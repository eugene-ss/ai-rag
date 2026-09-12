from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.schemas import Document


class PdfParser:
    """Stub PDF parser. Install the 'pdf' stack or provide text upstream."""

    mime_types: tuple[str, ...] = ("application/pdf",)

    def parse(self, document: Document) -> Document:
        if document.text is not None:
            return document.model_copy(
                update={"metadata": {**document.metadata, "parser": "pdf-passthrough"}}
            )
        raise MissingBackendError("PdfParser", "pdf")
