from __future__ import annotations

from rag.exceptions import MissingBackendError
from rag.observability.logging import get_logger
from rag.schemas import Document

log = get_logger("parsing.pdf")


class PdfParser:
    """Extracts text from PDFs via pypdf. Requires the 'pdf' extra.

    Scanned PDFs yield little or no text; route those through OCR upstream
    rather than indexing empty chunks.
    """

    mime_types: tuple[str, ...] = ("application/pdf",)

    def parse(self, document: Document) -> Document:
        path = document.metadata.get("path")
        if not path:
            if document.text is not None:
                return document.model_copy(
                    update={"metadata": {**document.metadata, "parser": "pdf-passthrough"}}
                )
            msg = f"no path or text available for {document.doc_id}"
            raise ValueError(msg)

        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("PdfParser", "pdf") from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text:
            log.warning("no extractable text in %s (scanned PDF?)", path)
        return document.model_copy(
            update={
                "text": text,
                "metadata": {
                    **document.metadata,
                    "parser": "pdf",
                    "page_count": len(reader.pages),
                },
            }
        )
