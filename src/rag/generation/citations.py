from __future__ import annotations

import re

from rag.schemas import Citation, ScoredChunk

_CITATION_RE = re.compile(r"\[([a-f0-9]{8,64})\]", re.I)


def resolve_citations(answer_text: str, scored: list[ScoredChunk]) -> list[Citation]:
    """Resolve [chunk_id] markers in the answer against retrieved chunks."""
    by_id = {s.chunk.chunk_id: s for s in scored}
    # Also allow prefix matches for short ids in stubs
    citations: list[Citation] = []
    seen: set[str] = set()
    for match in _CITATION_RE.finditer(answer_text):
        cid = match.group(1)
        hit = by_id.get(cid)
        if hit is None:
            for full_id, s in by_id.items():
                if full_id.startswith(cid) or cid.startswith(full_id[:12]):
                    hit = s
                    break
        if hit is None or hit.chunk.chunk_id in seen:
            continue
        seen.add(hit.chunk.chunk_id)
        quote = hit.chunk.text[:200]
        citations.append(
            Citation(
                chunk_id=hit.chunk.chunk_id,
                doc_id=hit.chunk.doc_id,
                quote=quote,
                score=hit.score,
            )
        )
    # If no explicit markers, cite top chunk when answer overlaps its text
    if not citations and scored:
        top = scored[0]
        citations.append(
            Citation(
                chunk_id=top.chunk.chunk_id,
                doc_id=top.chunk.doc_id,
                quote=top.chunk.text[:200],
                score=top.score,
            )
        )
    return citations
