from __future__ import annotations

import re

from rag.schemas import Citation, ScoredChunk, Source, SourceKind

_CITATION_RE = re.compile(r"\[([a-zA-Z0-9_.:\-/]{4,128})\]")


def resolve_citations(answer_text: str, scored: list[ScoredChunk]) -> list[Citation]:
    """Resolve [chunk_id] markers in the answer against retrieved chunks.

    Permissive fallback (cite top chunk when no markers) is intentional for the
    single-pass pipeline. Agent finalization must use `resolve_source_citations`
    instead, which refuses to invent grounding.
    """
    by_id = {s.chunk.chunk_id: s for s in scored}
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
        citations.append(citation_from_scored(hit))
    if not citations and scored:
        citations.append(citation_from_scored(scored[0]))
    return citations


def resolve_source_citations(
    answer_text: str,
    sources: list[Source],
    *,
    require_explicit: bool = True,
) -> list[Citation]:
    """Resolve citation markers against a heterogeneous source list.

    When `require_explicit` is True (agent mode), missing markers yield an empty
    list rather than inventing a top-source citation. Faked grounding is worse
    than a refusal.
    """
    by_ref: dict[str, Source] = {}
    for source in sources:
        by_ref[source.ref] = source
        if source.kind is SourceKind.CHUNK:
            by_ref[source.ref] = source
    citations: list[Citation] = []
    seen: set[str] = set()
    for match in _CITATION_RE.finditer(answer_text):
        token = match.group(1)
        hit = by_ref.get(token)
        if hit is None:
            for ref, source in by_ref.items():
                if ref.startswith(token) or token.startswith(ref[:12]):
                    hit = source
                    break
        if hit is None or hit.ref in seen:
            continue
        seen.add(hit.ref)
        citations.append(citation_from_source(hit))
    if not citations and not require_explicit and sources:
        citations.append(citation_from_source(sources[0]))
    return citations


def citation_from_scored(scored: ScoredChunk) -> Citation:
    return Citation(
        chunk_id=scored.chunk.chunk_id,
        doc_id=scored.chunk.doc_id,
        quote=scored.chunk.text[:200],
        score=scored.score,
        kind=SourceKind.CHUNK,
        ref=scored.chunk.chunk_id,
    )


def citation_from_source(source: Source) -> Citation:
    return Citation(
        chunk_id=source.ref if source.kind is SourceKind.CHUNK else "",
        doc_id=source.doc_id,
        quote=source.quote[:200],
        score=source.score,
        kind=source.kind,
        ref=source.ref,
        url=source.url,
        title=source.title,
    )


def source_from_scored(scored: ScoredChunk) -> Source:
    return Source(
        kind=SourceKind.CHUNK,
        ref=scored.chunk.chunk_id,
        doc_id=scored.chunk.doc_id,
        title=str(scored.chunk.metadata.get("title", scored.chunk.doc_id)),
        quote=scored.chunk.text[:500],
        score=scored.score,
        acl=scored.chunk.acl,
        metadata={"retriever": scored.retriever, "rank": scored.rank},
    )
