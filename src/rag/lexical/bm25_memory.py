from __future__ import annotations

import math
import re
from collections import Counter

from rag.schemas import Chunk, Principal, ScoredChunk
from rag.security.acl import is_allowed

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25MemoryIndex:
    """In-memory BM25 index for tests and demos."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._doc_len: dict[str, int] = {}
        self._avgdl = 0.0

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self._remove_stats(chunk.chunk_id)
            tokens = tokenize(chunk.text)
            self._chunks[chunk.chunk_id] = chunk
            self._tf[chunk.chunk_id] = Counter(tokens)
            self._doc_len[chunk.chunk_id] = len(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._recompute_avgdl()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        q_tokens = tokenize(query)
        if not q_tokens or not self._chunks:
            return []
        n = len(self._chunks)
        scores: list[tuple[float, Chunk]] = []
        for cid, chunk in self._chunks.items():
            if index_version and chunk.index_version != index_version:
                continue
            if not is_allowed(principal, chunk.acl):
                continue
            score = self._bm25(q_tokens, cid, n)
            if score > 0:
                scores.append((score, chunk))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            ScoredChunk(chunk=c, score=s, rank=r, retriever="lexical")
            for r, (s, c) in enumerate(scores[:top_k], start=1)
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            if cid in self._chunks:
                self._remove_stats(cid)
                del self._chunks[cid]
                del self._tf[cid]
                del self._doc_len[cid]
        self._recompute_avgdl()

    def _bm25(self, q_tokens: list[str], cid: str, n: int) -> float:
        score = 0.0
        dl = self._doc_len[cid]
        tf = self._tf[cid]
        for term in q_tokens:
            if term not in tf:
                continue
            df = self._df[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def _remove_stats(self, cid: str) -> None:
        for term in set(self._tf[cid]):
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def _recompute_avgdl(self) -> None:
        if not self._doc_len:
            self._avgdl = 0.0
        else:
            self._avgdl = sum(self._doc_len.values()) / len(self._doc_len)
