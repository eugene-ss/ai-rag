from __future__ import annotations

import hashlib
import math
import struct


class HashEmbedder:
    """Deterministic, dependency-free embedder for local tests and demos."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.model_name = f"hash-{dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values: list[float] = []
        # Expand digest as needed for dimensions
        seed = digest
        while len(values) < self.dimensions:
            for i in range(0, len(seed) - 3, 4):
                raw = struct.unpack_from(">I", seed, i)[0]
                values.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
                if len(values) >= self.dimensions:
                    break
            seed = hashlib.sha256(seed).digest()
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]
