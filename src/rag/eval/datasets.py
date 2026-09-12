from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenExample(BaseModel):
    """One row of a golden evaluation set."""

    id: str
    question: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def load_golden(path: Path | str) -> list[GoldenExample]:
    p = Path(path)
    examples: list[GoldenExample] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldenExample.model_validate(json.loads(line)))
    return examples
