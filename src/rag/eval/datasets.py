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
    expected_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    # --- agent trajectory expectations -------------------------------------
    # These describe the shape a correct answer should take, not just its
    # content, so the agent gate can measure the behaviours that distinguish it
    # from single-shot retrieval. Both default to the single-round case, so
    # retrieval-only datasets are unaffected.
    retrieval_rounds: int = Field(
        default=1,
        ge=1,
        description="Retrieval rounds a correct trajectory needs. >1 marks a multi-hop question.",
    )
    expect_self_correction: bool = Field(
        default=False,
        description="Whether the agent should recover from a rejected draft rather than degrade.",
    )


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
