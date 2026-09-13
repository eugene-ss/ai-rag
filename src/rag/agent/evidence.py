"""Deduplicated, budget-bounded evidence for one agent turn.

Owning the evidence in one place fixes a specific cost bug. Previously the
runtime appended each tool result to the message history *and* re-rendered every
accumulated source into a fresh user message on each iteration, so the model
received every passage twice and re-received all earlier passages on every
later step. Prompt size grew linearly per step even when the agent repeated the
same tool call with the same arguments, which makes total token cost quadratic
in step count.

The ledger keeps one entry per source ref, prefers the highest-scoring
observation of a ref, and renders under a character budget so a wide `top_k`
cannot push the planner prompt past the model's context window.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from rag.schemas.source import Source

# Roughly 1.5k tokens of evidence. Deliberately well under any model context so
# the history, system prompt and draft all still fit.
DEFAULT_MAX_CHARS = 6000
MIN_QUOTE_CHARS = 120


@dataclass
class EvidenceLedger:
    """Accumulated sources for one turn, deduplicated by ref."""

    max_chars: int = DEFAULT_MAX_CHARS
    max_quote_chars: int = 500
    _by_ref: dict[str, Source] = field(default_factory=dict)

    def add(self, sources: Iterable[Source]) -> list[Source]:
        """Merge sources, keeping the best-scoring observation of each ref.

        Returns only the refs that were not already known, so callers can tell
        whether a tool call actually produced new evidence.
        """
        added: list[Source] = []
        for source in sources:
            previous = self._by_ref.get(source.ref)
            if previous is None:
                self._by_ref[source.ref] = source
                added.append(source)
            elif source.score > previous.score:
                self._by_ref[source.ref] = source
        return added

    def sources(self) -> list[Source]:
        """All known sources, strongest first."""
        return sorted(self._by_ref.values(), key=lambda s: s.score, reverse=True)

    def __len__(self) -> int:
        return len(self._by_ref)

    def render(self) -> str:
        """Render the strongest sources for a prompt, under the char budget."""
        return render_sources(
            self.sources(),
            max_chars=self.max_chars,
            max_quote_chars=self.max_quote_chars,
        )


def render_sources(
    sources: list[Source],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_quote_chars: int = 500,
) -> str:
    """Render sources for planner/critic prompts, bounded by `max_chars`.

    Sources are emitted strongest-first, so truncation drops the weakest
    evidence rather than an arbitrary tail.
    """
    if not sources:
        return "(none)"

    parts: list[str] = []
    used = 0
    for index, src in enumerate(sources):
        header = f"[{src.ref}] kind={src.kind.value} doc={src.doc_id} score={src.score:.4f}"
        if src.title:
            header += f" title={src.title}"

        budget_left = max_chars - used - len(header) - 1
        if budget_left < MIN_QUOTE_CHARS and index > 0:
            omitted = len(sources) - index
            parts.append(f"({omitted} weaker source(s) omitted to stay within the prompt budget)")
            break

        entry = f"{header}\n{src.quote[: min(max_quote_chars, budget_left)]}"
        parts.append(entry)
        used += len(entry) + 2

    return "\n\n".join(parts)
