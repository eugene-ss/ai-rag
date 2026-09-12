from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    body: str

    def render(self, **kwargs: str) -> str:
        missing = set(_PLACEHOLDER.findall(self.body)) - set(kwargs)
        if missing:
            msg = f"Missing placeholders for {self.name}@{self.version}: {sorted(missing)}"
            raise KeyError(msg)

        def repl(match: re.Match[str]) -> str:
            return kwargs[match.group(1)]

        return _PLACEHOLDER.sub(repl, self.body)


@lru_cache
def get(name: str, version: str = "v1") -> PromptTemplate:
    path = _TEMPLATE_DIR / f"{name}.{version}.md"
    if not path.exists():
        msg = f"Prompt not found: {name}@{version} ({path})"
        raise FileNotFoundError(msg)
    return PromptTemplate(name=name, version=version, body=path.read_text())


def list_prompts() -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for path in sorted(_TEMPLATE_DIR.glob("*.md")):
        # name.version.md
        stem = path.stem  # e.g. answer_grounded.v1
        if "." not in stem:
            continue
        name, version = stem.rsplit(".", 1)
        results.append((name, version))
    return results
