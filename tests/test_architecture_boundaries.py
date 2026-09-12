from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = {
    "rag.ingestion",
    "rag.parsing",
    "rag.chunking",
    "rag.eval",
    "rag.jobs",
}

API_ROOT = Path(__file__).resolve().parents[1] / "src" / "rag" / "api"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_api_does_not_import_offline_modules() -> None:
    """Rule #1 as CI: online path must not pull in ingest/index/eval/jobs."""
    violations: list[str] = []
    for path in API_ROOT.rglob("*.py"):
        imports = _imported_modules(path)
        for mod in imports:
            for forbidden in FORBIDDEN:
                if mod == forbidden or mod.startswith(forbidden + "."):
                    violations.append(f"{path.relative_to(API_ROOT.parent.parent)}: {mod}")
    assert not violations, "API imported offline modules:\n" + "\n".join(violations)
