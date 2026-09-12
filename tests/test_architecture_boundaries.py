from __future__ import annotations

import ast
from collections import deque
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE_ROOT = SRC / "rag"

# The online request path must not be able to reach indexing, ingestion, or jobs.
# Not a style rule: if these are importable from a handler, one eventually gets
# called there, and a request triggers a reindex.
FORBIDDEN = (
    "rag.ingestion",
    "rag.parsing",
    "rag.chunking",
    "rag.eval",
    "rag.jobs",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _direct_imports(path: Path) -> set[str]:
    """Every `rag.*` module imported by this file, including inside functions."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return {m for m in found if m == "rag" or m.startswith("rag.")}


def _import_graph() -> dict[str, set[str]]:
    return {
        _module_name(path): _direct_imports(path) for path in PACKAGE_ROOT.rglob("*.py")
    }


def _reachable(graph: dict[str, set[str]], start: str) -> dict[str, list[str]]:
    """Modules reachable from `start`, each with the path that got there."""
    paths: dict[str, list[str]] = {start: [start]}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for target in graph.get(current, set()):
            # An import of a package also pulls in its __init__.
            for candidate in (target, target.rsplit(".", 1)[0]):
                if candidate in graph and candidate not in paths:
                    paths[candidate] = [*paths[current], candidate]
                    queue.append(candidate)
    return paths


def test_api_cannot_reach_offline_modules() -> None:
    """The separation is transitive, or it is not a separation."""
    graph = _import_graph()
    api_modules = [m for m in graph if m == "rag.api" or m.startswith("rag.api.")]
    assert api_modules, "no api modules discovered"

    violations: list[str] = []
    for entry in api_modules:
        for module, chain in _reachable(graph, entry).items():
            if module.startswith(FORBIDDEN):
                violations.append(" -> ".join(chain))

    assert not violations, "online path reaches offline modules:\n" + "\n".join(
        sorted(violations)
    )


def test_online_pipeline_cannot_reach_offline_pipeline() -> None:
    graph = _import_graph()
    reachable = _reachable(graph, "rag.pipelines.online")
    assert "rag.pipelines.offline" not in reachable


def test_offline_pipeline_may_use_offline_modules() -> None:
    """Sanity check: the test above proves separation, not an empty graph."""
    graph = _import_graph()
    reachable = _reachable(graph, "rag.pipelines.offline")
    assert "rag.chunking" in reachable
    assert "rag.parsing" in reachable
    assert "rag.ingestion.local_fs" in reachable


def test_schemas_depend_on_nothing_but_schemas() -> None:
    """The contract must stay free of implementation dependencies."""
    graph = _import_graph()
    for module, imports in graph.items():
        if not module.startswith("rag.schemas"):
            continue
        outside = {
            i
            for i in imports
            if not (i.startswith("rag.schemas") or i.startswith("rag.utils"))
        }
        assert not outside, f"{module} imports {sorted(outside)}"
