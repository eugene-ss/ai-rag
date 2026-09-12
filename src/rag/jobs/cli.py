from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.jobs.reindex import reindex
from rag.jobs.scheduled_eval import scheduled_eval
from rag.observability.logging import configure_logging
from rag.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(prog="rag", description="RAG offline jobs CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Run offline ingest+index")
    ingest_p.add_argument("--source", type=Path, default=Path("data/raw"))
    ingest_p.add_argument("--index-version", default=None)

    reindex_p = sub.add_parser("reindex", help="Rebuild index and flip alias")
    reindex_p.add_argument("--source", type=Path, default=Path("data/raw"))
    reindex_p.add_argument("--index-version", required=True)
    reindex_p.add_argument("--no-activate", action="store_true")

    eval_p = sub.add_parser("eval", help="Run golden-set evaluation")
    eval_p.add_argument("--dataset", type=Path, default=Path("tests/fixtures/golden.jsonl"))

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command in {"ingest", "reindex"}:
        version = args.index_version or settings.index_version
        activate = not getattr(args, "no_activate", False)
        n = reindex(args.source, index_version=version, activate=activate)
        print(f"Indexed {n} chunks at version {version}")
        return 0

    if args.command == "eval":
        # Ensure corpus is indexed before eval when stores are empty
        if VECTOR_EMPTY():
            fixture = Path("tests/fixtures/corpus")
            if fixture.exists():
                reindex(fixture, index_version=settings.index_version, activate=True)
        print(scheduled_eval(args.dataset))
        return 0

    return 1


def VECTOR_EMPTY() -> bool:
    from rag.jobs.reindex import VECTOR_STORE

    return VECTOR_STORE.count() == 0


if __name__ == "__main__":
    sys.exit(main())
