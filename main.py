#!/usr/bin/env python
"""Single entry point for the API process and the offline jobs.

    python main.py serve [--host H] [--port P] [--reload] [--workers N]
    python main.py ingest --source data/raw
    python main.py reindex --index-version v2 [--no-activate]
    python main.py activate --index-version v1
    python main.py versions
    python main.py eval --dataset data/golden/golden.jsonl

`serve` imports only uvicorn and settings, so the online process never pulls
ingestion, chunking, or indexing code into its address space.
"""

from __future__ import annotations

import sys

USAGE = __doc__


def _flag_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    index = args.index(flag) + 1
    if index >= len(args):
        msg = f"{flag} requires a value"
        raise SystemExit(msg)
    return args[index]


def serve(argv: list[str]) -> int:
    import uvicorn

    from rag.settings import get_settings

    settings = get_settings()
    host = _flag_value(argv, "--host") or "0.0.0.0"  # noqa: S104 - container bind
    port = int(_flag_value(argv, "--port") or 8000)
    workers = int(_flag_value(argv, "--workers") or 1)
    reload = "--reload" in argv

    uvicorn.run(
        "rag.api.app:app",
        host=host,
        port=port,
        # reload and workers are mutually exclusive in uvicorn.
        reload=reload,
        workers=None if reload else workers,
        log_level=settings.log_level.lower(),
        access_log=False,  # TraceMiddleware already logs one line per request.
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0 if args else 1

    if args[0] == "serve":
        return serve(args[1:])

    from rag.jobs.cli import main as jobs_main

    return jobs_main(args)


if __name__ == "__main__":
    sys.exit(main())
