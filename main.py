#!/usr/bin/env python
"""Single entry point for the API process and the offline jobs.

    python main.py serve                      # online query path
    python main.py ingest --source data/raw   # offline pipeline
    python main.py reindex --index-version v2
    python main.py eval

`serve` execs uvicorn rather than importing the jobs modules, so the online
path never pulls ingestion or indexing code into its process.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 1

    command, rest = args[0], args[1:]

    if command == "serve":
        import uvicorn

        from rag.settings import get_settings

        settings = get_settings()
        host = "0.0.0.0"  # noqa: S104 - container-facing bind is intentional
        port = 8000
        if "--host" in rest:
            host = rest[rest.index("--host") + 1]
        if "--port" in rest:
            port = int(rest[rest.index("--port") + 1])
        uvicorn.run(
            "rag.api.app:app",
            host=host,
            port=port,
            log_level=settings.log_level.lower(),
        )
        return 0

    from rag.jobs.cli import main as jobs_main

    return jobs_main(args)


if __name__ == "__main__":
    sys.exit(main())
