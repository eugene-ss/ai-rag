from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from rag.observability.tracing import current_trace_id

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the trace id attached.

    Log aggregators can then filter by trace id without parsing free text.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": current_trace_id(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    log_dir: Path | str | None = None,
    *,
    json_output: bool = False,
) -> None:
    """Configure the `rag` logger. Adds a file handler when log_dir is set."""
    root = logging.getLogger("rag")
    root.setLevel(level.upper())
    if root.handlers:
        return

    formatter: logging.Formatter = JsonFormatter() if json_output else logging.Formatter(_FORMAT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / "app.log")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rag.{name}")
