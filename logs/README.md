# Logs

Structured application logs for debugging and monitoring.

Nothing here is tracked in git. The `rag` logger writes to stdout by default;
set `RAG_LOG_DIR=logs` to also write `app.log` to this directory.

In production, ship stdout to your log aggregator instead of relying on files.
