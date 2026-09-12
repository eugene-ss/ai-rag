"""Offline and online RAG pipelines.

This package deliberately re-exports **nothing**. Importing `rag.pipelines.online`
must not drag the offline pipeline — and with it ingestion, parsing, and chunking
— into the API process. Import the module you need:

    from rag.pipelines.online import OnlinePipeline
    from rag.pipelines.offline import OfflinePipeline

`tests/test_architecture_boundaries.py` fails the build if that stops holding.
"""
