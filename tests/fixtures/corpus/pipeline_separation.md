# Pipeline Separation

Production RAG is two separate systems connected by a clear contract.

The offline pipeline handles: Ingest, Parse, Chunk, Embed, Index, Evaluate.
The online pipeline handles: Query Rewrite, Hybrid Retrieve, Rerank, Generate, Cite, Refuse.

If ingestion and indexing logic lives inside your API path, you will eventually
ship an indexer into production requests. That is where unnecessary latency,
cost, and incidents begin.
