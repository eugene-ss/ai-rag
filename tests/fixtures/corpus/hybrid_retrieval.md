# Hybrid Retrieval

Hybrid retrieval combines **dense** vector search with **lexical** (BM25) search.
Results are fused with Reciprocal Rank Fusion (RRF) before reranking and generation.
Dense retrieval alone is not enough for production RAG systems.
