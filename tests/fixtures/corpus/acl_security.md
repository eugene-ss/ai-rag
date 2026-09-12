# ACL Security

Never embed or retrieve a document a user is not allowed to access.

ACL pushdown means the Principal is converted into a filter that is passed
into both the vector store and the lexical index at query time. Results are
also re-checked after fusion as defense in depth.

Chunk-level ACLs include tenant, allow_groups, and classification tags.
