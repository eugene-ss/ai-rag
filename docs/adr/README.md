# Architecture Decision Records

Short records of the decisions that are expensive to reverse, and the reasoning
behind them. Written so that a future maintainer can tell whether the reasoning
still holds.

| ADR | Decision |
|---|---|
| [0001](0001-two-pipelines.md) | Offline indexing and online serving are separate systems |
| [0002](0002-hybrid-retrieval-rrf.md) | Hybrid retrieval fused with Reciprocal Rank Fusion |
| [0003](0003-chunk-level-acls.md) | Access control lives on the chunk |
| [0004](0004-index-versioning.md) | Index versions with alias-based promotion |
| [0005](0005-refusal-as-a-feature.md) | Refusal is a normal return value |
| [0006](0006-protocol-based-backends.md) | Protocols with dependency-free defaults |
| [0007](0007-agent-as-third-subsystem.md) | Bounded agent as a third subsystem |

Format: context, decision, consequences. Kept short on purpose — an ADR nobody
reads is worse than no ADR.
