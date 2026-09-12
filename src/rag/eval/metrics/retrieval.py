from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = top & set(relevant_ids)
    return len(hits) / len(set(relevant_ids))


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    import math

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
