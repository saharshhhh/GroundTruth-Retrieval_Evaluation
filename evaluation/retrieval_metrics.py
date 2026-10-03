import json
import numpy as np


def is_hit(chunk, gold, tolerance=50):
    """A retrieved chunk counts as correct if it overlaps the gold span."""
    return (chunk.doc_id == gold["doc_id"]
            and chunk.start < gold["end"] + tolerance
            and chunk.end > gold["start"] - tolerance)


def evaluate_query(retriever, query, gold, k=10):
    """Returns the rank (1-indexed) of the first hit within top-k, or None if no hit."""
    results = retriever.search(query, k=k)
    for rank, (chunk, _) in enumerate(results, start=1):
        if is_hit(chunk, gold):
            return rank
    return None


def compute_metrics(ranks, total):
    """ranks: list of (rank or None) for each query. total: number of queries."""
    hits = [r for r in ranks if r is not None]

    recall_at_10 = len(hits) / total

    # MRR: average of 1/rank, with 0 for misses
    mrr = sum(1 / r for r in hits) / total if total else 0

    # nDCG@10 with binary relevance: DCG = 1/log2(rank+1) for the one relevant doc
    ndcg_vals = []
    for r in ranks:
        if r is None:
            ndcg_vals.append(0.0)
        else:
            dcg = 1 / np.log2(r + 1)
            idcg = 1 / np.log2(1 + 1)  # ideal: relevant doc at rank 1
            ndcg_vals.append(dcg / idcg)
    ndcg_at_10 = sum(ndcg_vals) / total if total else 0

    return {
        "recall@10": round(recall_at_10, 4),
        "mrr": round(mrr, 4),
        "ndcg@10": round(ndcg_at_10, 4),
        "n": total
    }


def run_evaluation(retriever, golden_set, k=10):
    """Returns overall metrics + per-category breakdown for one retriever."""
    per_query_ranks = []
    by_category = {}

    for item in golden_set:
        rank = evaluate_query(retriever, item["query"], item["gold"], k=k)
        per_query_ranks.append({
            "query": item["query"],
            "category": item["category"],
            "rank": rank
        })
        by_category.setdefault(item["category"], []).append(rank)

    overall = compute_metrics([r for r in [pq["rank"] for pq in per_query_ranks]], len(golden_set))

    category_results = {
        cat: compute_metrics(ranks, len(ranks))
        for cat, ranks in by_category.items()
    }

    return {
        "overall": overall,
        "by_category": category_results,
        "per_query": per_query_ranks  # keep for error analysis later
    }