import json
import pytest
from src.chunking import load_docs, chunk_all
from src.retrieval import DenseRetriever
from src.hybrid_search import BM25Retriever, HybridRetriever
from evaluation.retrieval_metrics import run_evaluation

RECALL_DROP_THRESHOLD = 0.01  # 1 percentage point, per project spec


@pytest.fixture(scope="module")
def current_results():
    docs = load_docs()
    chunks = chunk_all(docs, chunk_size=1500, overlap=0)
    golden = json.load(open("data/golden_set.json"))

    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)

    return run_evaluation(hybrid, golden)["overall"]


@pytest.fixture(scope="module")
def baseline():
    with open("reports/baseline.json") as f:
        return json.load(f)


def test_recall_at_10_no_regression(current_results, baseline):
    current = current_results["recall@10"]
    base = baseline["recall@10"]
    drop = base - current

    assert drop <= RECALL_DROP_THRESHOLD, (
        f"Recall@10 regression detected!\n"
        f"  baseline = {base:.4f}\n"
        f"  current  = {current:.4f}\n"
        f"  drop     = {drop:.4f} (threshold: {RECALL_DROP_THRESHOLD})"
    )


def test_mrr_sanity_check(current_results, baseline):
    # MRR is informational here, not a hard gate -- but worth flagging big drops
    current = current_results["mrr"]
    base = baseline["mrr"]
    if base - current > 0.05:
        pytest.fail(f"MRR dropped notably: {base:.4f} -> {current:.4f}")