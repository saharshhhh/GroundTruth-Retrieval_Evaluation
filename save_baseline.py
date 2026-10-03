# save_baseline.py (project root)
import json
from src.chunking import load_docs, chunk_all
from src.retrieval import DenseRetriever
from src.hybrid_search import BM25Retriever, HybridRetriever
from evaluation.retrieval_metrics import run_evaluation

docs = load_docs()
chunks = chunk_all(docs, chunk_size=1500, overlap=0)
golden = json.load(open("data/golden_set.json"))

dense = DenseRetriever(chunks)
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)

results = run_evaluation(hybrid, golden)

baseline = {
    "config": "Hybrid",
    "recall@10": results["overall"]["recall@10"],
    "mrr": results["overall"]["mrr"],
    "ndcg@10": results["overall"]["ndcg@10"],
    "n_queries": results["overall"]["n"]
}

with open("reports/baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("Baseline saved:", baseline)