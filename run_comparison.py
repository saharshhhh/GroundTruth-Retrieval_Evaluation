import json
from src.chunking import load_docs, chunk_all
from src.retrieval import DenseRetriever
from src.hybrid_search import BM25Retriever, HybridRetriever
from src.reranker import Reranker, RerankedRetriever
from evaluation.retrieval_metrics import run_evaluation

docs = load_docs()
chunks = chunk_all(docs, chunk_size=1500, overlap=0)
golden = json.load(open("data/golden_set.json"))

dense = DenseRetriever(chunks)
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)
reranker = Reranker()
hybrid_reranked = RerankedRetriever(hybrid, reranker)

configs = {
    "Dense": dense,
    "Hybrid": hybrid,
    "Hybrid+Reranker": hybrid_reranked,
}

results = {}
for name, retriever in configs.items():
    print(f"Evaluating {name}...")
    results[name] = run_evaluation(retriever, golden)
    print(f"  {results[name]['overall']}")

with open("reports/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
print(f"{'Config':<20} {'Recall@10':<12} {'MRR':<10} {'nDCG@10':<10}")
for name, r in results.items():
    o = r["overall"]
    print(f"{name:<20} {o['recall@10']:<12} {o['mrr']:<10} {o['ndcg@10']:<10}")