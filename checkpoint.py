from src.chunking import load_docs, chunk_all
from src.retrieval import DenseRetriever
from src.hybrid_search import BM25Retriever, HybridRetriever
from src.reranker import Reranker, RerankedRetriever
import json

docs = load_docs()
chunks = chunk_all(docs, chunk_size=1500, overlap=0)

dense = DenseRetriever(chunks)
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)
reranker = Reranker()
hybrid_reranked = RerankedRetriever(hybrid, reranker)

# use a REAL golden-set question this time, not a made-up one
golden = json.load(open("data/golden_set.json"))
item = golden[0]
query = item["query"]
gold = item["gold"]

print("QUERY:", query)
print("GOLD DOC:", gold["doc_id"], gold["start"], gold["end"])
print()

for label, retriever in [("Dense", dense), ("Hybrid", hybrid), ("Hybrid+Reranker", hybrid_reranked)]:
    print(f"--- {label} ---")
    for c, s in retriever.search(query, k=5):
        marker = "✅" if c.doc_id == gold["doc_id"] and c.start < gold["end"] and c.end > gold["start"] else "  "
        print(f"{marker} {round(float(s), 3)}  {c.doc_id}  {c.text[:90].strip()!r}")
    print()