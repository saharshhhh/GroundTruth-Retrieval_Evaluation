import json

results = json.load(open("reports/results.json"))

hybrid_ranks = {q["query"]: q["rank"] for q in results["Hybrid"]["per_query"]}
reranked_ranks = {q["query"]: q["rank"] for q in results["Hybrid+Reranker"]["per_query"]}

flipped = []
for q, h_rank in hybrid_ranks.items():
    r_rank = reranked_ranks.get(q)
    if h_rank is not None and (r_rank is None or r_rank > h_rank):
        flipped.append((q, h_rank, r_rank))

print(f"{len(flipped)} queries got WORSE after reranking:\n")
for q, h, r in flipped[:10]:
    print(f"  Hybrid rank={h}  ->  Reranked rank={r}")
    print(f"  {q}\n")

# Investigate the one real miss (not just a minor reshuffle)
import json
golden = json.load(open("data/golden_set.json"))

target_q = "When may a party disclose the other party’s confidential information, and what notice and limitations apply to such disclosure?"
item = next(g for g in golden if g["query"] == target_q)

from src.chunking import load_docs
docs = load_docs()
gold_text = docs[item["gold"]["doc_id"]][item["gold"]["start"]:item["gold"]["end"]]

combined_len = len(target_q.split()) + len(gold_text.split())
print("Query words:", len(target_q.split()))
print("Gold passage words:", len(gold_text.split()))
print("Combined (rough token proxy):", combined_len)