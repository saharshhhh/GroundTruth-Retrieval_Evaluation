from collections import defaultdict
from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    # simple whitespace/punctuation tokenizer, lowercase
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=10):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]


class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, k_rrf=60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.k_rrf = k_rrf

    def search(self, query, k=10, pool_size=5):
        # pull a larger pool from each so RRF has enough candidates to fuse
        dense_results = self.dense.search(query, k=pool_size)
        bm25_results = self.bm25.search(query, k=pool_size)

        rrf_scores = defaultdict(float)
        chunk_lookup = {}

        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] += 1 / (self.k_rrf + rank)
            chunk_lookup[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] += 1 / (self.k_rrf + rank)
            chunk_lookup[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:k]
        return [(chunk_lookup[cid], rrf_scores[cid]) for cid in ranked_ids]