from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, k=10):
        """
        candidates: list of (chunk, score) tuples, e.g. hybrid retriever's output
        Returns top-k (chunk, rerank_score) tuples, re-sorted.
        """
        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(zip([c for c, _ in candidates], scores),
                           key=lambda x: -x[1])
        return reranked[:k]


class RerankedRetriever:
    """Combines a first-stage retriever with a reranker, matching the
    same .search(query, k) interface as DenseRetriever/HybridRetriever."""

    def __init__(self, base_retriever, reranker, pool_size=50):
        self.base = base_retriever
        self.reranker = reranker
        self.pool_size = pool_size

    def search(self, query, k=10):
        candidates = self.base.search(query, k=self.pool_size)
        return self.reranker.rerank(query, candidates, k=k)