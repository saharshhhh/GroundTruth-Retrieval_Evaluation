import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, chunks, model_name="BAAI/bge-small-en-v1.5"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True,
            batch_size=64, show_progress_bar=True)

    def search(self, query, k=10):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q                    # cosine similarity
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]