# pyrefly: ignore [missing-import]
from src.chunking import load_docs, chunk_all
docs = load_docs()
print("documents loaded:", len(docs))
chunks = chunk_all(docs)
print("total chunks:", len(chunks))
from collections import Counter
per_doc = Counter(c.doc_id for c in chunks)
print("chunks per doc — min/max/avg:",
      min(per_doc.values()), max(per_doc.values()),
      round(sum(per_doc.values())/len(per_doc), 1))