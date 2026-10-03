from dataclasses import dataclass
from pathlib import Path

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    start: int   # character offset in the original document
    end: int
    text: str

def load_docs(folder="data/docs"):
    return {p.stem: p.read_text(encoding="utf-8", errors="ignore")
            for p in Path(folder).glob("*.txt")}

def chunk_document(doc_id, text, chunk_size=1000, overlap=200):
    chunks, step = [], chunk_size - overlap
    for i, start in enumerate(range(0, len(text), step)):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(f"{doc_id}::{chunk_size}::{i}", doc_id, start, end, text[start:end]))
        if end == len(text):
            break
    return chunks

def chunk_all(docs, chunk_size=1000, overlap=200):
    return [c for d, t in docs.items() for c in chunk_document(d, t, chunk_size, overlap)]