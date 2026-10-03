import json, random, time
from pathlib import Path
import cohere
from src.chunking import load_docs, chunk_document
from dotenv import load_dotenv
import os
load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

OUTPUT_PATH = Path("data/golden_candidates.json")

CATEGORIES = ["termination", "liability", "payment", "confidentiality",
              "intellectual property", "governing law"]

PROMPT = """You are helping build a legal-document retrieval test set.
Below is a passage from a contract.

1. Pick the ONE category below that best describes what this passage is actually about:
   termination, liability, payment, confidentiality, intellectual property, governing law

2. Write ONE realistic question that a lawyer would ask, such that THIS passage
   (and no other passage) is the correct answer. Do not reuse distinctive phrases
   verbatim from the passage -- paraphrase the way someone would actually ask, not quote it.

Passage:
\"\"\"{passage}\"\"\"

Respond with EXACTLY two lines, nothing else:
CATEGORY: <one of the six categories>
QUESTION: <the question>"""


def load_existing():
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    return []


def save(candidates):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(candidates, f, indent=2)


def draft_candidates(n=150, chunk_size=1500):
    docs = load_docs()
    doc_ids = list(docs.keys())

    doc_chunks = {d: chunk_document(d, docs[d], chunk_size=chunk_size, overlap=0)
                  for d in doc_ids}
    doc_chunks = {d: [c for c in cs if len(c.text.strip()) >= 200]
                  for d, cs in doc_chunks.items()}
    weights = {d: len(cs) for d, cs in doc_chunks.items() if cs}

    candidates = load_existing()
    print(f"resuming with {len(candidates)} already drafted")

    # don't re-pick chunks we already have a question for
    used = {d: set() for d in weights}
    for c in candidates:
        gold = c["gold"]
        used.setdefault(gold["doc_id"], set())
        # reconstruct the chunk_id pattern used in chunking.py
        for ch in doc_chunks.get(gold["doc_id"], []):
            if ch.start == gold["start"] and ch.end == gold["end"]:
                used[gold["doc_id"]].add(ch.chunk_id)

    doc_pool = list(weights.keys())

    while len(candidates) < n and doc_pool:
        doc_id = random.choices(doc_pool, weights=[weights[d] for d in doc_pool], k=1)[0]
        available = [c for c in doc_chunks[doc_id] if c.chunk_id not in used[doc_id]]
        if not available:
            doc_pool.remove(doc_id)
            continue

        chunk = random.choice(available)
        used[doc_id].add(chunk.chunk_id)

        try:
            resp = co.chat(
                model="command-a-plus-05-2026",
                max_tokens=600,
                messages=[{"role": "user",
                           "content": PROMPT.format(passage=chunk.text)}]
            )
        except cohere.errors.TooManyRequestsError as e:
            print(f"⚠️ rate limited after {len(candidates)} candidates: {e}")
            print("Progress saved. Re-run the script later to resume.")
            save(candidates)
            return candidates

        text = ""
        for block in resp.message.content:
            if getattr(block, "type", None) == "text":
                text = (block.text or "").strip()
                break

        if not text or "QUESTION:" not in text:
            print(f"  ⚠️ malformed response for {doc_id}, skipping")
            continue
        if not text or "QUESTION:" not in text:
            print(f"  ⚠️ malformed response for {doc_id}, skipping")
            continue

        lines = text.split("\n")
        category = next((l.split(":", 1)[1].strip() for l in lines
                          if l.strip().startswith("CATEGORY:")), "uncategorized")
        question = next((l.split(":", 1)[1].strip() for l in lines
                          if l.strip().startswith("QUESTION:")), "")

        if not question:
            print(f"  ⚠️ empty question for {doc_id}, skipping")
            continue

        candidates.append({
            "query": question,
            "gold": {"doc_id": doc_id, "start": chunk.start, "end": chunk.end},
            "category": category,
            "source_passage_preview": chunk.text[:150]
        })

        if len(candidates) % 10 == 0:
            print(f"  drafted {len(candidates)}/{n}...")
            save(candidates)   # checkpoint every 10

    save(candidates)
    return candidates


if __name__ == "__main__":
    cands = draft_candidates(n=150)
    print(f"Drafted {len(cands)} total -> {OUTPUT_PATH}")