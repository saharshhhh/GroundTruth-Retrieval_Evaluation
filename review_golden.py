import json, re
from pathlib import Path
from src.chunking import load_docs, chunk_all
from src.retrieval import DenseRetriever

CANDIDATES_PATH = Path("data/golden_candidates.json")
GOLDEN_SET_PATH = Path("data/golden_set.json")
REJECTED_PATH = Path("data/golden_rejected.json")


def load_progress():
    kept = json.load(open(GOLDEN_SET_PATH)) if GOLDEN_SET_PATH.exists() else []
    rejected = json.load(open(REJECTED_PATH)) if REJECTED_PATH.exists() else []
    decided_ids = {(r["gold"]["doc_id"], r["gold"]["start"], r["gold"]["end"]) for r in kept + rejected}
    return kept, rejected, decided_ids


def save(kept, rejected):
    with open(GOLDEN_SET_PATH, "w") as f:
        json.dump(kept, f, indent=2)
    with open(REJECTED_PATH, "w") as f:
        json.dump(rejected, f, indent=2)


def shares_long_phrase(q, passage, n=6):
    q_words = re.findall(r"\w+", q.lower())
    p_text = " ".join(re.findall(r"\w+", passage.lower()))
    for i in range(len(q_words) - n + 1):
        if " ".join(q_words[i:i+n]) in p_text:
            return True
    return False


def auto_check(c, passage):
    """Returns a rejection reason string, or None if it passes automatic checks."""
    q = c["query"]
    if "[***]" in passage or "[ * ]" in passage:
        return "redacted"
    if len(q.split()) < 5 or not q.rstrip().endswith("?"):
        return "too_short_or_malformed"
    if shares_long_phrase(q, passage):
        return "verbatim_overlap"
    return None


def chunk_matches_gold(chunk, gold, tolerance=50):
    """A retrieved chunk 'matches' the gold span if they substantially overlap."""
    return (chunk.doc_id == gold["doc_id"]
            and chunk.start < gold["end"] + tolerance
            and chunk.end > gold["start"] - tolerance)


def main():
    candidates = json.load(open(CANDIDATES_PATH))
    docs = load_docs()
    kept, rejected, decided_ids = load_progress()

    remaining = [c for c in candidates
                 if (c["gold"]["doc_id"], c["gold"]["start"], c["gold"]["end"]) not in decided_ids]

    print("Building retriever for ambiguity checks (one-time setup)...")
    all_chunks = chunk_all(docs, chunk_size=1500, overlap=0)
    retriever = DenseRetriever(all_chunks)

    auto_rejected_count = 0
    needs_review = []

    for c in remaining:
        gold = c["gold"]
        passage = docs[gold["doc_id"]][gold["start"]:gold["end"]]
        reason = auto_check(c, passage)
        if reason:
            c["_reject_reason"] = reason
            rejected.append(c)
            auto_rejected_count += 1
        else:
            needs_review.append((c, passage))

    save(kept, rejected)
    print(f"Auto-rejected {auto_rejected_count} (redacted / malformed / verbatim).")
    print(f"{len(needs_review)} need your judgment.\n")

    for i, (c, passage) in enumerate(needs_review):
        gold = c["gold"]
        results = retriever.search(c["query"], k=5)

        print("=" * 80)
        print(f"[{i+1}/{len(needs_review)}]  doc={gold['doc_id']}  category={c['category']}")
        print("-" * 80)
        print("QUESTION:", c["query"])
        print("-" * 80)
        print("FULL PASSAGE:")
        print(passage.strip())
        print("-" * 80)
        print("RETRIEVER CHECK (top 5 results for this question):")
        for rank, (chunk, score) in enumerate(results, 1):
            is_gold = chunk_matches_gold(chunk, gold)
            marker = "✅ GOLD" if is_gold else "  "
            print(f"  {rank}. {marker} {chunk.doc_id}  score={score:.3f}  {chunk.text[:80]!r}")
        print("=" * 80)

        while True:
            choice = input("[k]eep / [r]eject / [e]dit question / [q]uit-and-save: ").strip().lower()
            if choice == "k":
                kept.append(c)
                break
            elif choice == "r":
                reason = input("  reason (optional): ").strip()
                c["_reject_reason"] = reason or "manual"
                rejected.append(c)
                break
            elif choice == "e":
                new_q = input("  new question text: ").strip()
                if new_q:
                    c["query"] = new_q
                    kept.append(c)
                    break
            elif choice == "q":
                save(kept, rejected)
                print(f"\nSaved. {len(kept)} kept, {len(rejected)} rejected so far.")
                return
            else:
                print("  please enter k, r, e, or q")

        if len(kept) % 10 == 0 and choice in ("k", "e"):
            save(kept, rejected)

    save(kept, rejected)
    print(f"\nDone. {len(kept)} kept, {len(rejected)} rejected.")


if __name__ == "__main__":
    main()