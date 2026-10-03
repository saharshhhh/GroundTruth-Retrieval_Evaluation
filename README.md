# Groundtruth Retrieval Evaluation

A systematic evaluation of three retrieval configurations — dense, hybrid, and hybrid with reranking — against a human-reviewed golden set of legal-contract questions. Built to answer one question with numbers instead of intuition: **did a change to the retrieval system actually make it better?**

## Why this exists

Most RAG demos show a handful of queries that happen to work. This project instead builds a fixed, human-validated evaluation set, measures three retrieval configurations against it with standard IR metrics, and wires the result into a CI gate that automatically fails a pull request if retrieval quality regresses — demonstrated live with a deliberately broken PR (see [CI Regression Gate](#ci-regression-gate) below).

## Corpus

- **Source:** [LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag) contract subset (CUAD-derived), real commercial contracts (consulting agreements, licensing deals, distribution agreements, M&A agreements, etc.)
- **23 documents**, chunked into ~1,600 passages at 1,500 characters per chunk (no overlap)
- Chunk boundaries are stored as **character offsets into the original document**, not fixed indices — this means the golden set survives changes to chunk size, since a retrieved chunk is scored as correct if it *overlaps* the gold span, not if it exactly matches a specific chunk ID

## Golden set construction

1. **150 candidate questions** were drafted by sampling real passages (weighted by document length) and prompting an LLM to (a) classify the passage's category and (b) write a realistic lawyer question answered by that passage alone.
2. **Automated filtering** removed candidates with redacted content (`[***]` sitting where the answer would be), malformed/truncated questions, or questions that lifted 6+ word phrases verbatim from the source passage.
3. **Human review**, assisted by running each surviving candidate through the dense retriever and inspecting the top 5 results: if a genuine rival passage (not just a topically-similar one) outranked or competed with the intended gold passage, the question was edited to anchor it to contract-specific details (party names, section numbers, specific figures) rather than rejected outright — preserving category coverage rather than discarding every hard case.
4. **Result:** 91 questions kept, 59 rejected, across 6 categories.

| Category | Count |
|---|---|
| Liability | 21 |
| Termination | 20 |
| Intellectual Property | 19 |
| Governing Law | 14 |
| Payment | 9 |
| Confidentiality | 8 |

**Note:** payment and confidentiality are comparatively small samples; per-category numbers for those two categories should be read as directional, not precise.

### A recurring finding from review

A large share of rejected/edited candidates failed not because the question was bad, but because the underlying clause was **boilerplate language that recurs near-verbatim across multiple contracts** (warranty disclaimers, termination-for-breach notice periods, generic indemnification language). Questions built from these clauses were ambiguous against the corpus as a whole, since a different contract's version of the same boilerplate could equally "answer" them. Where possible, these were rescued by anchoring the question to contract-specific details (party names, section numbers) rather than discarded, to preserve category balance.

## Retrieval configurations

| Config | Description |
|---|---|
| **Dense** | `bge-small-en-v1.5` embeddings, cosine similarity over all chunks |
| **Hybrid** | Dense + BM25 keyword search, merged via Reciprocal Rank Fusion (RRF, k=60), pooling top 50 from each before fusion |
| **Hybrid + Reranker** | Hybrid's top 50 re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), top 10 returned |

## Metrics

- **Recall@10** — was the correct passage anywhere in the top 10?
- **MRR** — how high did it rank, on average (rewards finding it early)?
- **nDCG@10** — binary-relevance special case (one correct passage per query), normalized against the ideal rank-1 case.

## Results

| Config | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Dense | 0.934 | 0.781 | 0.819 |
| **Hybrid** | **0.956** | **0.840** | **0.868** |
| Hybrid + Reranker | 0.945 | 0.799 | 0.836 |

n = 91 queries.

### The reranker underperformed hybrid — and that's a real finding, not a bug

This contradicts the naive assumption that adding a reranker always helps. Per-query analysis showed:

- **16 of 91 queries** got a worse rank after reranking, but the overwhelming majority were **minor reshuffles among already-correct top candidates** (e.g. rank 1 → rank 2), not genuine failures.
- **Only 1 query** dropped out of the top 10 entirely after reranking. Token-length analysis ruled out truncation as the cause (combined query + passage length was well within the reranker's input limit).

**Conclusion:** the off-the-shelf `ms-marco-MiniLM-L-6-v2` reranker, trained on general web search data (MS MARCO), does not transfer cleanly to legal contract language. It made small, mostly low-stakes reordering decisions that net-negative affected rank-sensitive metrics (MRR, nDCG) without meaningfully helping Recall@10. A domain-tuned or legal-specific reranker would likely be needed to realize the gains the technique promises in general-purpose search contexts.

## Error analysis

With Hybrid (the best-performing config), 4 queries failed to retrieve the correct passage in the top 10, and 4 more ranked it only 6–10. Diagnosed causes cluster into two mechanisms:

1. **Numeric/structured content underrepresented by both signals** — e.g. a question about dollar-amount thresholds that trigger default, where the passage is essentially a list of figures. Neither dense embeddings nor BM25 represent numeric specifics well.
2. **Boilerplate ambiguity** — the same root cause identified during golden-set review. Clauses using common legal phrasing (automatic termination, attorneys' fees, jurisdiction) recur across multiple contracts, diluting the retrieval signal for any single correct instance even when the question itself is well-formed.

One flagged question (`[party names]` placeholder left unfilled from an earlier edit pass) was caught during this analysis and excluded as a data-quality issue rather than counted as a retrieval failure — a reminder that error analysis also catches mistakes in the evaluation harness itself, not just the system under test.

## CI Regression Gate

`tests/test_regression.py` runs the Hybrid configuration against the full golden set and fails the build if Recall@10 drops by more than 1 percentage point from the saved baseline (`reports/baseline.json`).

```
baseline = 0.9560
current  = 0.8571
drop     = 0.0989 (threshold: 0.01)
```

**Demonstrated live:** PR `break-retrieval-demo` deliberately reduced `HybridRetriever`'s `pool_size` parameter (first to 5, then to 1) to verify the gate fires correctly.

- `pool_size=5` — Recall@10 stayed within threshold; CI passed. (A useful secondary finding: this corpus's hybrid search is fairly robust to a smaller fusion pool.)
- `pool_size=1` — Recall@10 collapsed well past the 1-point threshold; **CI correctly failed the build.**

The PR was closed without merging, preserving the CI failure history as a permanent artifact. [https://github.com/saharshhhh/Ground_Truth/pull/1]

## Project structure

```
GroundTruth/
├── data/
│   ├── golden_set.json          # 91 reviewed questions, the answer key
│   ├── golden_rejected.json     # 59 rejected candidates + reasons
│   └── docs/                    # corpus (23 contracts, .txt)
├── src/
│   ├── chunking.py
│   ├── retrieval.py             # dense retriever
│   ├── hybrid_search.py         # BM25 + RRF fusion
│   └── reranker.py              # cross-encoder reranking
├── evaluation/
│   └── retrieval_metrics.py     # Recall@10, MRR, nDCG@10
├── tests/
│   └── test_regression.py       # CI regression gate
├── reports/
│   ├── baseline.json
│   └── results.json
├── .github/workflows/evaluation.yml
├── run_comparison.py            # runs all 3 configs, writes results.json
├── draft_golden_candidates.py   # LLM-assisted golden-set drafting
├── review_golden.py             # human review tool (retriever-assisted)
└── error_analysis.py
```

## Running it yourself

```bash
pip install -r requirements.txt
python run_comparison.py          # evaluate all 3 configs
python -m pytest tests/test_regression.py -v   # run the CI gate locally
```

## Limitations

- Corpus size (23 documents) is modest; a larger corpus would increase distractor density and likely widen the gap between configurations.
- Payment and confidentiality categories have fewer golden-set entries (8–9) than the others; category-level numbers for those two should be read with that in mind.
- The reranker result is specific to this model and domain — it is not a general claim that reranking doesn't help, only that this off-the-shelf model didn't help *here*.
- Generation-layer evaluation (faithfulness/answer relevance of an LLM's final answer, as opposed to retrieval quality) is a planned extension, not yet implemented.
