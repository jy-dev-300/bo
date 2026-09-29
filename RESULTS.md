# Results and Ablations

Complete one row per controlled experiment. Change one primary variable at a time and retain raw per-query output.

## Environment

- Date / commit:
- Corpus version and size:
- Hardware:
- PostgreSQL / pgvector version:
- Parser and OCR versions:
- Embedding model and dimensionality:
- Reranker:
- Generator:

## Retrieval experiments

| Experiment | Variant | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms | Notes / failure classes |
|---|---|---:|---:|---:|---:|---:|---|
| Retriever | BM25 only | | | | | | |
| Retriever | Vector only | | | | | | |
| Retriever | Hybrid | | | | | | |
| Chunk size | small | | | | | | |
| Chunk size | large | | | | | | |
| Reranking | before | | | | | | |
| Reranking | after | | | | | | |

### Query-processing baseline - 2026-09-22

Command: `py -3.11 -m evaluation.compare_retrieval --stages bm25 --query-modes raw processed --k 3`

| Query mode | Recall@3 | MRR@3 | First query ms | Warm median ms | Result |
|---|---:|---:|---:|---:|---|
| Raw | 1.000 | 1.000 | 0.158 | 0.065 | All three bundled cases found their expected chunk first. |
| Processed | 0.667 | 0.667 | 0.142 | 0.091 | The explicit PDF filter exposed that `q-orange-chart` points to a `.txt` fixture. |

This is a fixture/label mismatch, not evidence that query processing is generally worse. The
processed path improved the Python case's precision by returning only the `.py` file, but the tiny
three-case set still has no basis for a broad quality claim. Correct the PDF fixture or its gold
wording, then add harder paraphrase, filename, date, and distractor cases before comparing again.

## RAG experiments

| Experiment | Variant | Evidence found | Evidence used | Claims supported | Citations correct | p95 ms | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Top-k | | | | | | | |
| Context budget | | | | | | | |
| Generator | | | | | | | |

## Failure log

| Query ID | Stage | Expected | Observed | Root-cause hypothesis | Evidence | Next experiment |
|---|---|---|---|---|---|---|

## Conclusion

State what changed, the measured tradeoff, important counterexamples, and the next decision. Do not call a variant “better” without naming the metric and cost.
