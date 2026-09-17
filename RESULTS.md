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

