# Sample corpus

These synthetic fixtures make the baseline deterministic and safe to publish. `chunks.jsonl` is
a stable retrieval fixture; Student Implementation A is now implemented separately in
`ingestion/chunking.py` and can regenerate future indexed corpora through the ingestion pipeline.

- `documents/` contains the readable source fixtures.
- `chunks.jsonl` contains stable chunk IDs and provenance.
- `retrieval_gold.jsonl` maps vague-memory queries to relevant chunks.

When adding cases, include hard negatives and queries that distinguish exact lexical matching from semantic matching. Never include private user data in a committed corpus.
