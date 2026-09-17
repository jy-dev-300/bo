# Architecture

## Product boundary

The system keeps original files and derived indexes local. In release one, a user selects local files, the system derives searchable representations, and every result points back to a source location. RAG is allowed to use only selected chunks; a whole document is never smuggled into the prompt as a shortcut.

The code separates commodity adapters (PDF parsing, OCR engines, model APIs) from the reasoning-bearing pipeline. Provider choices can change without hiding chunking, ranking, context selection, or citation logic.

## Ingestion flow

```text
file
  -> parser / OCR
  -> normalized document
  -> chunks
  -> metadata
  +-> lexical index
  `-> embeddings -> vector index
```

1. **File:** preserves source identity, checksum, media type, timestamps, and a resolvable local path. A checksum makes re-ingestion idempotent.
2. **Parser/OCR:** extracts text and structural hints. Native text is preferred; OCR is used for pixels or image-only pages and records its engine/confidence.
3. **Normalized document:** converts unlike inputs into one typed representation while retaining pages, sections, code symbols, and image observations.
4. **Chunks:** create retrieval-sized evidence units with Chonkie sentence/semantic strategies and Tree-sitter code boundaries. Boundaries affect recall, precision, prompt cost, and whether a citation remains meaningful. Student Implementation A is complete.
5. **Metadata:** carries provenance and filterable facts (document ID, page, section, language, date, file type, offsets) through every later stage.
6. **Lexical index:** supports exact tokens, filenames, addresses, identifiers, and rare phrases that semantic similarity can miss.
7. **Embeddings/vector index:** supports conceptual or vague-memory matches even when query and source use different words.

Ingestion should commit document, chunks, and index state transactionally or mark partial work explicitly. Derived data must be reproducible from originals and versioned pipeline settings.

## Query flow

```text
query
  -> query processing
  +-> lexical retrieval --+
  `-> vector retrieval ----+-> candidate fusion
                              -> reranking
                              -> context selection
                              -> generation
                              -> citations
```

1. **Query processing:** validates input and may derive filters or alternate forms without discarding the original query.
2. **Lexical retrieval:** produces ranked exact-match candidates with scores and rank provenance.
3. **Vector retrieval:** produces ranked semantic candidates using the same chunk IDs.
4. **Candidate fusion:** combines incomparable score spaces using ranks or calibrated scores. This is Student Implementation B.
5. **Reranking:** spends more compute on a small candidate set to estimate query/chunk relevance. Part of Student Implementation C.
6. **Context selection:** deduplicates evidence and fills a declared token/character budget while preserving source diversity and citation metadata. Part of Student Implementation C.
7. **Generation:** instructs a pluggable LLM to answer only from the selected evidence and abstain when evidence is insufficient.
8. **Citations:** maps answer claims to stable chunk/source locations that a user can inspect. This is Student Implementation D; model-emitted citation strings alone are not trusted.

## Main data contracts

- `NormalizedDocument`: extracted content plus source-level provenance.
- `Chunk`: the atomic indexed and cited unit with stable ID and location metadata.
- `RetrievalCandidate`: a chunk plus per-retriever score/rank provenance.
- `SelectedEvidence`: a context position and budget cost tied to a candidate.
- `GroundedAnswer`: answer text, structured citations, and an abstention flag.

PostgreSQL is the durable system of record. `tsvector`/GIN supports production lexical search and pgvector supports vector search. The baseline in-memory BM25 index exists so retrieval behavior is inspectable and testable before persistence is wired end-to-end.

## Failure modes the design exposes

- Bad extraction cannot be repaired by a better retriever.
- Chunks that lose page or section provenance produce unverifiable citations.
- Vector-only search misses exact identifiers; lexical-only search misses paraphrases.
- Raw BM25 and cosine scores are not directly comparable.
- More top-k can reduce answer quality by adding distraction and duplicates.
- A fluent answer can still be unsupported even when retrieval recall is high.
- A valid source can still be attached to the wrong claim.

## Release-one sequence

1. Make ingestion deterministic and implement chunking.
2. Persist chunks; implement lexical and vector adapters.
3. Implement and evaluate hybrid fusion.
4. Add reranking and budgeted evidence selection.
5. Add grounded generation and verified citation assembly.
6. Run separate retrieval and RAG evaluations, then record ablations in `RESULTS.md`.
