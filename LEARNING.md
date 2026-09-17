# Learning Log

Notes are grouped by the A-I assignments in `STUDENT_TASKS.md`; cross-cutting study habits are at the end.

## A. Document-to-chunk transformation

- A Pydantic model validates required fields when it is created, so gather field values before constructing the object.
- `chunk_document` transforms one normalized document into an ordered list of searchable, provenance-rich chunks.
- A deterministic chunk ID produces the same value whenever the same document and policy are processed again.
- A chunk needs enough context to make sense when retrieved, but it does not need to be a complete standalone document.
- Document structure chooses sensible chunk boundaries, while size acts as a budget that prevents unbounded chunks.
- Characters are exact written symbols, while model-dependent tokens may be words, word pieces, punctuation, or spaces.
- Sentence-aware chunking packs complete sentences toward a target size instead of cutting blindly at an exact position.
- An oversized sentence can be divided into size-bounded windows with overlap rather than repeatedly split in half.
- Overlap repeats a limited amount of context between neighboring chunks so boundary-crossing ideas are not lost.
- Five hundred tokens is roughly 350-400 English words or a few ordinary paragraphs, but the visual length varies for code and different languages.
- A false sentence boundary may recombine harmlessly, but it can still change chunk packing or overlap and separate a phrase at a chunk edge.
- To retain sentence punctuation, split on the whitespace after `.`, `?`, or `!` instead of splitting on the punctuation itself.
- The scaffold's BM25 tokenizer finds searchable terms, but it is not a model tokenizer and cannot provide exact model-token counts.
- Oversized text needs no repeated halving: use fixed windows whose step is `target_size - overlap` so adjacent windows share the requested units.
- A slice from `start` to `start + target_size` can contain at most `target_size` units, and Python safely returns a shorter final slice.
- Preserve provenance during every text transformation because reducing blocks to bare strings loses the source locations needed by final chunks and citations.
- A chunk ID establishes identity, while the numeric `ordinal` field establishes order; formatting the ordinal inside the ID is only a convenience.
- Treat input configuration as a caller-owned contract and avoid silently mutating it inside a function, because hidden side effects make behavior difficult to predict and test.
- A non-text file such as a JPG must first be converted by OCR or vision into provenance-rich normalized text blocks, after which the generic chunker processes the extracted text rather than the image bytes.
- Code and prose need different chunk boundaries: structural parsing can preserve functions/classes, while unknown code can fall back to line boundaries.

## B. Hybrid retrieval

- BM25 finds exact terms and identifiers; dense embeddings can find semantically similar wording, so their candidate lists complement each other.
- Reciprocal Rank Fusion combines incomparable retrieval systems using rank positions rather than raw scores, summing `1 / (k + rank)` for each document across the ranked lists.
- BGE-M3 provides dense, learned-sparse, and ColBERT-style multi-vector signals; SPLADE++ is a separate learned-sparse retriever, not another name for BGE-M3 sparse output.
- ColBERT-style late interaction compares query and document token vectors, preserving matching detail that one document vector can lose.
- Fusion must deduplicate by chunk ID and retain each retriever's rank/score as inspectable provenance; correlated signals should be judged by evaluation, not assumed to help.
- RRF cannot recover a relevant chunk that none of its input retrievers found.

## C. Reranking and context selection

- A cross-encoder scores the query and each candidate together, so apply it to a retrieved shortlist rather than every document.
- Reranking can reorder the candidate pool but cannot find evidence missing from that pool.
- Diversity selection trades a little raw relevance for less redundant evidence; context selection then enforces a budget while retaining provenance.

## D. Grounded citation assembly

- Retrieval and reranking must carry chunk IDs and source locations forward so a later citation can point to actual evidence.

## E. Retrieval evaluation

- Measure candidate Recall@k before judging final order with MRR or nDCG; compare BM25, hybrid, and reranked variants on the same labeled queries.
- A sophisticated model stack is not proof of better search: measure quality alongside latency and memory use before keeping each additional stage.

## F. RAG evaluation

- No notes yet.

## G. Query processing

- Keep the original query while adding deterministic variants for identifiers or paths; expansion should not erase exact-match intent.
- Apply metadata filters before candidate limits where possible, or filtering can discard the entire useful shortlist.

## H. Index lifecycle and updates

- Document embeddings are derived index data: cache them for repeated queries, but rebuild them when chunks or the embedding model changes.
- Lazy model loading keeps ordinary startup light, but the first advanced search must load/download weights and may be slow.
- Scoring every in-memory chunk is a reference implementation, not a scalable index; larger corpora need persistent vector, sparse, and multi-vector indexing.

## I. End-to-end failure diagnosis

- Inspect the first failing stage: missing candidates are a recall problem, while badly ordered present candidates are a ranking problem.

## Across tasks: working approach

- Keep normal syntax and type autocomplete on, but disable AI-generated code completion while first attempting the project's core reasoning functions.
- Boilerplate is predictable wiring such as imports, constructing a `Chunk` from already-decided values, or serializing results; boundary and grouping decisions are algorithmic.
- When syntax and system design overload each other, isolate one concept in a tiny executable exercise before integrating it into the full pipeline.
