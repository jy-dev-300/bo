# Student Tasks

Implement these in order. Replace each `NotImplementedError`, enable its marked tests, and record design choices rather than optimizing only for a green test suite.

## A. Document-to-chunk transformation (implemented)

**Why it matters / category:** Ingestion and data preparation — turns source documents into reliable, traceable units that every later retrieval and citation step depends on.

**File:** `ingestion/chunking.py`  
**Production function:** `chunk_document(document, policy)`

### Current production implementation

The manual regular-expression packer has been replaced by a strategy-driven pipeline:

- `strategy="sentence"` uses Chonkie's `SentenceChunker` for sentence boundaries, exact
  character or declared-tokenizer budgets, and one-directional overlap;
- `strategy="semantic"` uses Chonkie's `SemanticChunker` with local embeddings to detect
  topic changes (token mode only; the model is downloaded on first use);
- code blocks continue through Tree-sitter AST boundaries, with Chonkie's `TokenChunker`
  enforcing the final hard budget when an AST unit is too large;
- headings, pages, and sections are grouped before chunking, and every library-produced span
  is mapped back to the original `SourceLocation` objects;
- IDs hash the document checksum, full policy, ordinal, text, and locations, so content or
  policy changes cannot silently reuse an old chunk identity.

Examples:

```python
# Fast, deterministic, offline-safe character chunks.
ChunkingPolicy(target_size=800, overlap=80, strategy="sentence")

# Model-token-aware chunks; replace "word" with the exact downstream tokenizer name.
ChunkingPolicy(
    target_size=384,
    overlap=48,
    size_unit="tokens",
    tokenizer="word",
    strategy="sentence",
)

# Topic-shift boundaries using the embedding model's tokenizer.
ChunkingPolicy(
    target_size=384,
    overlap=48,
    size_unit="tokens",
    strategy="semantic",
    semantic_embedding_model="minishlab/potion-base-32M",
)
```

Docling's `HybridChunker` remains a good option at the upstream parsing layer when the input is
a `DoclingDocument`; converting these already-normalized blocks into a second document model
would duplicate parsing and weaken provenance. Late chunking is reserved for the future embedding
stage because it pools token embeddings with full-document context rather than merely choosing
text boundaries.

### Exact goal

Transform one already-normalized `NormalizedDocument` into an ordered `list[Chunk]`. The function does **not** open raw TXT, Markdown, PDF, image, or source-code files. An upstream parser or OCR adapter must first convert a raw file into `NormalizedBlock` objects. This assignment begins after that conversion.

Each input block contains three important things:

- `kind`: `heading`, `paragraph`, `code`, or `image_text`;
- `text`: the extracted text to chunk;
- `location`: the source path and any available page, section, line, or character offsets.

Each output chunk is a searchable evidence unit and a future citation target. It must contain useful text, stable identity, ordering, and enough source location information for a user to find the evidence again.

### Required behavior by block kind

| Block kind | Meaning | Required baseline behavior |
| --- | --- | --- |
| `heading` | A Markdown heading, document title, or parsed section heading | Keep it with compatible following prose when it fits; do not leave a meaningless heading-only chunk unless no content follows. |
| `paragraph` | Ordinary prose from TXT, Markdown, PDF text extraction, and similar sources | Split at sentence boundaries, pack complete sentences toward the size limit, and split an individually oversized sentence into bounded windows. |
| `image_text` | Text already extracted from an image by OCR or vision | Treat the extracted text like prose while preserving its image/OCR provenance; this function does not perform OCR itself. |
| `code` | Literal computer program source such as Python, JavaScript, SQL, or a fenced Markdown code block | When `preserve_code_blocks=True`, do not use prose punctuation as boundaries. Keep the block intact when it fits; if it is oversized, prefer line boundaries and preserve line ranges. |

### Required size policy

In character mode, `target_size` and `overlap` count characters. In token mode, they use the
tokenizer named by `policy.tokenizer`; the offline-safe default is `word`, while a Hugging Face
tokenizer name can be supplied to match the downstream embedding or generation model.

`target_size` is a maximum final chunk budget, including overlap. A chunk must not exceed it. Empty or whitespace-only chunks are forbidden.

### Required overlap behavior

Adjacent final chunks should share up to `policy.overlap` characters. Think of the next chunk as beginning with the tail of the previous chunk:

```text
chunk 0: [new evidence ----------------][shared tail]
chunk 1:                              [shared tail][new evidence ----------------]
```

The shared text is counted in both chunks' size budgets. `overlap=0` must produce no repeated boundary text. A middle chunk naturally overlaps with the previous chunk at its beginning, while its tail can become the next chunk's beginning; do not independently add a full overlap from both neighbors and accidentally double the requested overlap.

### Required provenance

Every `Chunk` must include:

- `document_id`: the source document ID;
- `ordinal`: its zero-based position within that document;
- `id`: deterministic for the same document, content, policy, and order;
- `metadata.source_path` and `metadata.media_type`;
- every available page, section, line, and character location that still describes the chunk.

Do not flatten blocks into bare strings and lose their locations. Carry text together with its `SourceLocation` while splitting and packing. Only combine blocks when the resulting chunk can still be represented by a truthful continuous location. If two blocks have incompatible pages or sections, finish the current chunk before crossing that boundary.

### Minimum implementation sequence

1. Return `[]` for a document with no non-empty block text.
2. Walk through blocks in source order while retaining each block's kind and location.
3. Apply the correct boundary strategy for prose/image text versus program code.
4. Pack resulting units into base chunks without exceeding the space reserved for overlap.
5. Add overlap so adjacent chunks share no more than the requested amount and final chunks remain within `target_size`.
6. Build `Chunk` objects with deterministic IDs, sequential ordinals, non-empty text, and preserved provenance.
7. Return chunks in ordinal order without mutating the caller's document or policy.

### Definition of done and required tests

Task A's focused tests now demonstrate all of the following:

- An empty document and whitespace-only blocks return `[]`.
- Running the same document and policy twice produces identical chunk IDs, text, and ordinals.
- Every chunk is non-empty and no chunk exceeds `target_size` in the declared size unit.
- `overlap=0` repeats nothing, and a non-zero overlap produces the expected shared boundary text without exceeding the size limit.
- A heading remains associated with compatible following prose.
- Page, section, line, and character provenance survives when provided by the input blocks.
- A prose block uses sentence-aware boundaries.
- A `code` block is not split on prose punctuation when `preserve_code_blocks=True`.
- Character mode and declared-tokenizer mode are both tested explicitly.
- The Task A `xfail` marker has been removed; Task B remains intentionally xfailed.

### Not part of Task A

- Opening or parsing raw TXT, Markdown, PDF, image, or source-code files;
- running OCR or a vision model;
- generating embeddings or writing to the lexical/vector indexes;
- solving every abbreviation or language-specific sentence-boundary edge case;
- implementing model-accurate token counting when using the character-mode baseline.

## B. Hybrid retrieval

**Why it matters / category:** Information retrieval — combines exact-word and semantic search so the system can find evidence across different kinds of queries.

**File:** `retrieval/hybrid.py`  
**Function:** `fuse_candidates(...)`

### Concept in plain English

The system has two complementary ways to search. **Lexical retrieval** finds matching words, so it is excellent for exact filenames, addresses, error messages, and code identifiers. **Vector retrieval** compares meaning, so it can match “tries the request again” with source text containing “retry.”

Each retriever returns an ordered list, but its scores mean different things. A BM25 score of `8.2` and a vector similarity of `0.81` are not measurements on the same scale. Hybrid retrieval combines the ranked lists into one candidate list without pretending those raw numbers are directly comparable.

One common method is **Reciprocal Rank Fusion**: a result earns points based on its position in each list. A chunk appearing near the top of both lists becomes especially strong. You still need policies for chunks appearing in only one list, duplicate chunk IDs, ties, weighting, and deterministic ordering.

The naive mistake is adding BM25 and vector scores together. Whichever system happens to produce larger numbers can dominate for reasons unrelated to relevance.

BM25 scores and vector similarities have different scales. Study Reciprocal Rank Fusion (RRF), weighted normalized scores, missing candidates, ties, and deterministic ordering. Implement one declared method; do not add raw scores together without calibration.

## C. Reranking and context selection

**Why it matters / category:** Relevance ranking and context management — chooses the strongest non-redundant evidence that fits within the model's limited context window.

**File:** `rag/selection.py`  
**Functions:** `rerank_candidates(...)`, `select_context(...)`

### Concept in plain English

Initial retrieval is designed to search a large collection quickly, so its top results are only rough candidates. A **reranker** looks more carefully at the query and each candidate, then places the most relevant evidence first. It is slower, which is why it runs on a small candidate set rather than the whole corpus.

The generator also has a limited **context budget**—the amount of evidence it can read in one request. Context selection decides which reranked chunks deserve that space. It must avoid spending the budget on repeated or heavily overlapping passages and may need to balance evidence from different documents.

Think of retrieval as finding twenty potentially useful library books, reranking as inspecting their tables of contents, and context selection as choosing the few pages you can photocopy for an open-book exam.

The naive mistake is sending every retrieved chunk to the model. More context can add duplicates, contradictions, distraction, cost, and latency while making the final answer worse.

Use the provided `Reranker` protocol. Rank candidates, collapse duplicates/overlap, enforce the budget, retain provenance, and choose a deterministic policy for source diversity. Do not let the generator decide which evidence fits.

## D. Grounded citation assembly

**Why it matters / category:** Grounding and provenance — connects answer claims to verified source passages so users can inspect and trust the result.

**File:** `rag/citations.py`  
**Function:** `assemble_citations(...)`

### Concept in plain English

A citation is more than a document displayed beneath an answer. It is a claim that a specific piece of evidence supports a specific statement. Citation assembly connects answer claims to the chunks actually selected as evidence and turns their stored provenance into inspectable locations such as a path, page, section, or line range.

The generator may be asked to reference chunk IDs, but generated text is not automatically trustworthy. The assembly layer must verify that every cited ID exists in the selected evidence and that the referenced passage really supports the associated claim. If no evidence supports a claim, the system should mark it unsupported or abstain.

Think of this stage as checking footnotes before publishing an essay: opening every source, confirming the page, and ensuring it says what the sentence claims.

The naive mistake is treating any plausible-looking source ID emitted by the model as a valid citation.

Map claims to selected evidence and validate every cited chunk ID. Expose document path plus page/section/line where available. Decide how to represent an unsupported claim; never invent a citation because a model emitted a plausible ID.

## E. Retrieval evaluation (baseline metrics implemented)

**Why it matters / category:** Search evaluation — measures whether the retrieval system consistently finds known relevant evidence and reveals where it misses.

**File:** `evaluation/retrieval_metrics.py`  
**Functions:** `recall_at_k(...)`, `reciprocal_rank(...)`, `evaluate_retrieval(...)`

### Concept in plain English

Retrieval evaluation asks a narrow question: **did search find the evidence we already know is relevant?** A gold dataset contains test queries and the document or chunk IDs that a human marked as correct. The retriever runs each query, and its ranked results are compared with those labels.

**Recall@k** measures how much of the known relevant evidence appeared in the first `k` results. If two chunks are relevant and the first ten results contain one, Recall@10 is `0.5`. **Reciprocal Rank** rewards placing the first relevant result near the top: rank 1 scores `1`, rank 2 scores `0.5`, and rank 10 scores `0.1`. Mean Reciprocal Rank averages that value across queries.

Metrics summarize behavior, but failure analysis explains it. Inspect individual misses and group them by causes such as OCR errors, bad chunk boundaries, missing exact terms, metadata filters, or semantic mismatch.

The naive mistake is judging retrieval by reading one impressive answer. A generator can hide poor retrieval, and one successful demo says little about repeatable search quality.

Use `sample_corpus/retrieval_gold.jsonl`. Measure whether relevant evidence was found, not whether the answer sounded good. Report per-query results and failures by file type/query type, then compare BM25, vector, and hybrid.

The baseline `Recall@k`, reciprocal-rank, and per-query evaluator are implemented. Use `python -m evaluation.compare_retrieval --stages bm25 --k 3` first; only add model-backed stages after reviewing the baseline misses. Expanding the gold set and doing failure analysis remain student work.

## F. RAG evaluation

**Why it matters / category:** End-to-end quality evaluation — separates retrieval, evidence use, faithfulness, and citation quality so each failure can be fixed at the right layer.

**File:** `evaluation/rag_metrics.py`  
**Function:** `evaluate_rag_case(...)`

### Concept in plain English

RAG evaluation asks what happened after a user asked a question, but several different things can go wrong. Search might fail to find the answer. Search might succeed, but context selection might omit it. The model might receive good evidence and ignore or contradict it. The answer might be correct but cite the wrong chunk.

Those failures require different fixes, so this assignment records them separately:

- **Evidence retrieval:** Did the pipeline find the required source?
- **Evidence use:** Did the answer use the information that was provided?
- **Support or faithfulness:** Can the answer's claims be justified by the evidence?
- **Citation correctness:** Do citations point to passages supporting the associated claims?

Some checks can be deterministic, such as whether a required chunk ID was retrieved. More subjective judgments may use a human rubric or an LLM judge, but the prompt, model, rubric, and raw judgment must be retained so the result can be audited.

The naive mistake is producing one “RAG quality” score. A single number hides whether the problem lives in retrieval, generation, or citation handling.

Keep four judgments separate:

1. evidence retrieval success;
2. answer use of the evidence;
3. claim support/faithfulness;
4. citation correctness.

Prefer deterministic checks where possible and retain judge prompts/model versions when an LLM judge is genuinely needed. Never average the four fields into one unexplained score.

## G. Query processing

**Why it matters / category:** Query understanding — converts messy user language into an inspectable search plan without losing exact terms or intent.

**Suggested file:** `retrieval/query_processing.py`  
**Suggested functions:** `process_query(...)`, `derive_metadata_filters(...)`

### Concept in plain English

Users rarely phrase searches using the exact words stored in their files. Query processing turns a messy request into a clearer search plan. It may normalize harmless differences, create alternate phrasings, identify an exact filename, or infer a possible date or file-type filter.

For example, “that retry code from last March” contains a topic, a likely file type, and a date clue. The processor might retain the original query, add “retry attempts timeout,” and propose a March date filter. An inferred filter should remain visible and may need softer treatment than a filter the user explicitly requested.

Rewriting is risky because exact evidence can be accidentally removed. `ERR_CONNECTION_RESET`, a Korean street address, or `fetch_with_retries` should not disappear merely because a model produced a smoother paraphrase.

The naive mistake is replacing the user's query with one generated rewrite and assuming it must be better. Keep the original, trace every transformation, and measure both versions.

Rewrite or expand ambiguous queries without losing exact evidence such as filenames, addresses, dates, quoted phrases, or code identifiers. Decide when metadata filters should apply and distinguish user-specified constraints from inferred filters. Preserve the raw query alongside every rewritten form so behavior remains inspectable.

Compare raw and processed queries using the same gold cases. Record Recall@k, MRR, latency, and exact-term regressions; a rewrite that improves one vague query but destroys a filename match is not an unconditional improvement.

Acceptance questions:

- Is the original query always retained?
- Are rewrites and inferred filters visible in traces and evaluation output?
- Can metadata filters be applied consistently to lexical and vector retrieval?
- Does the processor preserve exact identifiers and quoted text?
- Can evaluation compare raw-query and processed-query results case by case?

## H. Index lifecycle and updates

**Why it matters / category:** Data and index operations — keeps searchable artifacts accurate, consistent, and recoverable as source files change or disappear.

**Suggested file:** `ingestion/indexing.py`  
**Suggested functions:** `index_document(...)`, `update_document(...)`, `delete_document(...)`

### Concept in plain English

An index is derived data that helps the system search quickly. One source file may produce a document row, many chunks, lexical index entries, embeddings, and vector entries. These pieces must describe the same current version of the file.

When a file is added, all required artifacts must be created. When it changes, obsolete chunks must disappear and affected embeddings must be recomputed. When it is deleted, nothing derived from it should remain searchable. A checksum identifies whether file contents changed; pipeline and model versions identify whether the same contents need new derived representations.

**Idempotent** means safely repeating an operation produces the same final state rather than duplicate data. A transaction makes a group of database changes succeed or fail together. For work involving external model calls, you may also need explicit processing states and retry behavior because one database transaction cannot cover every external operation.

The naive mistake is appending new chunks whenever a file changes. Old chunks then remain searchable and can support outdated or contradictory answers.

Implement incremental add, update, and delete behavior across the document store, chunks, lexical index, embeddings, and vector index. Use stable source identity, checksums, and pipeline/model versions to decide whether work can be skipped or must be recomputed.

An update must not leave stale chunks or embeddings searchable. A delete must remove every derived artifact belonging to the source. Define transaction and recovery behavior so partial failures are detectable instead of silently producing inconsistent indexes.

Acceptance questions:

- Is re-indexing an unchanged file idempotent?
- Does replacing a file remove chunks that no longer exist?
- Does deletion remove the document, chunks, lexical entries, and vectors?
- Does an embedding-model or chunking-policy change trigger the correct rebuild?
- Can interrupted indexing be detected and safely retried?

## I. End-to-end failure diagnosis

**Why it matters / category:** Observability and debugging — traces a bad result to the earliest failing pipeline stage and turns the fix into measurable regression protection.

**Primary artifact:** `RESULTS.md` failure log  
**Supporting artifact:** one reproducible failing query and focused regression test

### Concept in plain English

An end-to-end RAG failure is the visible symptom of a pipeline problem. The wrong answer does not automatically mean the language model is the problem. The correct text may have been lost during chunking, missed during retrieval, pushed down by reranking, excluded from context, misused during generation, or attached to the wrong citation.

Debugging means observing the output of each stage until you find the earliest point where expected behavior becomes wrong. If the relevant chunk never appeared in retrieval, changing the generation prompt cannot repair the root cause. If the chunk was present and the answer distorted it, retrieval is probably not the layer to change.

Start with a reproducible query and an expected result. Save intermediate candidates, ranks, selected context, answer claims, and citations. Form a hypothesis, change one responsible layer, rerun the same case, and use a relevant metric to show improvement without causing unrelated regressions.

The naive mistake is tweaking several stages until the demo looks right. That produces no reliable explanation of what failed or why the change helped.

Choose a query that returns the wrong result or produces an unsupported answer. Before changing code, classify the suspected failure layer:

```text
chunking -> retrieval -> reranking -> context -> generation -> citation
```

Inspect the intermediate artifacts at every boundary, fix only the responsible layer, and demonstrate improvement with the appropriate metric. Preserve the original failure as a regression case. Do not compensate for a retrieval failure with a more persuasive generation prompt.

Acceptance questions:

- Is the failure reproducible from a committed fixture or documented corpus version?
- Does the diagnosis cite concrete intermediate evidence rather than intuition?
- Is the fix applied to the layer that caused the failure?
- Is there a before/after retrieval, grounding, citation, or latency measurement?
- Does a regression test protect the corrected behavior?

## Suggested workflow

Recommended execution order: **A -> H -> G -> B -> C -> D -> E -> F -> I**. Index lifecycle follows chunking because it operates on chunks and derived indexes. Query processing precedes retrieval work because it defines what both retrievers receive. The debugging drill comes last because it tests the complete system.

For each task: state a hypothesis, add or enable focused tests, implement the smallest explicit function, inspect failures, and write the result or tradeoff in `RESULTS.md`. Avoid adding an orchestration framework around unfinished reasoning.
