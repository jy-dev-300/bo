# Personal Search + Grounded RAG

A local-first learning project for finding files from vague memories and, eventually, answering questions with inspectable evidence. The first pass deliberately ships only a lexical-search vertical slice. The most educational retrieval and RAG functions are left for the student and are named in `STUDENT_TASKS.md`.

## What works now

- A FastAPI service with `/health` and `/api/search`.
- An explicit BM25 implementation over pre-chunked sample data.
- Typed models for documents, chunks, candidates, evidence, and answers.
- PostgreSQL + pgvector schema and Docker Compose infrastructure.
- Pluggable interfaces for parsers, OCR, embeddings, vector search, reranking, and generation.
- A small React/TypeScript search interface.
- Contract fixtures, a gold retrieval dataset, and tests.

The API intentionally does **not** yet ingest arbitrary files or answer RAG questions. Those paths depend on student-owned implementations and must not silently fall back to sending whole documents to a model.

## Run locally

Prerequisites: Python 3.12, Node 20+, and optionally Docker Desktop.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn backend.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The default API reads `sample_corpus/chunks.jsonl`, so PostgreSQL is not required for the baseline. To start the full infrastructure instead:

```powershell
docker compose up --build
```

## Measured retrieval progression

Set `RETRIEVAL_MODE` to one stage at a time in `.env`:

1. `bm25` (default): exact-word search only; no model download.
2. `hybrid`: add BGE-M3 dense, learned-sparse, and ColBERT-style candidate lists; combine them with RRF.
3. `rerank`: add the BGE cross-encoder and diversity selection to `hybrid`.
4. `specialists`: add a separate English-only SPLADE++ candidate list to `rerank`.

`advanced` remains an alias for `specialists` for existing configurations.
Model-backed stages load weights on first search and need several gigabytes of
local storage. Compare stages on the labeled queries before advancing:

```powershell
python -m evaluation.compare_retrieval --stages bm25 --k 3
python -m evaluation.compare_retrieval --stages bm25 hybrid rerank --k 3
python -m evaluation.compare_retrieval --stages bm25 hybrid rerank specialists --k 3
```

Only the stages named in a command run; commands containing `hybrid`, `rerank`,
or `specialists` may download model weights. The report shows each query's
hits, Recall@k, MRR@k, first-query time, and warm-query median time. Keep an
additional stage only if it fixes misses or ranking errors worth its cost.
The bundled gold set has only three easy cases and BM25 currently scores
perfectly at `k=3`; add harder labeled queries before using it to judge the
model-backed stages.
This is an in-memory reference index over `sample_corpus/chunks.jsonl`, not a
persistent or production-scale vector/sparse/ColBERT index. Real-model quality,
latency, and memory use have not yet been measured.

## Verify

```powershell
pytest
ruff check .
```

## Repository map

- `backend/`: HTTP boundary, configuration, and persistence schema.
- `ingestion/`: parser/OCR contracts, normalized models, and the student chunker.
- `retrieval/`: BM25, multi-signal advanced ranking, and provider contracts.
- `rag/`: implemented candidate reranking/context selection, with generation and citation boundaries.
- `evaluation/`: gold-set loading plus student retrieval/RAG metrics.
- `frontend/`: minimal retrieval UI.
- `sample_corpus/`: deterministic, already-chunked fixtures and gold queries.
- `tests/`: working-baseline tests and skipped student acceptance tests.

Read `ARCHITECTURE.md` before writing code. Student Implementation A is complete; continue with
Student Implementation H in `STUDENT_TASKS.md`.
