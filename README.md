# Bo — Local-First Search + Grounded RAG

A local-first application for finding files from vague memories and answering questions with inspectable evidence. Its retrieval pipeline keeps exact-match, semantic, fusion, reranking, and citation stages explicit so their quality and cost can be measured independently.

## What works now

- A FastAPI service with `/health` and `/api/search`.
- An explicit BM25 implementation over pre-chunked sample data.
- Typed models for documents, chunks, candidates, evidence, and answers.
- PostgreSQL + pgvector schema and Docker Compose infrastructure.
- Pluggable interfaces for parsers, OCR, embeddings, vector search, reranking, and generation.
- A small React/TypeScript search interface.
- Contract fixtures, a gold retrieval dataset, and tests.

The API does **not** silently fall back to sending whole documents to a model when a pipeline stage is unavailable.

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
2. `bge_dense`: BGE-M3 dense-vector retrieval alone, exposed for baseline testing.
3. `hybrid`: add BGE-M3 dense, learned-sparse, and ColBERT-style candidate lists; combine them with RRF.
4. `rerank` / `rerank_bge`: add the BGE cross-encoder and diversity selection to `hybrid`.
5. `rerank_jev`: apply TypeSafe Jev to the same `hybrid` shortlist and diversity policy.
6. `specialists`: add a separate English-only SPLADE++ candidate list to `rerank`.

`advanced` remains an alias for `specialists` for existing configurations.
Model-backed stages load weights on first search and need several gigabytes of
local storage. Compare stages on the labeled queries before advancing:

```powershell
python -m evaluation.compare_retrieval --stages bm25 --k 3
python -m evaluation.compare_retrieval --stages bm25 hybrid rerank --k 3
python -m evaluation.compare_retrieval --stages rerank_bge rerank_jev --k 3
python -m evaluation.compare_retrieval --stages bm25 hybrid rerank specialists --k 3
```

The Jev comparison requires `TYPESAFE_API_KEY`. Both reranker stages use the same
BGE-M3 hybrid candidate generator; only the final reranker changes. Compare their
per-query ranks and MRR on harder labeled queries, since the bundled three-query set
already has no useful headroom at `k=3`.

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
- `ingestion/`: parser/OCR contracts, normalized models, and provenance-aware chunking.
- `retrieval/`: BM25, multi-signal advanced ranking, and provider contracts.
- `rag/`: implemented candidate reranking/context selection, with generation and citation boundaries.
- `evaluation/`: gold-set loading plus retrieval and RAG metrics.
- `frontend/`: minimal retrieval UI.
- `sample_corpus/`: deterministic, already-chunked fixtures and gold queries.
- `tests/`: unit, contract, and pipeline tests.

Read `ARCHITECTURE.md` for the pipeline boundaries, data contracts, and known failure modes.
