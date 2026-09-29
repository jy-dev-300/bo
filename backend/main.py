import base64
import binascii
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.schemas import (
    LocalCorpusRequest,
    LocalCorpusResponse,
    LocalFileSkipRequest,
    SearchHitResponse,
    SearchResponse,
)
from ingestion.chunking import ChunkingPolicy
from ingestion.indexing import IndexingCancelled, IndexingError, PersistentDocumentIndex
from ingestion.local_files import normalize_local_file
from retrieval.advanced import BGEDenseEmbeddingProvider
from retrieval.corpus import load_chunks_jsonl
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.query_processing import process_query
from retrieval.stages import build_retriever

TestingStage = Literal["bm25", "bge_dense", "hybrid", "rerank_bge", "rerank_jev"]
LOCAL_CORPUS_ID = "local-bge"


def _testing_corpus(store: PersistentDocumentIndex) -> dict[str, object]:
    """Create one runtime handle around the persistent local corpus."""
    return {
        "store": store,
        "retrievers": {},
        "files_received": 0,
        "files_indexed": store.document_count,
        "skipped_files": [],
        "cancelled_files": set(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the configured corpus and retrieval stage when the API starts."""
    settings = get_settings()
    chunks = load_chunks_jsonl(settings.corpus_path)
    stage = "specialists" if settings.retrieval_mode == "advanced" else settings.retrieval_mode
    app.state.chunks = chunks
    app.state.retrieval_method = stage
    app.state.retriever = build_retriever(chunks, stage)
    app.state.retrievers = {stage: app.state.retriever}
    app.state.dense_embedding_provider = BGEDenseEmbeddingProvider()
    app.state.testing_store = PersistentDocumentIndex(
        settings.bge_file_index_dir,
        settings.bge_embedding_dir,
    )
    app.state.testing_corpora = {
        LOCAL_CORPUS_ID: _testing_corpus(app.state.testing_store),
    }
    yield


app = FastAPI(
    title="Personal Search + Grounded RAG",
    version="0.1.0",
    description="A local-first, inspectable retrieval and grounded-answering system.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.post("/api/testing/corpus", response_model=LocalCorpusResponse)
def create_testing_corpus(payload: LocalCorpusRequest) -> LocalCorpusResponse:
    """Create or extend the persistent local corpus with browser-selected files."""
    max_file_bytes = 20 * 1024 * 1024
    policy = ChunkingPolicy(target_size=1200, overlap=150, size_unit="characters")
    corpus_id = payload.corpus_id or LOCAL_CORPUS_ID
    corpus = app.state.testing_corpora.get(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=404,
            detail="This local corpus no longer exists. Start indexing again.",
        )

    if payload.source_paths is not None:
        selected_paths = set(payload.source_paths)
        for removed_path in corpus["store"].source_paths() - selected_paths:
            corpus["store"].delete_document(removed_path)
        corpus["files_indexed"] = corpus["store"].document_count
        corpus["retrievers"].clear()

    for selected_file in payload.files:
        corpus["files_received"] += 1
        try:
            if selected_file.relative_path in corpus["cancelled_files"]:
                raise IndexingCancelled(
                    f"Indexing cancelled for {selected_file.relative_path}"
                )
            content = base64.b64decode(selected_file.content_base64, validate=True)
            if len(content) > max_file_bytes:
                raise ValueError("file is larger than the 20 MB limit")
            document = normalize_local_file(
                selected_file.relative_path,
                selected_file.media_type,
                content,
            )
            result = corpus["store"].index_document(
                document,
                policy=policy,
                pipeline_version="local-dashboard-v1",
                embedding_provider=app.state.dense_embedding_provider,
                embedding_model_version=app.state.dense_embedding_provider.version,
                cancelled=lambda path=selected_file.relative_path: (
                    path in corpus["cancelled_files"]
                ),
            )
            if result.chunk_count == 0:
                raise ValueError("file produced no searchable chunks")
            corpus["files_indexed"] = corpus["store"].document_count
            corpus["retrievers"].clear()
        except IndexingCancelled:
            skipped = f"{selected_file.relative_path}: skipped by user"
            if skipped not in corpus["skipped_files"]:
                corpus["skipped_files"].append(skipped)
        except (ValueError, binascii.Error, IndexingError) as exc:
            corpus["skipped_files"].append(f"{selected_file.relative_path}: {exc}")

    return LocalCorpusResponse(
        corpus_id=corpus_id,
        files_received=corpus["files_received"],
        files_indexed=corpus["files_indexed"],
        chunks_indexed=len(corpus["store"].chunks()),
        skipped_files=corpus["skipped_files"],
    )


@app.post("/api/testing/corpus/{corpus_id}/skip", status_code=202)
def skip_testing_file(
    corpus_id: str,
    payload: LocalFileSkipRequest,
) -> dict[str, str]:
    """Cancel one in-progress file without blocking the next upload."""
    corpus = app.state.testing_corpora.get(corpus_id)
    if corpus is None:
        raise HTTPException(status_code=404, detail="This local corpus no longer exists.")
    corpus["cancelled_files"].add(payload.relative_path)
    return {"status": "skip_requested"}


@app.post("/api/testing/corpus/{corpus_id}/remove")
def remove_testing_file(
    corpus_id: str,
    payload: LocalFileSkipRequest,
) -> dict[str, str]:
    """Remove one source and all of its persisted chunks and dense embeddings."""
    corpus = app.state.testing_corpora.get(corpus_id)
    if corpus is None:
        raise HTTPException(status_code=404, detail="This local corpus no longer exists.")
    result = corpus["store"].delete_document(payload.relative_path)
    corpus["files_indexed"] = corpus["store"].document_count
    corpus["retrievers"].clear()
    return {"status": result.action}


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=50),
    stage: TestingStage | None = None,
    corpus_id: str | None = Query(default=None, min_length=1, max_length=64),
) -> SearchResponse:
    """Search with the configured stage or a stage chosen by the testing dashboard."""
    query_plan = process_query(q)
    if not query_plan.normalized_query:
        raise HTTPException(status_code=422, detail="Query must contain non-whitespace text")

    requested_stage = stage or app.state.retrieval_method
    chunks = app.state.chunks
    retrievers = app.state.retrievers
    corpus_revision: int | None = None
    dense_embeddings = None
    dense_embedding_provider = None
    if corpus_id is not None:
        corpus = app.state.testing_corpora.get(corpus_id)
        if corpus is None:
            raise HTTPException(
                status_code=404,
                detail="This local corpus no longer exists. Re-index the files.",
            )
        corpus_revision, chunks = corpus["store"].snapshot()
        retrievers = corpus["retrievers"]
        if requested_stage == "bge_dense":
            dense_embeddings = corpus["store"].vector_entries()
            dense_embedding_provider = app.state.dense_embedding_provider

    active_filters = query_plan.active_filters()
    filter_key = tuple(
        (item.field, item.operator, item.value, item.source) for item in active_filters
    )
    retriever_key = (requested_stage, filter_key, corpus_revision)
    retriever = retrievers.get(retriever_key)
    if retriever is None:
        retriever = build_retriever(
            chunks,
            requested_stage,
            filters=active_filters,
            dense_embeddings=dense_embeddings,
            dense_embedding_provider=dense_embedding_provider,
        )
        retrievers[retriever_key] = retriever

    try:
        rankings = {
            f"query_{index}": retriever.search(search_query, limit=limit)
            for index, search_query in enumerate(query_plan.search_queries)
        }
        candidates = (
            next(iter(rankings.values()), [])
            if len(rankings) <= 1
            else reciprocal_rank_fusion(rankings, limit=limit)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"The {requested_stage} test could not run: {exc}",
        ) from exc
    hits = [
        SearchHitResponse(
            chunk_id=item.chunk.id,
            document_id=item.chunk.document_id,
            source_path=item.chunk.metadata.source_path,
            text=item.chunk.text,
            score=item.score,
            rank=item.rank,
            page=item.chunk.metadata.page,
            section=item.chunk.metadata.section,
            matched_terms=item.details.get("matched_terms", []),
        )
        for item in candidates
    ]
    return SearchResponse(
        query=query_plan.raw_query,
        retrieval_method=requested_stage,
        hits=hits,
        query_variants=query_plan.search_queries,
        exact_terms=query_plan.exact_terms,
        metadata_filters=[
            {key: str(value) for key, value in item.model_dump().items()}
            for item in query_plan.filters
        ],
    )
