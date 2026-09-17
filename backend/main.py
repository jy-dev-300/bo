from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.schemas import SearchHitResponse, SearchResponse
from retrieval.corpus import load_chunks_jsonl
from retrieval.stages import build_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    chunks = load_chunks_jsonl(settings.corpus_path)
    stage = "specialists" if settings.retrieval_mode == "advanced" else settings.retrieval_mode
    app.state.retrieval_method = stage
    app.state.retriever = build_retriever(chunks, stage)
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
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=10, ge=1, le=50),
) -> SearchResponse:
    normalized_query = q.strip()
    if not normalized_query:
        raise HTTPException(status_code=422, detail="Query must contain non-whitespace text")

    candidates = app.state.retriever.search(normalized_query, limit=limit)
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
        query=normalized_query,
        retrieval_method=app.state.retrieval_method,
        hits=hits,
    )
