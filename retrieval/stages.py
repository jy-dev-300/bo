from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from ingestion.models import Chunk
from rag.selection import JevReranker
from retrieval.advanced import BGEDenseEmbeddingProvider, BGEDenseRetriever, PowerfulRetriever
from retrieval.lexical import BM25Index
from retrieval.models import RetrievalCandidate
from retrieval.query_processing import MetadataFilter, filter_chunks

RetrievalStage = Literal[
    "bm25",
    "bge_dense",
    "hybrid",
    "rerank",
    "rerank_bge",
    "rerank_jev",
    "specialists",
]
STAGES: tuple[RetrievalStage, ...] = (
    "bm25",
    "bge_dense",
    "hybrid",
    "rerank",
    "rerank_bge",
    "rerank_jev",
    "specialists",
)


class Searcher(Protocol):
    def search(self, query: str, *, limit: int) -> list[RetrievalCandidate]:
        """Return up to the requested number of ranked candidates for a query."""
        ...


def build_retriever(
    chunks: Sequence[Chunk],
    stage: RetrievalStage,
    *,
    filters: Sequence[MetadataFilter] | Mapping[str, object] | None = None,
    dense_embeddings: Mapping[str, Sequence[float]] | None = None,
    dense_embedding_provider: BGEDenseEmbeddingProvider | None = None,
) -> Searcher:
    """Change one retrieval component while applying the same metadata filters."""
    eligible_chunks = filter_chunks(chunks, filters)
    if stage == "bm25":
        return BM25Index(eligible_chunks)
    if stage == "bge_dense":
        return BGEDenseRetriever(
            eligible_chunks,
            embeddings=dense_embeddings,
            embedding_provider=dense_embedding_provider,
        )
    if stage not in STAGES:
        raise ValueError(f"Unknown retrieval stage: {stage}")
    return PowerfulRetriever(
        eligible_chunks,
        reranker=JevReranker() if stage == "rerank_jev" else None,
        use_reranker=stage in {"rerank", "rerank_bge", "rerank_jev", "specialists"},
        use_splade=stage == "specialists",
    )
