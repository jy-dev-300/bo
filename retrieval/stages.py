from collections.abc import Sequence
from typing import Literal, Protocol

from ingestion.models import Chunk
from retrieval.advanced import PowerfulRetriever
from retrieval.lexical import BM25Index
from retrieval.models import RetrievalCandidate

RetrievalStage = Literal["bm25", "hybrid", "rerank", "specialists"]
STAGES: tuple[RetrievalStage, ...] = ("bm25", "hybrid", "rerank", "specialists")


class Searcher(Protocol):
    def search(self, query: str, *, limit: int) -> list[RetrievalCandidate]: ...


def build_retriever(chunks: Sequence[Chunk], stage: RetrievalStage) -> Searcher:
    """Change one retrieval component at each successive stage."""
    if stage == "bm25":
        return BM25Index(chunks)
    if stage not in STAGES:
        raise ValueError(f"Unknown retrieval stage: {stage}")
    return PowerfulRetriever(
        chunks,
        use_reranker=stage in {"rerank", "specialists"},
        use_splade=stage == "specialists",
    )
