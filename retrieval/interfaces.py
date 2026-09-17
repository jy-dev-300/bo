from typing import Protocol, Sequence

from ingestion.models import Chunk
from retrieval.models import RetrievalCandidate


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorRetriever(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalCandidate]: ...


class ChunkRepository(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

