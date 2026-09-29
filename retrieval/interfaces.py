from collections.abc import Sequence
from typing import Protocol

from ingestion.models import Chunk
from retrieval.models import RetrievalCandidate


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        """Return the fixed number of values in every produced embedding."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert document texts into vectors suitable for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Convert one search query into a vector compatible with document vectors."""
        ...


class VectorRetriever(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalCandidate]:
        """Return the nearest indexed chunks for a query vector and optional filters."""
        ...


class ChunkRepository(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Insert new chunks or replace stored chunks that share their IDs."""
        ...

    def get(self, chunk_id: str) -> Chunk | None:
        """Return a stored chunk by ID, or none when it does not exist."""
        ...
