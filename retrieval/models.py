from typing import Any, Literal

from pydantic import BaseModel, Field

from ingestion.models import Chunk


class RetrievalCandidate(BaseModel):
    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    source: Literal[
        "lexical",
        "vector",
        "sparse",
        "splade",
        "late_interaction",
        "metadata",
        "hybrid",
        "reranker",
    ]
    details: dict[str, Any] = Field(default_factory=dict)
