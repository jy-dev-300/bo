from pydantic import BaseModel, Field

from rag.models import GroundedAnswer


class RetrievalGoldCase(BaseModel):
    id: str
    query: str
    relevant_chunk_ids: set[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class RetrievalCaseResult(BaseModel):
    query_id: str
    retrieved_chunk_ids: list[str]
    recall_at_k: float
    reciprocal_rank: float


class RAGGoldCase(BaseModel):
    id: str
    query: str
    relevant_chunk_ids: set[str]
    required_facts: list[str]


class RAGCaseResult(BaseModel):
    query_id: str
    retrieval_found_evidence: bool
    answer_used_evidence: bool | None
    answer_supported: bool | None
    citations_correct: bool | None
    notes: list[str] = Field(default_factory=list)
    answer: GroundedAnswer

