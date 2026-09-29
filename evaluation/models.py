from pydantic import BaseModel, Field

from rag.models import GroundedAnswer


class RetrievalGoldCase(BaseModel):
    """One search test containing a question and the chunks expected to answer it."""

    id: str
    query: str
    relevant_chunk_ids: set[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class RetrievalCaseResult(BaseModel):
    """The search results and calculated scores for one test question."""

    query_id: str
    retrieved_chunk_ids: list[str]
    recall_at_k: float
    reciprocal_rank: float


class RAGGoldCase(BaseModel):
    """One answer test containing the question, required evidence, and required facts."""

    id: str
    query: str
    relevant_chunk_ids: set[str]
    required_facts: list[str]


class RAGCaseResult(BaseModel):
    """Separate checks showing where one generated answer succeeded or failed."""

    query_id: str
    retrieval_found_evidence: bool
    answer_used_evidence: bool | None
    answer_supported: bool | None
    citations_correct: bool | None
    notes: list[str] = Field(default_factory=list)
    answer: GroundedAnswer
