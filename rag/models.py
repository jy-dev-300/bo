from pydantic import BaseModel, Field

from retrieval.models import RetrievalCandidate


class SelectedEvidence(BaseModel):
    candidate: RetrievalCandidate
    context_order: int = Field(ge=1)
    budget_cost: int = Field(ge=0)


class Citation(BaseModel):
    claim: str
    chunk_id: str
    source_path: str
    page: int | None = None
    section: str | None = None


class GroundedAnswer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False

