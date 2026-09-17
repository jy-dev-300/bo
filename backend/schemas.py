from pydantic import BaseModel, Field


class SearchHitResponse(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    text: str
    score: float
    rank: int
    page: int | None = None
    section: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    retrieval_method: str
    hits: list[SearchHitResponse]

