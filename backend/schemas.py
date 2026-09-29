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
    query_variants: list[str] = Field(default_factory=list)
    exact_terms: list[str] = Field(default_factory=list)
    metadata_filters: list[dict[str, str]] = Field(default_factory=list)


class LocalFileRequest(BaseModel):
    """Carry one browser-selected file to the local indexing endpoint."""

    relative_path: str = Field(min_length=1, max_length=1000)
    media_type: str | None = Field(default=None, max_length=200)
    content_base64: str


class LocalCorpusRequest(BaseModel):
    """Carry files and an optional complete path snapshot for the local corpus."""

    corpus_id: str | None = Field(default=None, min_length=1, max_length=64)
    files: list[LocalFileRequest] = Field(max_length=100)
    source_paths: list[str] | None = Field(default=None, max_length=100_000)


class LocalFileSkipRequest(BaseModel):
    """Identify the in-progress file that the browser asked to skip."""

    relative_path: str = Field(min_length=1, max_length=1000)


class LocalCorpusResponse(BaseModel):
    """Summarize which selected files were indexed and which were skipped."""

    corpus_id: str
    files_received: int
    files_indexed: int
    chunks_indexed: int
    skipped_files: list[str] = Field(default_factory=list)
