from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/personal_search"
    corpus_path: Path = Path("sample_corpus/chunks.jsonl")
    bge_file_index_dir: Path = Path("bo-bge-file-index")
    bge_embedding_dir: Path = Path("bo-bge-embeds")
    cors_origins: str = "http://localhost:5173"
    retrieval_mode: Literal[
        "bm25",
        "bge_dense",
        "hybrid",
        "rerank",
        "rerank_bge",
        "rerank_jev",
        "specialists",
        "advanced",
    ] = "bm25"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        """Split the configured comma-separated CORS origins into clean URL strings."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Create the application settings once and reuse it for later requests."""
    return Settings()
