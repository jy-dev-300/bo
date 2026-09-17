from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/personal_search"
    corpus_path: Path = Path("sample_corpus/chunks.jsonl")
    cors_origins: str = "http://localhost:5173"
    retrieval_mode: Literal["bm25", "hybrid", "rerank", "specialists", "advanced"] = "bm25"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
