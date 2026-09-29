from pathlib import Path

import pytest

from ingestion.models import Chunk, NormalizedBlock, NormalizedDocument, SourceLocation
from retrieval.corpus import load_chunks_jsonl


class FakeLocalBGEProvider:
    """Keep API tests offline while exercising persistent dense-vector wiring."""

    dimension = 2
    model_name = "fake-local-bge"
    version = "fake-local-bge:dense:v1"
    document_calls = 0
    query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        """Create one deterministic test vector from text length and character values."""
        return [float(len(text)), float(sum(ord(character) for character in text) % 997)]

    def embed_documents(self, texts):
        """Return one deterministic dense vector per test chunk."""
        type(self).document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        """Return a query vector with the same test dimension."""
        type(self).query_calls += 1
        return self._vector(text)


@pytest.fixture(autouse=True)
def isolate_local_bge_index(tmp_path, monkeypatch):
    """Prevent API tests from loading BGE or sharing persisted files across cases."""
    monkeypatch.setenv("BGE_FILE_INDEX_DIR", str(tmp_path / "file-index"))
    monkeypatch.setenv("BGE_EMBEDDING_DIR", str(tmp_path / "embeddings"))

    from backend import config, main

    config.get_settings.cache_clear()
    FakeLocalBGEProvider.document_calls = 0
    FakeLocalBGEProvider.query_calls = 0
    monkeypatch.setattr(main, "BGEDenseEmbeddingProvider", FakeLocalBGEProvider)
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide the sample JSONL corpus as validated chunks for tests."""
    return load_chunks_jsonl(Path("sample_corpus/chunks.jsonl"))


@pytest.fixture
def prose_document() -> NormalizedDocument:
    """Provide a small prose document with stable source locations for chunking tests."""
    return NormalizedDocument(
        id="doc-fixture",
        source_path="notes/architecture.txt",
        media_type="text/plain",
        checksum="a" * 64,
        blocks=[
            NormalizedBlock(
                kind="heading",
                text="Retrieval",
                location=SourceLocation(source_path="notes/architecture.txt", section="Retrieval"),
            ),
            NormalizedBlock(
                kind="paragraph",
                text="Lexical retrieval preserves exact identifiers and rare terms.",
                location=SourceLocation(
                    source_path="notes/architecture.txt",
                    section="Retrieval",
                    char_start=0,
                    char_end=61,
                ),
            ),
        ],
    )
