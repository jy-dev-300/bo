from pathlib import Path

import pytest

from ingestion.models import Chunk, NormalizedBlock, NormalizedDocument, SourceLocation
from retrieval.corpus import load_chunks_jsonl


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return load_chunks_jsonl(Path("sample_corpus/chunks.jsonl"))


@pytest.fixture
def prose_document() -> NormalizedDocument:
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

