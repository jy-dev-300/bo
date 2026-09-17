import pytest
from chonkie import TokenChunker

from ingestion.chunking import ChunkingPolicy, chunk_document, exact_token_chunk_spans
from ingestion.models import NormalizedBlock, NormalizedDocument, SourceLocation


def make_document(*blocks: NormalizedBlock) -> NormalizedDocument:
    return NormalizedDocument(
        id="doc-test",
        source_path="notes/test.md",
        media_type="text/markdown",
        checksum="b" * 64,
        blocks=list(blocks),
    )


def test_empty_and_whitespace_only_documents_return_no_chunks() -> None:
    document = make_document(
        NormalizedBlock(
            kind="paragraph",
            text="   ",
            location=SourceLocation(source_path="notes/test.md"),
        )
    )
    assert chunk_document(document, ChunkingPolicy()) == []


def test_sentence_chunker_obeys_character_budget_and_overlap() -> None:
    text = "Alpha one. Beta two. Gamma three. Delta four."
    document = make_document(
        NormalizedBlock(
            kind="paragraph",
            text=text,
            location=SourceLocation(source_path="notes/test.md", char_start=10, char_end=55),
        )
    )
    chunks = chunk_document(
        document,
        ChunkingPolicy(target_size=30, overlap=12, size_unit="characters"),
    )

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert chunks[0].text.endswith("Beta two. ")
    assert chunks[1].text.startswith("Beta two. ")
    assert chunks[0].metadata.char_start == 10
    assert chunks[1].metadata.char_start == 21


def test_heading_stays_with_compatible_prose_and_source_locations_survive() -> None:
    document = make_document(
        NormalizedBlock(
            kind="heading",
            text="Retrieval",
            location=SourceLocation(
                source_path="notes/test.md", section="Retrieval", line_start=1, line_end=1
            ),
        ),
        NormalizedBlock(
            kind="paragraph",
            text="Exact identifiers matter.",
            location=SourceLocation(
                source_path="notes/test.md",
                section="Retrieval",
                line_start=3,
                line_end=3,
                char_start=11,
                char_end=36,
            ),
        ),
    )
    chunks = chunk_document(document, ChunkingPolicy(target_size=100, overlap=0))

    assert [chunk.text for chunk in chunks] == ["Retrieval\n\nExact identifiers matter."]
    assert chunks[0].metadata.section == "Retrieval"
    assert chunks[0].metadata.line_start == 1
    assert chunks[0].metadata.line_end == 3
    assert chunks[0].metadata.char_start is None
    assert len(chunks[0].metadata.attributes["source_locations"]) == 2


def test_new_heading_and_incompatible_page_start_new_groups() -> None:
    document = make_document(
        NormalizedBlock(
            kind="paragraph",
            text="Page one.",
            location=SourceLocation(source_path="notes/test.md", page=1),
        ),
        NormalizedBlock(
            kind="heading",
            text="Page Two",
            location=SourceLocation(source_path="notes/test.md", page=2, section="Two"),
        ),
        NormalizedBlock(
            kind="paragraph",
            text="Page two evidence.",
            location=SourceLocation(source_path="notes/test.md", page=2, section="Two"),
        ),
    )
    chunks = chunk_document(document, ChunkingPolicy(target_size=100, overlap=0))
    assert [chunk.metadata.page for chunk in chunks] == [1, 2]
    assert chunks[1].text.startswith("Page Two\n\n")


def test_token_mode_uses_declared_tokenizer_not_character_count() -> None:
    text = "one two three four five six seven eight"
    document = make_document(
        NormalizedBlock(
            kind="paragraph",
            text=text,
            location=SourceLocation(source_path="notes/test.md"),
        )
    )
    chunks = chunk_document(
        document,
        ChunkingPolicy(
            target_size=4,
            overlap=0,
            size_unit="tokens",
            tokenizer="word",
        ),
    )
    assert len(chunks) == 2
    assert all(len(chunk.text.split()) <= 4 for chunk in chunks)


def test_tokenizer_normalization_does_not_corrupt_source_offsets() -> None:
    text = "Deep learning systems train on large datasets."
    external = TokenChunker(tokenizer="word", chunk_size=4)(text)
    spans = exact_token_chunk_spans(text, external)
    assert [text[start:end] for _, start, end in spans] == [
        "Deep learning systems train",
        "on large datasets.",
    ]


def test_code_uses_syntax_path_and_preserves_line_ranges() -> None:
    code = "def one():\n    return 1\n\ndef two():\n    return 2\n"
    document = make_document(
        NormalizedBlock(
            kind="code",
            text=code,
            location=SourceLocation(
                source_path="notes/example.py",
                line_start=10,
                line_end=14,
                char_start=100,
                char_end=151,
            ),
            attributes={"language": "python"},
        )
    )
    chunks = chunk_document(document, ChunkingPolicy(target_size=30, overlap=0))
    assert chunks
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert all(chunk.metadata.attributes["content_kind"] == "code" for chunk in chunks)
    assert all(chunk.metadata.line_start is not None for chunk in chunks)


def test_policy_rejects_impossible_and_mismatched_budgets() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        ChunkingPolicy(target_size=10, overlap=10)
    with pytest.raises(ValueError, match="size_unit='tokens'"):
        ChunkingPolicy(strategy="semantic", size_unit="characters")
