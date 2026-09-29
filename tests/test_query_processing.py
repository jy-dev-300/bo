from datetime import date

from ingestion.models import Chunk, ChunkMetadata
from retrieval.query_processing import filter_chunks, process_query


def make_chunk(chunk_id: str, source_path: str, created_at: str) -> Chunk:
    """Create one filterable chunk for query-processing tests."""
    return Chunk(
        id=chunk_id,
        document_id=chunk_id.split(":")[0],
        ordinal=0,
        text=f"Text from {source_path}",
        metadata=ChunkMetadata(source_path=source_path, created_at=created_at),
    )


def test_query_plan_retains_raw_request_and_exact_evidence() -> None:
    """Verify that normalization and rewrites never replace exact user evidence."""
    raw = '  find  "ERR_CONNECTION_RESET" in fetch_withRetries.py  '

    plan = process_query(raw, supplied_variants=["connection retry failure"])

    assert plan.raw_query == raw
    assert plan.normalized_query == 'find "ERR_CONNECTION_RESET" in fetch_withRetries.py'
    assert plan.search_queries[0] == plan.normalized_query
    assert "ERR_CONNECTION_RESET" in plan.exact_terms
    assert "fetch_withRetries.py" in plan.exact_terms
    assert "ERR_CONNECTION_RESET" in plan.search_queries[-1]
    assert "fetch_withRetries.py" in plan.search_queries[-1]
    assert plan.transformations


def test_explicit_and_inferred_filters_remain_distinguishable() -> None:
    """Verify that user constraints are hard filters while date guesses stay visible and soft."""
    plan = process_query("find PDF files from last March", today=date(2026, 9, 22))

    assert any(
        item.field == "file_extension"
        and item.value == ".pdf"
        and item.source == "explicit"
        for item in plan.filters
    )
    assert any(item.field == "created_at" and item.source == "inferred" for item in plan.filters)
    assert all(item.source == "explicit" for item in plan.active_filters())


def test_shared_filters_select_the_same_corpus_for_every_retriever() -> None:
    """Verify that lexical and vector stages can begin with one identical filtered corpus."""
    chunks = [
        make_chunk("pdf:0", "records/taxes.pdf", "2026-03-02"),
        make_chunk("text:0", "records/taxes.txt", "2026-03-02"),
    ]
    plan = process_query("find PDF files")

    eligible = filter_chunks(chunks, plan.active_filters())

    assert [chunk.id for chunk in eligible] == ["pdf:0"]

