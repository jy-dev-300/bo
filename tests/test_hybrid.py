from ingestion.models import Chunk, ChunkMetadata
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.models import RetrievalCandidate
from retrieval.query_processing import expand_query


def make_candidate(chunk_id: str, rank: int, source: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(
            id=chunk_id,
            document_id=chunk_id.split(":")[0],
            ordinal=0,
            text=f"Text for {chunk_id}",
            metadata=ChunkMetadata(source_path=f"{chunk_id}.txt"),
        ),
        score=float(10 - rank),
        rank=rank,
        source=source,
    )


def test_rrf_deduplicates_and_is_deterministic() -> None:
    lexical = [
        make_candidate("a:0", 1, "lexical"),
        make_candidate("a:0", 3, "lexical"),
        make_candidate("b:0", 2, "lexical"),
    ]
    vector = [
        make_candidate("b:0", 1, "vector"),
        make_candidate("a:0", 2, "vector"),
    ]

    first = reciprocal_rank_fusion(
        {"lexical": lexical, "vector": vector},
        limit=10,
    )
    second = reciprocal_rank_fusion(
        {"lexical": lexical, "vector": vector},
        limit=10,
    )

    assert [candidate.chunk.id for candidate in first] == ["a:0", "b:0"]
    assert first == second
    assert first[0].details["signals"]["lexical"]["rank"] == 1


def test_query_expansion_preserves_original_and_splits_identifiers() -> None:
    assert expand_query("find fetch_withRetries.py") == [
        "find fetch_withRetries.py",
        "find fetch with Retries py",
    ]
