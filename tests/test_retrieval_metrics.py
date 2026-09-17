import pytest

from evaluation.models import RetrievalGoldCase
from evaluation.retrieval_metrics import evaluate_retrieval, recall_at_k, reciprocal_rank
from ingestion.models import Chunk, ChunkMetadata
from retrieval.models import RetrievalCandidate


def candidate(chunk_id: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc",
            ordinal=rank - 1,
            text=chunk_id,
            metadata=ChunkMetadata(source_path="doc.txt"),
        ),
        score=float(10 - rank),
        rank=rank,
        source="lexical",
    )


def test_recall_counts_distinct_relevant_chunks_within_k() -> None:
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, k=2) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=0)


def test_reciprocal_rank_uses_first_relevant_hit() -> None:
    assert reciprocal_rank(["wrong", "right", "right"], {"right"}) == 0.5
    assert reciprocal_rank(["wrong"], {"right"}) == 0.0


def test_evaluation_keeps_per_query_hits_and_scores() -> None:
    cases = [
        RetrievalGoldCase(id="q1", query="found", relevant_chunk_ids={"right"}),
        RetrievalGoldCase(id="q2", query="missing", relevant_chunk_ids={"other"}),
    ]

    def search(query: str, limit: int) -> list[RetrievalCandidate]:
        assert limit == 2
        return [candidate("wrong", 1), candidate("right", 2)]

    results = evaluate_retrieval(cases, search, k=2)

    assert [item.recall_at_k for item in results] == [1.0, 0.0]
    assert [item.reciprocal_rank for item in results] == [0.5, 0.0]
    assert results[0].retrieved_chunk_ids == ["wrong", "right"]
