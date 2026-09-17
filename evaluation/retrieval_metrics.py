from collections.abc import Callable, Sequence

from evaluation.models import RetrievalCaseResult, RetrievalGoldCase
from retrieval.models import RetrievalCandidate


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
    *,
    k: int,
) -> float:
    """Fraction of distinct known-relevant chunks present in the first k hits."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids cannot be empty")
    return len(set(retrieved_chunk_ids[:k]) & relevant_chunk_ids) / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
) -> float:
    """Return 1/rank of the first relevant result, or zero when absent."""
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    cases: Sequence[RetrievalGoldCase],
    search: Callable[[str, int], Sequence[RetrievalCandidate]],
    *,
    k: int,
) -> list[RetrievalCaseResult]:
    """Evaluate every gold query and keep its ranked IDs for failure analysis."""
    if k <= 0:
        raise ValueError("k must be positive")
    results: list[RetrievalCaseResult] = []
    for case in cases:
        retrieved_ids = [candidate.chunk.id for candidate in search(case.query, k)][:k]
        results.append(
            RetrievalCaseResult(
                query_id=case.id,
                retrieved_chunk_ids=retrieved_ids,
                recall_at_k=recall_at_k(retrieved_ids, case.relevant_chunk_ids, k=k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, case.relevant_chunk_ids),
            )
        )
    return results
