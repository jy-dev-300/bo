from collections.abc import Callable, Sequence

from evaluation.models import RetrievalCaseResult, RetrievalGoldCase
from retrieval.models import RetrievalCandidate


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
    *,
    k: int,
) -> float:
    """Calculate what portion of the expected chunks appear in the first k results."""
    # Formula: Recall@k = number of expected chunks found in the first k results
    #                     / total number of expected chunks
    # Set notation: Recall@k = |top_k_results intersect expected_chunks| / |expected_chunks|
    # Example output: 0.5
    # Meaning: the search found one of the two expected chunks in its first k results.
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids cannot be empty")
    return len(set(retrieved_chunk_ids[:k]) & relevant_chunk_ids) / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
) -> float:
    """Score how early the first expected chunk appears.

    First place scores 1.0, second place scores 0.5, third place scores about 0.33,
    and no expected chunk scores 0.0.
    """
    # Formula: Reciprocal Rank = 1 / position of the first expected chunk
    # If no expected chunk appears, Reciprocal Rank = 0.
    # Example output: 0.5
    # Meaning: the first expected chunk appeared as the second search result.
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
    """Run every test question and record the returned chunks and both search scores."""
    # Example output for one test question:
    # Test question: "Find files that talk about startup ideas."
    # query_id: "find-startup-ideas"
    # retrieved_chunk_ids: ["startup-notes:003", "product-ideas:001", "meeting-notes:008"]
    # recall_at_k: 1.0
    # reciprocal_rank: 1.0
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
