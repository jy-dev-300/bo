from collections.abc import Sequence

from rag.interfaces import Reranker
from rag.models import SelectedEvidence
from retrieval.lexical import tokenize
from retrieval.models import RetrievalCandidate


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    reranker: Reranker,
) -> list[RetrievalCandidate]:
    """Student Implementation C: rescore candidates and preserve provenance."""
    if not candidates:
        return []

    scores = reranker.score(query, candidates)
    if len(scores) != len(candidates):
        raise ValueError("reranker must return exactly one score per candidate")

    rescored = [
        candidate.model_copy(
            update={
                "score": float(score),
                "source": "reranker",
                "details": {
                    **candidate.details,
                    "pre_rerank": {
                        "source": candidate.source,
                        "rank": candidate.rank,
                        "score": candidate.score,
                    },
                    "reranker_score": float(score),
                },
            }
        )
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda candidate: (-candidate.score, candidate.chunk.id))
    return [
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(rescored, start=1)
    ]


def text_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def select_context(
    candidates: Sequence[RetrievalCandidate],
    *,
    budget: int,
) -> list[SelectedEvidence]:
    """Student Implementation C: deduplicate and fill a declared evidence budget."""
    if budget < 0:
        raise ValueError("budget cannot be negative")

    unique: list[RetrievalCandidate] = []
    seen_chunk_ids: set[str] = set()
    for candidate in candidates:
        if candidate.chunk.id in seen_chunk_ids:
            continue
        if any(text_similarity(candidate.chunk.text, item.chunk.text) >= 0.85 for item in unique):
            continue
        seen_chunk_ids.add(candidate.chunk.id)
        unique.append(candidate)

    # Give each document one opportunity before filling remaining space by rank.
    first_from_document: list[RetrievalCandidate] = []
    remaining: list[RetrievalCandidate] = []
    seen_documents: set[str] = set()
    for candidate in unique:
        if candidate.chunk.document_id in seen_documents:
            remaining.append(candidate)
        else:
            seen_documents.add(candidate.chunk.document_id)
            first_from_document.append(candidate)

    selected: list[SelectedEvidence] = []
    used = 0
    for candidate in [*first_from_document, *remaining]:
        cost = len(candidate.chunk.text)
        if used + cost > budget:
            continue
        selected.append(
            SelectedEvidence(
                candidate=candidate,
                context_order=len(selected) + 1,
                budget_cost=cost,
            )
        )
        used += cost

    return selected
