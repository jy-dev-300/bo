from collections.abc import Mapping, Sequence

from retrieval.models import RetrievalCandidate


def fuse_candidates(
    lexical: Sequence[RetrievalCandidate],
    vector: Sequence[RetrievalCandidate],
    *,
    limit: int,
    rank_constant: int = 60,
    additional_rankings: Mapping[str, Sequence[RetrievalCandidate]] | None = None,
) -> list[RetrievalCandidate]:
    """Combine two ranked lists without assuming their raw scores are comparable.

    Student Implementation B. Consider rank fusion, duplicates, ties, missing
    candidates, provenance, and deterministic output. Do not sum raw scores.

    Josh:
    Reciprocal Rank Fusion (RRF):
        each doc gets a score:
        1 / (k + rank)
        say lex rank = 1, vec rank = 3 for doc A
        doc A score = 1 / (k + 1) + 1 / (k + 3), where k is smoothing constant

    """
    rankings: dict[str, Sequence[RetrievalCandidate]] = {
        "lexical": lexical,
        "vector": vector,
    }
    if additional_rankings:
        rankings.update(additional_rankings)
    return reciprocal_rank_fusion(
        rankings,
        limit=limit,
        rank_constant=rank_constant,
    )


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalCandidate]],
    *,
    limit: int,
    rank_constant: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[RetrievalCandidate]:
    """Fuse any number of result lists using stable, explainable RRF scores."""
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    if limit == 0:
        return []

    fused: dict[str, dict[str, object]] = {}
    for ranking_name, candidates in rankings.items():
        weight = 1.0 if weights is None else weights.get(ranking_name, 1.0)
        if weight < 0:
            raise ValueError("RRF weights cannot be negative")

        # A retriever should contribute at most once per chunk. If it emits a
        # duplicate, only its strongest (lowest-rank) occurrence counts.
        best_by_chunk: dict[str, RetrievalCandidate] = {}
        for candidate in candidates:
            current = best_by_chunk.get(candidate.chunk.id)
            if current is None or candidate.rank < current.rank:
                best_by_chunk[candidate.chunk.id] = candidate

        for candidate in best_by_chunk.values():
            contribution = weight / (rank_constant + candidate.rank)
            record = fused.setdefault(
                candidate.chunk.id,
                {
                    "chunk": candidate.chunk,
                    "score": 0.0,
                    "best_rank": candidate.rank,
                    "signals": {},
                },
            )
            record["score"] = float(record["score"]) + contribution
            record["best_rank"] = min(int(record["best_rank"]), candidate.rank)
            signals = record["signals"]
            assert isinstance(signals, dict)
            signals[ranking_name] = {
                "rank": candidate.rank,
                "raw_score": candidate.score,
                "rrf_contribution": contribution,
                "source": candidate.source,
            }

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["best_rank"]),
            item["chunk"].id,  # type: ignore[union-attr]
        ),
    )[:limit]

    return [
        RetrievalCandidate(
            chunk=item["chunk"],  # type: ignore[arg-type]
            score=float(item["score"]),
            rank=rank,
            source="hybrid",
            details={
                "method": "reciprocal_rank_fusion",
                "rank_constant": rank_constant,
                "signals": item["signals"],
            },
        )
        for rank, item in enumerate(ordered, start=1)
    ]
