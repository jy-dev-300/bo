from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rag.interfaces import Reranker
from rag.models import SelectedEvidence
from retrieval.lexical import tokenize
from retrieval.models import RetrievalCandidate


class JevReranker:
    """TypeSafe Jev adapter implementing the project's reranker protocol.

    Each query-candidate pair receives an independent yes/no relevance judgment.
    Jev's returned Noul probability is used directly as the ranking score.
    """

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        client: Any | None = None,
        question: Any | None = None,
        max_workers: int = 12,
    ) -> None:
        """Configure the Jev model, client, relevance question, and concurrency limit."""
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._model_name = model_name
        self._client = client
        self._max_workers = max_workers
        self._question = question

    @property
    def details(self) -> dict[str, str]:
        """Describe the provider and model recorded beside each reranked candidate."""
        return {"provider": "typesafe", "model": self._model_name, "primitive": "noul"}

    def _get_client(self) -> Any:
        """Create the TypeSafe client only when Jev scoring is first requested."""
        if self._client is None:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError as error:
                raise RuntimeError(
                    "Jev reranking requires typesafe-sdk; install the project dependencies first"
                ) from error
            self._client = TypeSafeClient()
        return self._client

    def _get_question(self) -> Any:
        """Create and cache the yes-or-no relevance question sent to Jev."""
        if self._question is None:
            try:
                from typesafe_sdk import Noul, NoulCriteria
            except ImportError as error:
                raise RuntimeError(
                    "Jev reranking requires typesafe-sdk; install the project dependencies first"
                ) from error
            self._question = Noul(
                instructions=(
                    "Does the candidate passage contain information that directly helps answer "
                    "the search query?"
                ),
                criteria=NoulCriteria(
                    true=(
                        "The candidate directly answers the query or supplies evidence needed "
                        "to answer it, even when it uses different wording."
                    ),
                    false=(
                        "The candidate is unrelated, merely shares a broad topic, or does not "
                        "supply useful evidence for answering the query."
                    ),
                ),
            )
        return self._question

    def _score_candidate(self, query: str, candidate: RetrievalCandidate) -> float:
        """Ask Jev whether one candidate directly helps answer the query."""
        response = self._get_client().system_one(
            state={"query": query, "candidate_passage": candidate.chunk.text},
            questions={"is_relevant": self._get_question()},
            model=self._model_name,
        )
        return float(response.answers["is_relevant"].noul)

    def score(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[float]:
        """Score all candidates concurrently while preserving their input order."""
        if not candidates:
            return []
        self._get_client()
        self._get_question()
        workers = min(self._max_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(
                    lambda candidate: self._score_candidate(query, candidate),
                    candidates,
                )
            )


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    reranker: Reranker,
) -> list[RetrievalCandidate]:
    """Rescore candidates while preserving retrieval provenance."""
    if not candidates:
        return []

    scores = reranker.score(query, candidates)
    if len(scores) != len(candidates):
        raise ValueError("reranker must return exactly one score per candidate")

    reranker_details = getattr(reranker, "details", {})
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
                    "reranker": {
                        "adapter": type(reranker).__name__,
                        **reranker_details,
                    },
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
    """Measure lexical overlap between two texts using Jaccard similarity."""
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
    """Deduplicate candidates and fill a declared evidence budget."""
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
