from collections.abc import Sequence
from typing import Protocol

from rag.models import SelectedEvidence
from retrieval.models import RetrievalCandidate


class Reranker(Protocol):
    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[float]:
        """Assign one relevance score to each candidate for the supplied query."""
        ...


class Generator(Protocol):
    def generate(self, query: str, evidence: Sequence[SelectedEvidence]) -> str:
        """Generate an answer to the query using only the selected evidence."""
        ...
