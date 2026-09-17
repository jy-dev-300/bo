from typing import Protocol, Sequence

from rag.models import SelectedEvidence
from retrieval.models import RetrievalCandidate


class Reranker(Protocol):
    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[float]: ...


class Generator(Protocol):
    def generate(self, query: str, evidence: Sequence[SelectedEvidence]) -> str: ...

