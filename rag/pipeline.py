from dataclasses import dataclass

from rag.interfaces import Generator, Reranker
from rag.models import GroundedAnswer
from retrieval.interfaces import EmbeddingProvider, VectorRetriever
from retrieval.lexical import BM25Index


@dataclass(frozen=True)
class RAGDependencies:
    lexical: BM25Index
    embeddings: EmbeddingProvider
    vector: VectorRetriever
    reranker: Reranker
    generator: Generator


def answer_question(query: str, deps: RAGDependencies) -> GroundedAnswer:
    """Future orchestration point; stages stay named and inspectable.

    This remains unavailable until Student Implementations B-D are complete, so
    the API cannot present ungrounded generation as a working RAG answer.
    """
    raise NotImplementedError("Complete hybrid retrieval, selection, and citations first")

