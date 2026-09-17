import math
import re
from collections import Counter
from collections.abc import Sequence

from ingestion.models import Chunk
from retrieval.models import RetrievalCandidate

TOKEN_PATTERN = re.compile(r"[\w.-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """A deliberately visible baseline tokenizer; production analyzers may replace it."""
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._tokens = [tokenize(chunk.text) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._tokens else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalCandidate]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        scored: list[tuple[float, Chunk, list[str]]] = []
        corpus_size = len(self._chunks)

        for chunk, tokens, frequencies in zip(
            self._chunks, self._tokens, self._term_frequencies, strict=True
        ):
            score = 0.0
            matched_terms: list[str] = []
            length_ratio = len(tokens) / self._average_length if self._average_length else 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if frequency == 0:
                    continue
                matched_terms.append(term)
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                score += inverse_document_frequency * (frequency * (self._k1 + 1)) / denominator

            if score > 0:
                scored.append((score, chunk, matched_terms))

        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            RetrievalCandidate(
                chunk=chunk,
                score=score,
                rank=rank,
                source="lexical",
                details={"matched_terms": matched_terms},
            )
            for rank, (score, chunk, matched_terms) in enumerate(scored[:limit], start=1)
        ]

