"""Small, in-memory BM25 search over text chunks.

An "index" here prepares word counts once, then uses them for each question.
This version still checks every chunk during search and saves nothing to disk.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

from ingestion.models import Chunk
from retrieval.models import RetrievalCandidate

# A token is a searchable piece of text. This pattern keeps letters, numbers,
# underscores, periods, and hyphens together for names and file paths.
TOKEN_PATTERN = re.compile(r"[\w.-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Turn text into search terms; casefold makes case differences harmless."""
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    """Rank chunks by matching words, giving rare words more weight.

    k1 controls how much repeated words help. b controls how much we adjust
    for chunk length, so long chunks do not win just by having more words.
    """

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        """Precompute token statistics needed to score this in-memory chunk corpus."""
        # Prepare these counts once so later searches can reuse them.
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        # Term frequency means how often a word appears in one chunk.
        self._tokens = [tokenize(chunk.text) for chunk in self._chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        # BM25 compares each chunk's length with this average.
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._tokens else 0.0
        )
        # Document frequency means how many different chunks contain a word.
        # set(tokens) makes repeated words count only once per chunk here.
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalCandidate]:
        """Score matching chunks and return at most `limit` results."""
        # A repeated query word should not be scored twice.
        query_terms = list(dict.fromkeys(tokenize(query)))
        scored: list[tuple[float, Chunk, list[str]]] = []
        corpus_size = len(self._chunks)

        # This small reference index checks every chunk for each search.
        for chunk, tokens, frequencies in zip(
            self._chunks, self._tokens, self._term_frequencies, strict=True
        ):
            score = 0.0
            matched_terms: list[str] = []
            # Above 1 means this chunk is longer than average.
            length_ratio = len(tokens) / self._average_length if self._average_length else 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if frequency == 0:
                    continue
                matched_terms.append(term)
                document_frequency = self._document_frequency[term]
                # Inverse document frequency (IDF): a word found in few chunks
                # tells us more than a word found almost everywhere.
                inverse_document_frequency = math.log(
                    1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                # Repeated matches help, but with diminishing returns. The
                # length adjustment stops long chunks getting a free boost.
                denominator = frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                score += inverse_document_frequency * (frequency * (self._k1 + 1)) / denominator

            # Zero means none of the query words appeared in this chunk.
            if score > 0:
                scored.append((score, chunk, matched_terms))

        # Highest score first; chunk ID gives a stable order when scores tie.
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            RetrievalCandidate(
                chunk=chunk,
                score=score,
                rank=rank,
                source="lexical",
                details={"matched_terms": matched_terms},  # Words explaining the match.
            )
            for rank, (score, chunk, matched_terms) in enumerate(scored[:limit], start=1)
        ]
