import re
from collections.abc import Sequence

from rag.models import Citation, SelectedEvidence
from retrieval.lexical import tokenize

CITATION_PATTERN = re.compile(r"\[([^\[\]\s]+)\]")
CLAIM_BOUNDARIES = ".!?\n"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def claim_for_citation(answer_text: str, marker_start: int) -> str:
    """Return the sentence fragment immediately preceding a citation marker."""
    claim_start = max(answer_text.rfind(boundary, 0, marker_start) for boundary in CLAIM_BOUNDARIES)
    fragment = answer_text[claim_start + 1 : marker_start]
    return CITATION_PATTERN.sub("", fragment).strip(" ,;:-")


def has_lexical_support(claim: str, chunk_text: str) -> bool:
    """Conservatively reject claims with too little textual support in a chunk."""
    claim_terms = {term for term in tokenize(claim) if term not in STOPWORDS}
    if not claim_terms:
        return False
    chunk_terms = set(tokenize(chunk_text))
    overlap = claim_terms & chunk_terms
    required = 1 if len(claim_terms) <= 2 else (len(claim_terms) + 1) // 2
    return len(overlap) >= required


def assemble_citations(
    answer_text: str,
    evidence: Sequence[SelectedEvidence],
) -> list[Citation]:
    """Validate ``[chunk-id]`` markers and assemble citations from selected evidence."""
    evidence_by_chunk_id = {
        selected.candidate.chunk.id: selected.candidate.chunk for selected in evidence
    }
    citations: list[Citation] = []

    for marker in CITATION_PATTERN.finditer(answer_text):
        chunk_id = marker.group(1)
        chunk = evidence_by_chunk_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"cited chunk ID is not present in selected evidence: {chunk_id}")

        claim = claim_for_citation(answer_text, marker.start())
        if not claim:
            raise ValueError(f"cited chunk ID has no associated claim: {chunk_id}")
        if not has_lexical_support(claim, chunk.text):
            raise ValueError(f"cited chunk does not support its associated claim: {chunk_id}")

        citations.append(
            Citation(
                claim=claim,
                chunk_id=chunk.id,
                source_path=chunk.metadata.source_path,
                page=chunk.metadata.page,
                section=chunk.metadata.section,
            )
        )

    return citations
