from collections.abc import Sequence

from evaluation.models import RAGCaseResult, RAGGoldCase
from rag.citations import has_lexical_support
from rag.models import GroundedAnswer, SelectedEvidence


def contains_fact(text: str, fact: str) -> bool:
    """Check whether a required fact appears in text, ignoring case and extra spaces."""
    normalized_text = " ".join(text.casefold().split())
    normalized_fact = " ".join(fact.casefold().split())
    return bool(normalized_fact) and normalized_fact in normalized_text


def evaluate_rag_case(
    gold: RAGGoldCase,
    answer: GroundedAnswer,
    *,
    retrieved_chunk_ids: Sequence[str],
    selected_evidence: Sequence[SelectedEvidence],
) -> RAGCaseResult:
    """Check search, evidence use, factual support, and citations separately.

    The checks are deterministic. Exact required facts must appear in both the answer
    and selected evidence. Claim support uses the citation module's lexical check.
    """
    # Example output:
    # Test question: "What startup ideas have I written down?"
    # query_id: "summarize-startup-ideas"
    # retrieval_found_evidence: True
    # answer_used_evidence: True
    # answer_supported: True
    # citations_correct: False
    # notes: ["The answer used the startup notes but cited an unrelated section."]
    notes: list[str] = []
    retrieved_ids = set(retrieved_chunk_ids)
    retrieval_found_evidence = gold.relevant_chunk_ids <= retrieved_ids
    if not retrieval_found_evidence:
        missing = sorted(gold.relevant_chunk_ids - retrieved_ids)
        notes.append(f"Search missed required chunks: {', '.join(missing)}")

    if answer.abstained:
        notes.append("BO abstained, so evidence use, support, and citations were not scored.")
        return RAGCaseResult(
            query_id=gold.id,
            retrieval_found_evidence=retrieval_found_evidence,
            answer_used_evidence=None,
            answer_supported=None,
            citations_correct=None,
            notes=notes,
            answer=answer,
        )

    evidence_by_id = {
        item.candidate.chunk.id: item.candidate.chunk for item in selected_evidence
    }
    selected_texts = [chunk.text for chunk in evidence_by_id.values()]
    facts_traced_to_evidence = bool(gold.required_facts) and all(
        contains_fact(answer.text, fact)
        and any(contains_fact(evidence_text, fact) for evidence_text in selected_texts)
        for fact in gold.required_facts
    )
    if not facts_traced_to_evidence:
        notes.append("Not every required fact appears in both the answer and selected evidence.")

    citations_correct = bool(answer.citations)
    answer_supported = bool(answer.citations)
    for citation in answer.citations:
        chunk = evidence_by_id.get(citation.chunk_id)
        if chunk is None:
            citations_correct = False
            answer_supported = False
            notes.append(f"Citation uses an unselected chunk: {citation.chunk_id}")
            continue

        provenance_matches = (
            citation.source_path == chunk.metadata.source_path
            and citation.page == chunk.metadata.page
            and citation.section == chunk.metadata.section
        )
        if not provenance_matches:
            citations_correct = False
            notes.append(f"Citation location does not match chunk metadata: {citation.chunk_id}")
        if not has_lexical_support(citation.claim, chunk.text):
            answer_supported = False
            notes.append(f"Selected chunk does not support the cited claim: {citation.chunk_id}")

    if not answer.citations:
        notes.append("The answer contains no citations.")

    return RAGCaseResult(
        query_id=gold.id,
        retrieval_found_evidence=retrieval_found_evidence,
        answer_used_evidence=facts_traced_to_evidence,
        answer_supported=answer_supported,
        citations_correct=citations_correct,
        notes=notes,
        answer=answer,
    )
