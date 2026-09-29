from evaluation.models import RAGGoldCase
from evaluation.rag_metrics import evaluate_rag_case
from ingestion.models import Chunk, ChunkMetadata
from rag.models import Citation, GroundedAnswer, SelectedEvidence
from retrieval.models import RetrievalCandidate


def evidence(chunk_id: str, text: str) -> SelectedEvidence:
    """Create selected evidence with stable provenance for RAG metric tests."""
    chunk = Chunk(
        id=chunk_id,
        document_id="startup-notes",
        ordinal=0,
        text=text,
        metadata=ChunkMetadata(
            source_path="notes/startup-ideas.pdf",
            page=2,
            section="Ideas",
        ),
    )
    return SelectedEvidence(
        candidate=RetrievalCandidate(
            chunk=chunk,
            score=1.0,
            rank=1,
            source="reranker",
        ),
        context_order=1,
        budget_cost=len(text),
    )


def gold_case() -> RAGGoldCase:
    """Create one BO-style question with required evidence and a required fact."""
    return RAGGoldCase(
        id="summarize-startup-ideas",
        query="What startup ideas have I written down?",
        relevant_chunk_ids={"startup-notes:003"},
        required_facts=["a shared neighborhood tool library"],
    )


def grounded_answer() -> GroundedAnswer:
    """Create a fully supported answer with matching citation provenance."""
    return GroundedAnswer(
        text="One idea is a shared neighborhood tool library.",
        citations=[
            Citation(
                claim="A shared neighborhood tool library is one startup idea",
                chunk_id="startup-notes:003",
                source_path="notes/startup-ideas.pdf",
                page=2,
                section="Ideas",
            )
        ],
    )


def test_rag_metrics_keep_all_four_checks_separate() -> None:
    """Verify that a fully grounded answer passes every independent check."""
    selected = evidence(
        "startup-notes:003",
        "Startup idea: a shared neighborhood tool library for apartment residents.",
    )

    result = evaluate_rag_case(
        gold_case(),
        grounded_answer(),
        retrieved_chunk_ids=["startup-notes:003"],
        selected_evidence=[selected],
    )

    assert result.retrieval_found_evidence is True
    assert result.answer_used_evidence is True
    assert result.answer_supported is True
    assert result.citations_correct is True
    assert result.notes == []


def test_rag_metrics_report_retrieval_and_citation_failures() -> None:
    """Verify that missing retrieval and incorrect provenance fail separately."""
    selected = evidence(
        "startup-notes:003",
        "Startup idea: a shared neighborhood tool library for apartment residents.",
    )
    answer = grounded_answer().model_copy(
        update={
            "citations": [
                grounded_answer().citations[0].model_copy(update={"page": 9})
            ]
        }
    )

    result = evaluate_rag_case(
        gold_case(),
        answer,
        retrieved_chunk_ids=["unrelated:001"],
        selected_evidence=[selected],
    )

    assert result.retrieval_found_evidence is False
    assert result.answer_used_evidence is True
    assert result.answer_supported is True
    assert result.citations_correct is False


def test_rag_metrics_leave_downstream_checks_empty_after_abstention() -> None:
    """Verify that an abstention is recorded without inventing downstream failures."""
    result = evaluate_rag_case(
        gold_case(),
        GroundedAnswer(text="I do not have enough evidence.", abstained=True),
        retrieved_chunk_ids=[],
        selected_evidence=[],
    )

    assert result.retrieval_found_evidence is False
    assert result.answer_used_evidence is None
    assert result.answer_supported is None
    assert result.citations_correct is None
