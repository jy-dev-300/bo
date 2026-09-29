import pytest

from ingestion.models import Chunk, ChunkMetadata
from rag.citations import assemble_citations
from rag.models import SelectedEvidence
from retrieval.models import RetrievalCandidate


def selected_evidence() -> SelectedEvidence:
    """Create one selected PDF chunk with provenance for citation tests."""
    chunk = Chunk(
        id="doc-march:000",
        document_id="doc-march",
        ordinal=0,
        text="The March report contains an orange chart showing project status.",
        metadata=ChunkMetadata(
            source_path="reports/march.pdf",
            page=3,
            section="Project status",
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
        budget_cost=len(chunk.text),
    )


def test_assemble_citations_validates_id_and_uses_chunk_provenance() -> None:
    """Verify that a valid marker produces a citation from trusted chunk metadata."""
    citations = assemble_citations(
        "The March report contains an orange chart [doc-march:000].",
        [selected_evidence()],
    )

    assert len(citations) == 1
    assert citations[0].claim == "The March report contains an orange chart"
    assert citations[0].chunk_id == "doc-march:000"
    assert citations[0].source_path == "reports/march.pdf"
    assert citations[0].page == 3
    assert citations[0].section == "Project status"


def test_assemble_citations_rejects_id_outside_selected_evidence() -> None:
    """Verify that an invented or unselected chunk ID is rejected."""
    with pytest.raises(ValueError, match="not present in selected evidence"):
        assemble_citations("The report has a chart [invented:999].", [selected_evidence()])


def test_assemble_citations_rejects_unsupported_claim() -> None:
    """Verify that a cited chunk must textually support its associated claim."""
    with pytest.raises(ValueError, match="does not support"):
        assemble_citations(
            "Quarterly revenue doubled after the acquisition [doc-march:000].",
            [selected_evidence()],
        )
