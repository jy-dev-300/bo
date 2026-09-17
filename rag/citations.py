from collections.abc import Sequence

from rag.models import Citation, SelectedEvidence


def assemble_citations(
    answer_text: str,
    evidence: Sequence[SelectedEvidence],
) -> list[Citation]:
    """Student Implementation D: validate claim-to-evidence citation mappings."""
    raise NotImplementedError("Student Implementation D: grounded citation assembly")

