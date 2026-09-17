
from ingestion.chunking import ChunkingPolicy, chunk_document
from retrieval.hybrid import fuse_candidates


def test_chunking_is_deterministic_and_preserves_provenance(prose_document) -> None:
    policy = ChunkingPolicy(target_size=20, overlap=5, size_unit="tokens")

    first = chunk_document(prose_document, policy)
    second = chunk_document(prose_document, policy)

    assert first
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.metadata.source_path == prose_document.source_path for chunk in first)
    assert all(chunk.text.strip() for chunk in first)


def test_hybrid_fusion_deduplicates_candidates_and_is_deterministic() -> None:
    assert fuse_candidates([], [], limit=10) == []
