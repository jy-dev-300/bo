import pytest

from ingestion.chunking import ChunkingPolicy
from ingestion.indexing import IndexingError, InMemoryDocumentIndex, PersistentDocumentIndex
from ingestion.models import NormalizedBlock, NormalizedDocument, SourceLocation


class FakeEmbeddingProvider:
    """Return deterministic vectors while exposing how often rebuilding occurred."""

    dimension = 2

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.document_calls = 0

    def embed_documents(self, texts):
        """Return one small vector per input text or simulate a provider failure."""
        self.document_calls += 1
        if self.fail:
            raise RuntimeError("embedding service unavailable")
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        """Return a query vector matching the declared dimension."""
        return [float(len(text)), 1.0]


def make_document(
    text: str,
    checksum: str,
    *,
    document_id: str = "doc-notes",
) -> NormalizedDocument:
    """Create a normalized document whose source identity remains constant."""
    source_path = "notes/current.txt"
    return NormalizedDocument(
        id=document_id,
        source_path=source_path,
        media_type="text/plain",
        checksum=checksum,
        blocks=[
            NormalizedBlock(
                kind="paragraph",
                text=text,
                location=SourceLocation(source_path=source_path, line_start=1, line_end=1),
            )
        ],
    )


def index_kwargs(provider):
    """Provide one repeatable set of pipeline and model versions."""
    return {
        "policy": ChunkingPolicy(target_size=200, overlap=0),
        "pipeline_version": "pipeline-v1",
        "embedding_provider": provider,
        "embedding_model_version": "tiny-v1",
    }


def test_unchanged_document_is_idempotent() -> None:
    """Verify that repeating the same operation creates no duplicates or model work."""
    store = InMemoryDocumentIndex()
    provider = FakeEmbeddingProvider()
    document = make_document("current policy", "a" * 64)

    first = store.index_document(document, **index_kwargs(provider))
    second = store.index_document(document, **index_kwargs(provider))

    assert first.action == "added"
    assert second.action == "skipped"
    assert store.document_count == 1
    assert len(store.chunks()) == 1
    assert provider.document_calls == 1


def test_update_atomically_removes_stale_chunks_and_vectors() -> None:
    """Verify that changed contents replace every artifact from the older version."""
    store = InMemoryDocumentIndex()
    provider = FakeEmbeddingProvider()
    old_document = make_document("obsolete launch plan", "a" * 64)
    store.index_document(old_document, **index_kwargs(provider))
    old_chunk_ids = {chunk.id for chunk in store.chunks()}

    result = store.update_document(
        make_document("current retirement plan", "b" * 64, document_id="new-content-id"),
        **index_kwargs(provider),
    )

    new_chunk_ids = {chunk.id for chunk in store.chunks()}
    assert result.action == "updated"
    assert old_chunk_ids.isdisjoint(new_chunk_ids)
    assert set(store.vector_entries()) == new_chunk_ids
    assert store.lexical_index().search("obsolete") == []
    assert store.lexical_index().search("retirement")[0].chunk.id in new_chunk_ids
    assert store.chunks()[0].document_id == old_document.id


def test_version_change_rebuilds_derived_artifacts() -> None:
    """Verify that model or pipeline changes rebuild unchanged file contents."""
    store = InMemoryDocumentIndex()
    provider = FakeEmbeddingProvider()
    document = make_document("stable content", "a" * 64)
    store.index_document(document, **index_kwargs(provider))

    result = store.index_document(
        document,
        policy=ChunkingPolicy(target_size=100, overlap=0),
        pipeline_version="pipeline-v2",
        embedding_provider=provider,
        embedding_model_version="tiny-v2",
    )

    assert result.action == "rebuilt"
    assert provider.document_calls == 2


def test_failed_update_is_visible_and_keeps_last_ready_version() -> None:
    """Verify that partial model failure cannot replace searchable ready artifacts."""
    store = InMemoryDocumentIndex()
    old_document = make_document("known good text", "a" * 64)
    store.index_document(old_document, **index_kwargs(FakeEmbeddingProvider()))
    ready_chunk_ids = [chunk.id for chunk in store.chunks()]

    with pytest.raises(IndexingError):
        store.update_document(
            make_document("uncommitted new text", "b" * 64),
            **index_kwargs(FakeEmbeddingProvider(fail=True)),
        )

    assert [chunk.id for chunk in store.chunks()] == ready_chunk_ids
    assert store.lexical_index().search("known")
    assert store.lexical_index().search("uncommitted") == []
    attempt = store.latest_attempt(old_document.source_path)
    assert attempt is not None
    assert attempt.status == "failed"
    assert "embedding service unavailable" in (attempt.error or "")


def test_delete_removes_document_chunks_lexical_entries_and_vectors() -> None:
    """Verify that deletion removes every searchable artifact for a source."""
    store = InMemoryDocumentIndex()
    document = make_document("delete this evidence", "a" * 64)
    store.index_document(document, **index_kwargs(FakeEmbeddingProvider()))

    result = store.delete_document(document.source_path)

    assert result.action == "deleted"
    assert store.document_count == 0
    assert store.chunks() == []
    assert store.vector_entries() == {}
    assert store.lexical_index().search("evidence") == []


def test_persistent_index_reloads_skips_unchanged_and_cleans_updates(tmp_path) -> None:
    """Verify restart reuse, incremental replacement, and on-disk vector cleanup."""
    file_index_dir = tmp_path / "file-index"
    embedding_dir = tmp_path / "embeddings"
    provider = FakeEmbeddingProvider()
    original = make_document("persistent launch plan", "a" * 64)
    store = PersistentDocumentIndex(file_index_dir, embedding_dir)

    first = store.index_document(original, **index_kwargs(provider))
    original_chunk_ids = {chunk.id for chunk in store.chunks()}

    assert first.action == "added"
    assert provider.document_calls == 1
    assert len(list(file_index_dir.glob("*.json"))) == 1
    assert len(list(embedding_dir.glob("*.npy"))) == 1

    reloaded = PersistentDocumentIndex(file_index_dir, embedding_dir)
    retry_provider = FakeEmbeddingProvider()
    unchanged = reloaded.index_document(original, **index_kwargs(retry_provider))

    assert unchanged.action == "skipped"
    assert retry_provider.document_calls == 0
    assert {chunk.id for chunk in reloaded.chunks()} == original_chunk_ids
    assert set(reloaded.vector_entries()) == original_chunk_ids

    updated = reloaded.index_document(
        make_document("persistent retirement plan", "b" * 64, document_id="changed-id"),
        **index_kwargs(retry_provider),
    )
    updated_chunk_ids = {chunk.id for chunk in reloaded.chunks()}

    assert updated.action == "updated"
    assert retry_provider.document_calls == 1
    assert updated_chunk_ids.isdisjoint(original_chunk_ids)
    assert len(list(embedding_dir.glob("*.npy"))) == 1

    reloaded_again = PersistentDocumentIndex(file_index_dir, embedding_dir)
    assert {chunk.id for chunk in reloaded_again.chunks()} == updated_chunk_ids
    assert reloaded_again.chunks()[0].document_id == original.id

    deleted = reloaded_again.delete_document(original.source_path)
    assert deleted.action == "deleted"
    assert list(file_index_dir.glob("*.json")) == []
    assert list(embedding_dir.glob("*.npy")) == []
