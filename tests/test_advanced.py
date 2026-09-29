import numpy as np

from ingestion.models import Chunk, ChunkMetadata
from rag.selection import JevReranker, rerank_candidates, select_context
from retrieval.advanced import (
    BGECrossEncoderReranker,
    BGEDenseEmbeddingProvider,
    BGEDenseRetriever,
    BGEHybridIndex,
    PowerfulRetriever,
    SpladeIndex,
    prefer_cached_hugging_face_model,
)
from retrieval.lexical import BM25Index
from retrieval.stages import build_retriever


def make_chunk(chunk_id: str, text: str) -> Chunk:
    """Create a minimal source chunk for advanced retrieval tests."""
    return Chunk(
        id=chunk_id,
        document_id=chunk_id.split(":")[0],
        ordinal=0,
        text=text,
        metadata=ChunkMetadata(source_path=f"{chunk_id}.txt"),
    )


def test_cached_hugging_face_snapshot_is_preferred(monkeypatch) -> None:
    """Verify that an available local snapshot avoids a model-network request."""
    import huggingface_hub

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool) -> str:
        assert repo_id == "BAAI/bge-m3"
        assert local_files_only is True
        return "cached/bge-m3"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert prefer_cached_hugging_face_model("BAAI/bge-m3") == "cached/bge-m3"


class FakeBGE:
    def __init__(self) -> None:
        """Initialize the counter used to verify document-embedding reuse."""
        self.document_encodes = 0
        self.encode_calls = []

    def encode(self, texts, **kwargs):
        """Return deterministic dense, sparse, and late-interaction fixture vectors."""
        self.encode_calls.append(kwargs)
        if isinstance(texts, list):
            self.document_encodes += 1
            return {
                "dense_vecs": np.array([[1.0, 0.0], [0.0, 1.0]]),
                "lexical_weights": [{"a": 1.0}, {"b": 1.0}],
                "colbert_vecs": [np.array([[0.1]]), np.array([[0.9]])],
            }
        return {
            "dense_vecs": np.array([1.0, 0.0]),
            "lexical_weights": {"a": 1.0},
            "colbert_vecs": np.array([[1.0]]),
        }

    def compute_lexical_matching_score(self, query, document):
        """Compute a small sparse dot product for the fake BGE model."""
        return sum(weight * document.get(token, 0.0) for token, weight in query.items())

    def colbert_score(self, query, document):
        """Return the fake document's single late-interaction value."""
        return float(document[0][0])


class FakeSplade:
    def __init__(self) -> None:
        """Initialize the counter used to verify sparse document-vector reuse."""
        self.document_encodes = 0

    def encode_document(self, texts, **kwargs):
        """Return deterministic sparse document vectors for the fixture corpus."""
        self.document_encodes += 1
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    def encode_query(self, queries):
        """Return a deterministic sparse query vector."""
        return np.array([[0.0, 1.0]])

    def similarity(self, query, documents):
        """Compute fake SPLADE similarity with a matrix product."""
        return query @ documents.T


class FakeCrossEncoder:
    def compute_score(self, pairs, **kwargs):
        """Prefer candidate texts containing the word used by the fixture."""
        return [1.0 if "other" in text else 0.1 for _, text in pairs]


class FakeJevAnswer:
    def __init__(self, noul: float) -> None:
        """Store one fake Noul probability."""
        self.noul = noul


class FakeJevResponse:
    def __init__(self, noul: float) -> None:
        """Expose a fake relevance answer using the SDK's response shape."""
        self.answers = {"is_relevant": FakeJevAnswer(noul)}


class FakeJevClient:
    def __init__(self) -> None:
        """Initialize a call log for assertions about Jev requests."""
        self.calls = []

    def system_one(self, *, state, questions, model):
        """Record a request and return a deterministic fake Jev response."""
        self.calls.append({"state": state, "questions": questions, "model": model})
        score = 0.9 if "other" in state["candidate_passage"] else 0.1
        return FakeJevResponse(score)


def test_all_neural_legs_share_ids_and_cache_documents() -> None:
    """Verify aligned chunk IDs, filtering, and cached document encodings."""
    chunks = [make_chunk("a:0", "alpha"), make_chunk("b:0", "other")]
    bge_model = FakeBGE()
    splade_model = FakeSplade()
    bge = BGEHybridIndex(chunks, model=bge_model)
    splade = SpladeIndex(chunks, model=splade_model)

    first = bge.search("alpha", limit=2)
    second = bge.search("alpha", limit=2, filters={"source_path": "b:0.txt"})
    splade_first = splade.search("alpha", limit=2)
    splade.search("alpha", limit=2)

    assert set(first) == {"vector", "sparse", "late_interaction"}
    assert first["vector"][0].chunk.id == "a:0"
    assert first["late_interaction"] == []
    assert [candidate.chunk.id for candidate in second["vector"]] == ["b:0"]
    assert [candidate.chunk.id for candidate in second["sparse"]] == ["b:0"]
    assert second["late_interaction"] == []
    assert splade_first[0].chunk.id == "b:0"
    assert bge_model.document_encodes == 1
    assert splade_model.document_encodes == 1


def test_filters_excluding_everything_skip_neural_inference() -> None:
    """Verify that empty filter results avoid unnecessary model calls."""
    chunks = [make_chunk("a:0", "alpha")]
    bge_model = FakeBGE()
    splade_model = FakeSplade()
    missing = {"source_path": "missing.txt"}

    assert BGEHybridIndex(chunks, model=bge_model).search(
        "alpha", limit=2, filters=missing
    ) == {"vector": [], "sparse": [], "late_interaction": []}
    assert SpladeIndex(chunks, model=splade_model).search(
        "alpha", limit=2, filters=missing
    ) == []
    assert bge_model.document_encodes == 0
    assert splade_model.document_encodes == 0


def test_bge_dense_stage_returns_only_vector_results() -> None:
    """Verify that the dense baseline does not mix in other BGE signals."""
    chunks = [make_chunk("a:0", "alpha"), make_chunk("b:0", "other")]
    retriever = BGEDenseRetriever(
        chunks,
        semantic_index=BGEHybridIndex(chunks, model=FakeBGE()),
    )

    results = retriever.search("alpha", limit=2)

    assert [candidate.source for candidate in results] == ["vector", "vector"]
    assert [candidate.chunk.id for candidate in results] == ["a:0", "b:0"]


def test_dense_provider_disables_sparse_and_colbert_outputs() -> None:
    """Verify persistent indexing asks BGE for dense vectors and nothing larger."""
    model = FakeBGE()
    provider = BGEDenseEmbeddingProvider(model=model)

    assert len(provider.embed_documents(["alpha", "other"])) == 2
    assert provider.embed_query("alpha") == [1.0, 0.0]
    assert all(call["return_dense"] is True for call in model.encode_calls)
    assert all(call["return_sparse"] is False for call in model.encode_calls)
    assert all(call["return_colbert_vecs"] is False for call in model.encode_calls)


def test_dense_retriever_uses_saved_documents_and_embeds_only_query() -> None:
    """Verify dense search ranks saved vectors without document encoding work."""
    chunks = [make_chunk("a:0", "alpha"), make_chunk("b:0", "other")]
    provider = BGEDenseEmbeddingProvider(model=FakeBGE())
    retriever = BGEDenseRetriever(
        chunks,
        embeddings={"a:0": (1.0, 0.0), "b:0": (0.0, 1.0)},
        embedding_provider=provider,
    )

    results = retriever.search("alpha", limit=2)

    assert [candidate.chunk.id for candidate in results] == ["a:0", "b:0"]
    assert provider._model.document_encodes == 0
    assert len(provider._model.encode_calls) == 1


def test_advanced_pipeline_reranks_and_preserves_fusion_signals() -> None:
    """Verify reranking order while retaining the fused retrieval details."""
    chunks = [make_chunk("a:0", "alpha"), make_chunk("b:0", "other")]
    retriever = PowerfulRetriever(
        chunks,
        semantic_index=BGEHybridIndex(chunks, model=FakeBGE()),
        splade_index=SpladeIndex(chunks, model=FakeSplade()),
        reranker=BGECrossEncoderReranker(model=FakeCrossEncoder()),
        use_splade=True,
    )

    results = retriever.search("alpha", limit=2)

    assert [candidate.chunk.id for candidate in results] == ["b:0", "a:0"]
    assert [candidate.rank for candidate in results] == [1, 2]
    assert results[0].source == "reranker"
    assert "query_0:splade" in results[0].details["signals"]
    assert "pre_rerank" in results[0].details


def test_context_selection_deduplicates_and_respects_character_budget() -> None:
    """Verify that context selection removes duplicates and honors its budget."""
    chunks = [make_chunk("a:0", "alpha"), make_chunk("b:0", "alpha")]
    retriever = PowerfulRetriever(
        chunks,
        semantic_index=BGEHybridIndex(chunks, model=FakeBGE()),
        splade_index=SpladeIndex(chunks, model=FakeSplade()),
        reranker=BGECrossEncoderReranker(model=FakeCrossEncoder()),
        use_splade=True,
    )
    candidates = retriever.search("alpha", limit=2)
    selected = select_context(candidates, budget=6)

    assert len(selected) == 1
    assert selected[0].budget_cost == 5


def test_reranker_rejects_wrong_number_of_scores() -> None:
    """Verify that every candidate must receive exactly one reranker score."""
    class BadReranker:
        def score(self, query, candidates):
            """Return no scores to deliberately violate the reranker contract."""
            return []

    candidate = PowerfulRetriever(
        [make_chunk("a:0", "alpha")], use_splade=False
    )._lexical.search("alpha", limit=1)[0]
    try:
        rerank_candidates("alpha", [candidate], BadReranker())
    except ValueError as error:
        assert "one score per candidate" in str(error)
    else:
        raise AssertionError("expected a validation error")


def test_jev_reranker_uses_noul_probabilities_and_preserves_provider_details() -> None:
    """Verify Jev ordering and the provider details attached to results."""
    candidates = PowerfulRetriever(
        [make_chunk("a:0", "alpha"), make_chunk("b:0", "other")],
        use_reranker=False,
    )._lexical.search("alpha other", limit=2)
    client = FakeJevClient()
    reranker = JevReranker(client=client, question=object(), max_workers=1)

    results = rerank_candidates("alpha other", candidates, reranker)

    assert [candidate.chunk.id for candidate in results] == ["b:0", "a:0"]
    assert [call["model"] for call in client.calls] == ["jev-latest", "jev-latest"]
    assert results[0].details["reranker"] == {
        "adapter": "JevReranker",
        "provider": "typesafe",
        "model": "jev-latest",
        "primitive": "noul",
    }


def test_stages_add_models_one_step_at_a_time_without_loading_them() -> None:
    """Verify which model adapters are enabled by each retrieval stage."""
    chunks = [make_chunk("a:0", "alpha")]
    assert isinstance(build_retriever(chunks, "bm25"), BM25Index)

    dense = build_retriever(chunks, "bge_dense")
    assert isinstance(dense, BGEDenseRetriever)

    hybrid = build_retriever(chunks, "hybrid")
    assert isinstance(hybrid, PowerfulRetriever)
    assert hybrid._reranker is None
    assert hybrid._splade is None

    rerank = build_retriever(chunks, "rerank")
    assert isinstance(rerank, PowerfulRetriever)
    assert rerank._reranker is not None
    assert rerank._splade is None

    rerank_bge = build_retriever(chunks, "rerank_bge")
    assert isinstance(rerank_bge, PowerfulRetriever)
    assert isinstance(rerank_bge._reranker, BGECrossEncoderReranker)

    rerank_jev = build_retriever(chunks, "rerank_jev")
    assert isinstance(rerank_jev, PowerfulRetriever)
    assert isinstance(rerank_jev._reranker, JevReranker)

    specialists = build_retriever(chunks, "specialists")
    assert isinstance(specialists, PowerfulRetriever)
    assert specialists._reranker is not None
    assert specialists._splade is not None
