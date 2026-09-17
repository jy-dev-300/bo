import numpy as np

from ingestion.models import Chunk, ChunkMetadata
from rag.selection import rerank_candidates, select_context
from retrieval.advanced import (
    BGECrossEncoderReranker,
    BGEHybridIndex,
    PowerfulRetriever,
    SpladeIndex,
)
from retrieval.lexical import BM25Index
from retrieval.stages import build_retriever


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=chunk_id.split(":")[0],
        ordinal=0,
        text=text,
        metadata=ChunkMetadata(source_path=f"{chunk_id}.txt"),
    )


class FakeBGE:
    def __init__(self) -> None:
        self.document_encodes = 0

    def encode(self, texts, **kwargs):
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
        return sum(weight * document.get(token, 0.0) for token, weight in query.items())

    def colbert_score(self, query, document):
        return float(document[0][0])


class FakeSplade:
    def __init__(self) -> None:
        self.document_encodes = 0

    def encode_document(self, texts, **kwargs):
        self.document_encodes += 1
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    def encode_query(self, queries):
        return np.array([[0.0, 1.0]])

    def similarity(self, query, documents):
        return query @ documents.T


class FakeCrossEncoder:
    def compute_score(self, pairs, **kwargs):
        return [1.0 if "other" in text else 0.1 for _, text in pairs]


def test_all_neural_legs_share_ids_and_cache_documents() -> None:
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
    assert first["late_interaction"][0].chunk.id == "b:0"
    assert all([candidate.chunk.id for candidate in leg] == ["b:0"] for leg in second.values())
    assert splade_first[0].chunk.id == "b:0"
    assert bge_model.document_encodes == 1
    assert splade_model.document_encodes == 1


def test_filters_excluding_everything_skip_neural_inference() -> None:
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


def test_advanced_pipeline_reranks_and_preserves_fusion_signals() -> None:
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
    class BadReranker:
        def score(self, query, candidates):
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


def test_stages_add_models_one_step_at_a_time_without_loading_them() -> None:
    chunks = [make_chunk("a:0", "alpha")]
    assert isinstance(build_retriever(chunks, "bm25"), BM25Index)

    hybrid = build_retriever(chunks, "hybrid")
    assert isinstance(hybrid, PowerfulRetriever)
    assert hybrid._reranker is None
    assert hybrid._splade is None

    rerank = build_retriever(chunks, "rerank")
    assert isinstance(rerank, PowerfulRetriever)
    assert rerank._reranker is not None
    assert rerank._splade is None

    specialists = build_retriever(chunks, "specialists")
    assert isinstance(specialists, PowerfulRetriever)
    assert specialists._reranker is not None
    assert specialists._splade is not None
