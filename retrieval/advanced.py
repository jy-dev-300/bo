from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

import numpy as np

from ingestion.models import Chunk
from rag.interfaces import Reranker
from rag.selection import rerank_candidates, text_similarity
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.lexical import BM25Index, tokenize
from retrieval.models import RetrievalCandidate
from retrieval.query_processing import MetadataFilter, chunk_matches_filters, expand_query

_BGE_MODEL_CACHE: dict[tuple[str, bool, str], Any] = {}
_BGE_MODEL_LOAD_LOCK = Lock()
_BGE_INFERENCE_LOCK = Lock()


def prefer_cached_hugging_face_model(model_name_or_path: str) -> str:
    """Use an already-downloaded model snapshot without contacting Hugging Face."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_name_or_path, local_files_only=True)
    except Exception:
        return model_name_or_path


def get_cached_bge_model(
    model_name: str,
    *,
    use_fp16: bool,
    devices: str | list[str] | None,
) -> Any:
    """Load one process-wide BGE model shared by indexing and query embedding."""
    device_key = repr(devices)
    cache_key = (model_name, use_fp16, device_key)
    with _BGE_MODEL_LOAD_LOCK:
        model = _BGE_MODEL_CACHE.get(cache_key)
        if model is None:
            from FlagEmbedding import BGEM3FlagModel

            model = BGEM3FlagModel(
                prefer_cached_hugging_face_model(model_name),
                use_fp16=use_fp16,
                devices=devices,
            )
            _BGE_MODEL_CACHE[cache_key] = model
    return model


class BGEDenseEmbeddingProvider:
    """Generate only BGE-M3 dense vectors for persistent indexing and query search."""

    dimension = 1024

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        model: Any | None = None,
        use_fp16: bool = False,
        devices: str | list[str] | None = None,
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> None:
        """Configure dense-only encoding while deferring model loading until needed."""
        self.model_name = model_name
        self._model = model
        self._use_fp16 = use_fp16
        self._devices = devices
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def version(self) -> str:
        """Identify the model and dense-only settings that make saved vectors reusable."""
        return f"{self.model_name}:dense:max_length={self._max_length}:v1"

    def _get_model(self) -> Any:
        """Reuse the same process-wide BGE model as all other BGE components."""
        if self._model is None:
            self._model = get_cached_bge_model(
                self.model_name,
                use_fp16=self._use_fp16,
                devices=self._devices,
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode chunk text once without generating sparse or ColBERT artifacts."""
        if not texts:
            return []
        with _BGE_INFERENCE_LOCK:
            output = self._get_model().encode(
                list(texts),
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        return np.asarray(output["dense_vecs"], dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode only the new search query against previously saved corpus vectors."""
        with _BGE_INFERENCE_LOCK:
            output = self._get_model().encode(
                text,
                batch_size=1,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        return np.asarray(output["dense_vecs"], dtype=np.float32).tolist()


def matches_filters(
    chunk: Chunk,
    filters: Sequence[MetadataFilter] | Mapping[str, object] | None,
) -> bool:
    """Apply the shared query-processing filters to one candidate chunk."""
    return chunk_matches_filters(chunk, filters)


def ranked_candidates(
    chunks: Sequence[Chunk],
    scores: Mapping[int, float],
    *,
    source: str,
    limit: int,
    details: Mapping[str, object],
) -> list[RetrievalCandidate]:
    """Convert index scores into a deterministic ranked candidate list."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], chunks[item[0]].id))[:limit]
    return [
        RetrievalCandidate(
            chunk=chunks[index],
            score=float(score),
            rank=rank,
            source=source,
            details=dict(details),
        )
        for rank, (index, score) in enumerate(ordered, start=1)
    ]


class BGEHybridIndex:
    """In-memory BGE-M3 dense and learned-sparse index."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model_name: str = "BAAI/bge-m3",
        model: Any | None = None,
        use_fp16: bool = False,
        devices: str | list[str] | None = None,
        batch_size: int = 8,
        max_length: int = 1024,
        late_interaction_pool: int = 100,
    ) -> None:
        """Configure the BGE-M3 index and defer model loading until search time."""
        self._chunks = list(chunks)
        self._model_name = model_name
        self._model = model
        self._use_fp16 = use_fp16
        self._devices = devices
        self._batch_size = batch_size
        self._max_length = max_length
        self._late_interaction_pool = late_interaction_pool
        self._document_embeddings: dict[str, Any] | None = None

    def _get_model(self) -> Any:
        """Load and cache the BGE-M3 model on first use."""
        if self._model is None:
            self._model = get_cached_bge_model(
                self._model_name,
                use_fp16=self._use_fp16,
                devices=self._devices,
            )
        return self._model

    def _get_document_embeddings(self) -> dict[str, Any]:
        """Encode and cache corpus chunks for dense and learned-sparse retrieval."""
        if self._document_embeddings is None:
            model = self._get_model()
            with _BGE_INFERENCE_LOCK:
                self._document_embeddings = model.encode(
                    [chunk.text for chunk in self._chunks],
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    return_dense=True,
                    return_sparse=True,
                    # ColBERT creates a large vector for every token; Bo does not use it.
                    return_colbert_vecs=False,
                )
        return self._document_embeddings

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Sequence[MetadataFilter] | Mapping[str, object] | None = None,
    ) -> dict[str, list[RetrievalCandidate]]:
        """Rank eligible chunks with dense and learned-sparse signals."""
        if limit <= 0 or not self._chunks:
            return {"vector": [], "sparse": [], "late_interaction": []}

        eligible = [
            index
            for index, chunk in enumerate(self._chunks)
            if matches_filters(chunk, filters)
        ]
        if not eligible:
            return {"vector": [], "sparse": [], "late_interaction": []}

        model = self._get_model()
        documents = self._get_document_embeddings()
        with _BGE_INFERENCE_LOCK:
            query_output = model.encode(
                query,
                batch_size=1,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=True,
                # Keep query output aligned with the document embeddings above.
                return_colbert_vecs=False,
            )
        dense_scores = {
            index: float(query_output["dense_vecs"] @ documents["dense_vecs"][index])
            for index in eligible
        }
        sparse_scores = {
            index: float(
                model.compute_lexical_matching_score(
                    query_output["lexical_weights"],
                    documents["lexical_weights"][index],
                )
            )
            for index in eligible
        }

        # Preserve the result key for caller compatibility while ColBERT is disabled.
        late_scores: dict[int, float] = {}
        common_details = {"model": self._model_name, "query": query}
        return {
            "vector": ranked_candidates(
                self._chunks,
                dense_scores,
                source="vector",
                limit=limit,
                details={**common_details, "method": "dense"},
            ),
            "sparse": ranked_candidates(
                self._chunks,
                sparse_scores,
                source="sparse",
                limit=limit,
                details={**common_details, "method": "learned_sparse"},
            ),
            "late_interaction": ranked_candidates(
                self._chunks,
                late_scores,
                source="late_interaction",
                limit=limit,
                details={**common_details, "method": "colbert_maxsim"},
            ),
        }


class BGEDenseRetriever:
    """Expose only BGE-M3's dense-vector results for a clean baseline comparison."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        semantic_index: BGEHybridIndex | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        embedding_provider: BGEDenseEmbeddingProvider | None = None,
    ) -> None:
        """Use saved dense vectors when supplied, or retain the legacy in-memory fallback."""
        self._chunks = list(chunks)
        self._embedding_provider = embedding_provider or BGEDenseEmbeddingProvider()
        self._semantic = None
        self._document_matrix: np.ndarray | None = None
        if embeddings is None:
            self._semantic = semantic_index or BGEHybridIndex(chunks)
        else:
            missing = [chunk.id for chunk in self._chunks if chunk.id not in embeddings]
            if missing:
                raise ValueError(f"missing saved dense embeddings for {len(missing)} chunks")
            if self._chunks:
                self._document_matrix = np.asarray(
                    [embeddings[chunk.id] for chunk in self._chunks],
                    dtype=np.float32,
                )
            else:
                self._document_matrix = np.empty(
                    (0, self._embedding_provider.dimension),
                    dtype=np.float32,
                )

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalCandidate]:
        """Return only the dense-vector candidate list for this query."""
        if self._semantic is not None:
            return self._semantic.search(query, limit=limit)["vector"]
        if limit <= 0 or not self._chunks or self._document_matrix is None:
            return []
        query_vector = np.asarray(self._embedding_provider.embed_query(query), dtype=np.float32)
        if query_vector.shape != (self._document_matrix.shape[1],):
            raise ValueError("query vector dimension does not match saved dense embeddings")
        scores = self._document_matrix @ query_vector
        return ranked_candidates(
            self._chunks,
            {index: float(score) for index, score in enumerate(scores)},
            source="vector",
            limit=limit,
            details={
                "model": self._embedding_provider.model_name,
                "query": query,
                "method": "dense_persisted",
            },
        )


class SpladeIndex:
    """Optional English SPLADE++ ranker over the same chunk IDs."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model_name: str = "prithivida/Splade_PP_en_v2",
        model: Any | None = None,
        batch_size: int = 8,
    ) -> None:
        """Configure the optional SPLADE index and defer loading its model."""
        self._chunks = list(chunks)
        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._document_embeddings: Any | None = None

    def _get_model(self) -> Any:
        """Load and cache the SPLADE sparse encoder on first use."""
        if self._model is None:
            from sentence_transformers import SparseEncoder

            self._model = SparseEncoder(self._model_name)
        return self._model

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Sequence[MetadataFilter] | Mapping[str, object] | None = None,
    ) -> list[RetrievalCandidate]:
        """Rank filter-matching chunks using SPLADE sparse similarity."""
        eligible = [
            index
            for index, chunk in enumerate(self._chunks)
            if matches_filters(chunk, filters)
        ]
        if limit <= 0 or not eligible:
            return []

        model = self._get_model()
        if self._document_embeddings is None:
            self._document_embeddings = model.encode_document(
                [chunk.text for chunk in self._chunks],
                batch_size=self._batch_size,
            )
        query_embedding = model.encode_query([query])
        similarity = model.similarity(query_embedding, self._document_embeddings)
        scores = {index: float(similarity[0][index]) for index in eligible}
        return ranked_candidates(
            self._chunks,
            scores,
            source="splade",
            limit=limit,
            details={"model": self._model_name, "method": "splade_sparse", "query": query},
        )


class BGECrossEncoderReranker:
    """Lazy BGE cross-encoder adapter implementing the project's Reranker protocol."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        model: Any | None = None,
        use_fp16: bool = False,
        devices: str | list[str] | None = None,
    ) -> None:
        """Configure the BGE cross-encoder and defer model loading until scoring."""
        self._model_name = model_name
        self._model = model
        self._use_fp16 = use_fp16
        self._devices = devices

    @property
    def details(self) -> dict[str, str]:
        """Describe the provider and model recorded beside reranked candidates."""
        return {"provider": "BAAI", "model": self._model_name, "primitive": "cross_encoder"}

    def _get_model(self) -> Any:
        """Load and cache the BGE cross-encoder on first use."""
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self._model_name,
                use_fp16=self._use_fp16,
                devices=self._devices,
            )
        return self._model

    def score(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[float]:
        """Return one normalized cross-encoder relevance score per candidate."""
        if not candidates:
            return []
        pairs = [[query, candidate.chunk.text] for candidate in candidates]
        scores = self._get_model().compute_score(pairs, normalize=True)
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]


def metadata_candidates(
    chunks: Sequence[Chunk],
    query: str,
    *,
    limit: int,
    filters: Sequence[MetadataFilter] | Mapping[str, object] | None = None,
) -> list[RetrievalCandidate]:
    """Rank chunks whose path or metadata contains exact query terms."""
    query_terms = set(tokenize(query))
    scores: dict[int, float] = {}
    for index, chunk in enumerate(chunks):
        if not matches_filters(chunk, filters):
            continue
        metadata_text = " ".join(
            str(value)
            for value in (
                chunk.metadata.source_path,
                chunk.metadata.section,
                chunk.metadata.language,
                chunk.metadata.created_at,
                *chunk.metadata.attributes.values(),
            )
            if value is not None
        )
        overlap = query_terms & set(tokenize(metadata_text))
        if overlap:
            scores[index] = float(len(overlap))

    return ranked_candidates(
        chunks,
        scores,
        source="metadata",
        limit=limit,
        details={"method": "metadata_exact_match", "query": query},
    )


def diversify_candidates(
    candidates: Sequence[RetrievalCandidate],
    *,
    limit: int,
    relevance_weight: float = 0.8,
) -> list[RetrievalCandidate]:
    """Apply deterministic MMR-like selection using lexical near-duplicate similarity."""
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between zero and one")
    if limit <= 0 or not candidates:
        return []

    highest = max(candidate.score for candidate in candidates)
    lowest = min(candidate.score for candidate in candidates)
    score_range = highest - lowest
    remaining = list(candidates)
    selected: list[RetrievalCandidate] = []

    while remaining and len(selected) < limit:
        scored: list[tuple[float, RetrievalCandidate]] = []
        for candidate in remaining:
            relevance = (
                1.0
                if score_range == 0
                else (candidate.score - lowest) / score_range
            )
            redundancy = max(
                (
                    text_similarity(candidate.chunk.text, chosen.chunk.text)
                    for chosen in selected
                ),
                default=0.0,
            )
            mmr_score = relevance_weight * relevance - (1.0 - relevance_weight) * redundancy
            scored.append((mmr_score, candidate))

        mmr_score, winner = min(
            scored,
            key=lambda item: (-item[0], item[1].rank, item[1].chunk.id),
        )
        selected.append(
            winner.model_copy(
                update={
                    "details": {**winner.details, "diversity_score": mmr_score},
                }
            )
        )
        remaining.remove(winner)

    return [
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(selected, start=1)
    ]


class PowerfulRetriever:
    """Configurable BM25 + neural retrieval stages for measured comparisons."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        semantic_index: BGEHybridIndex | None = None,
        splade_index: SpladeIndex | None = None,
        reranker: Reranker | None = None,
        use_splade: bool = False,
        use_reranker: bool = True,
        candidate_limit: int = 50,
        rank_constant: int = 60,
    ) -> None:
        """Assemble the configured retrieval legs without eagerly loading models."""
        self._chunks = list(chunks)
        self._lexical = BM25Index(self._chunks)
        self._semantic = semantic_index or BGEHybridIndex(self._chunks)
        self._splade = (splade_index or SpladeIndex(self._chunks)) if use_splade else None
        self._reranker = (reranker or BGECrossEncoderReranker()) if use_reranker else None
        self._candidate_limit = candidate_limit
        self._rank_constant = rank_constant

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: Sequence[MetadataFilter] | Mapping[str, object] | None = None,
        query_variants: Sequence[str] = (),
    ) -> list[RetrievalCandidate]:
        """Fuse enabled retrieval signals, optionally rerank, then diversify results."""
        variants = expand_query(query, query_variants)
        if not variants or limit <= 0:
            return []

        lexical_index = self._lexical
        if filters:
            lexical_index = BM25Index(
                [chunk for chunk in self._chunks if matches_filters(chunk, filters)]
            )

        rankings: dict[str, Sequence[RetrievalCandidate]] = {}
        for variant_index, variant in enumerate(variants):
            lexical = lexical_index.search(variant, limit=self._candidate_limit)
            rankings[f"query_{variant_index}:bm25"] = lexical

            semantic_rankings = self._semantic.search(
                variant,
                limit=self._candidate_limit,
                filters=filters,
            )
            for method, candidates in semantic_rankings.items():
                rankings[f"query_{variant_index}:{method}"] = candidates

            if self._splade is not None:
                rankings[f"query_{variant_index}:splade"] = self._splade.search(
                    variant,
                    limit=self._candidate_limit,
                    filters=filters,
                )

            rankings[f"query_{variant_index}:metadata"] = metadata_candidates(
                self._chunks,
                variant,
                limit=self._candidate_limit,
                filters=filters,
            )

        fused = reciprocal_rank_fusion(
            rankings,
            limit=self._candidate_limit,
            rank_constant=self._rank_constant,
        )

        # TODO(learned-ranking): once relevance labels exist, replace or augment
        # fixed RRF with a trained ranker over ranks, raw signals, and metadata.
        if self._reranker is None:
            return fused[:limit]

        reranked = rerank_candidates(query, fused, self._reranker)
        return diversify_candidates(reranked, limit=limit)
