from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ingestion.models import Chunk
from rag.selection import rerank_candidates, text_similarity
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.lexical import BM25Index, tokenize
from retrieval.models import RetrievalCandidate
from retrieval.query_processing import expand_query


def matches_filters(chunk: Chunk, filters: Mapping[str, object] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = getattr(chunk.metadata, key, chunk.metadata.attributes.get(key))
        if actual != expected:
            return False
    return True


def ranked_candidates(
    chunks: Sequence[Chunk],
    scores: Mapping[int, float],
    *,
    source: str,
    limit: int,
    details: Mapping[str, object],
) -> list[RetrievalCandidate]:
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
    """In-memory BGE-M3 dense, learned-sparse, and late-interaction index."""

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
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=self._use_fp16,
                devices=self._devices,
            )
        return self._model

    def _get_document_embeddings(self) -> dict[str, Any]:
        if self._document_embeddings is None:
            model = self._get_model()
            self._document_embeddings = model.encode(
                [chunk.text for chunk in self._chunks],
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
        return self._document_embeddings

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, object] | None = None,
    ) -> dict[str, list[RetrievalCandidate]]:
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
        query_output = model.encode(
            query,
            batch_size=1,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
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

        dense_order = sorted(
            dense_scores,
            key=lambda index: (-dense_scores[index], self._chunks[index].id),
        )
        sparse_order = sorted(
            sparse_scores,
            key=lambda index: (-sparse_scores[index], self._chunks[index].id),
        )
        if len(eligible) <= self._late_interaction_pool:
            late_indices = eligible
        else:
            half_pool = max(1, self._late_interaction_pool // 2)
            late_indices = list(
                dict.fromkeys([*dense_order[:half_pool], *sparse_order[:half_pool]])
            )

        late_scores = {
            index: float(
                model.colbert_score(
                    query_output["colbert_vecs"],
                    documents["colbert_vecs"][index],
                )
            )
            for index in late_indices
        }
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
        self._chunks = list(chunks)
        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._document_embeddings: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SparseEncoder

            self._model = SparseEncoder(self._model_name)
        return self._model

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[RetrievalCandidate]:
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
        self._model_name = model_name
        self._model = model
        self._use_fp16 = use_fp16
        self._devices = devices

    def _get_model(self) -> Any:
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
    filters: Mapping[str, object] | None = None,
) -> list[RetrievalCandidate]:
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
        reranker: BGECrossEncoderReranker | None = None,
        use_splade: bool = False,
        use_reranker: bool = True,
        candidate_limit: int = 50,
        rank_constant: int = 60,
    ) -> None:
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
        filters: Mapping[str, object] | None = None,
        query_variants: Sequence[str] = (),
    ) -> list[RetrievalCandidate]:
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
