"""Atomic in-memory add, update, rebuild, and delete behavior for derived indexes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

import numpy as np

from ingestion.chunking import ChunkingPolicy, chunk_document
from ingestion.models import Chunk, NormalizedDocument
from retrieval.interfaces import EmbeddingProvider
from retrieval.lexical import BM25Index


@dataclass(frozen=True)
class IndexVersions:
    """Identify every implementation choice that can make derived data stale."""

    pipeline: str
    chunking: str
    embedding_model: str | None


@dataclass(frozen=True)
class IndexedDocument:
    """Keep one ready document and all of its derived artifacts together."""

    document: NormalizedDocument
    chunks: tuple[Chunk, ...]
    embeddings: dict[str, tuple[float, ...]]
    versions: IndexVersions


@dataclass(frozen=True)
class IndexAttempt:
    """Expose whether the latest indexing attempt completed or failed."""

    source_path: str
    status: Literal["processing", "ready", "failed", "deleted"]
    started_at: str
    finished_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class IndexResult:
    """Summarize the lifecycle decision made for one source file."""

    source_path: str
    action: Literal["added", "updated", "rebuilt", "skipped", "deleted", "missing"]
    chunk_count: int
    embedding_count: int


class IndexingError(RuntimeError):
    """Report a failed rebuild while leaving the last ready version searchable."""


class IndexingCancelled(IndexingError):
    """Report that indexing was cancelled before its derived data was committed."""


def chunking_version(policy: ChunkingPolicy) -> str:
    """Create a stable version fingerprint from every chunking-policy setting."""
    payload = json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class InMemoryDocumentIndex:
    """Store ready artifacts atomically and retain explicit processing state."""

    def __init__(self) -> None:
        """Create an empty lifecycle store protected against concurrent updates."""
        self._records: dict[str, IndexedDocument] = {}
        self._attempts: dict[str, IndexAttempt] = {}
        self._revision = 0
        self._lock = RLock()

    @property
    def document_count(self) -> int:
        """Return the number of source files with a ready indexed version."""
        with self._lock:
            return len(self._records)

    def get(self, source_path: str) -> IndexedDocument | None:
        """Return the ready artifacts for one source path when present."""
        return self._records.get(source_path)

    def latest_attempt(self, source_path: str) -> IndexAttempt | None:
        """Return the visible processing state for one source path."""
        return self._attempts.get(source_path)

    def chunks(self) -> list[Chunk]:
        """Return all ready chunks in deterministic source and ordinal order."""
        return self.snapshot()[1]

    def source_paths(self) -> set[str]:
        """Return the original paths represented by completed index records."""
        with self._lock:
            return set(self._records)

    def snapshot(self) -> tuple[int, list[Chunk]]:
        """Return one consistent corpus revision and its completed chunks."""
        with self._lock:
            chunks = [
                chunk
                for source_path in sorted(self._records)
                for chunk in sorted(
                    self._records[source_path].chunks,
                    key=lambda item: item.ordinal,
                )
            ]
            return self._revision, chunks

    def vector_entries(self) -> dict[str, tuple[float, ...]]:
        """Return a copy of every ready chunk embedding keyed by chunk ID."""
        return {
            chunk_id: vector
            for source_path in sorted(self._records)
            for chunk_id, vector in self._records[source_path].embeddings.items()
        }

    def lexical_index(self) -> BM25Index:
        """Build a lexical index only from current ready chunks."""
        return BM25Index(self.chunks())

    def _persist_record(self, record: IndexedDocument) -> None:
        """Provide a commit hook for stores that persist completed artifacts."""

    def _delete_persisted_record(self, source_path: str) -> None:
        """Provide a deletion hook for stores that persist completed artifacts."""

    def index_document(
        self,
        document: NormalizedDocument,
        *,
        policy: ChunkingPolicy,
        pipeline_version: str,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model_version: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> IndexResult:
        """Add, update, rebuild, or skip one source as a single atomic operation."""
        source_path = document.source_path
        versions = IndexVersions(
            pipeline=pipeline_version,
            chunking=chunking_version(policy),
            embedding_model=embedding_model_version,
        )
        with self._lock:
            current = self._records.get(source_path)
            if (
                current is not None
                and current.document.checksum == document.checksum
                and current.versions == versions
            ):
                return IndexResult(
                    source_path=source_path,
                    action="skipped",
                    chunk_count=len(current.chunks),
                    embedding_count=len(current.embeddings),
                )

            started_at = datetime.now(UTC).isoformat()
            self._attempts[source_path] = IndexAttempt(
                source_path=source_path,
                status="processing",
                started_at=started_at,
            )

        try:
            stable_document = document
            if current is not None and document.id != current.document.id:
                stable_document = document.model_copy(update={"id": current.document.id})
            chunks = tuple(chunk_document(stable_document, policy))
            if not chunks:
                raise ValueError("document produced no searchable chunks")
            embeddings: dict[str, tuple[float, ...]] = {}
            if embedding_provider is not None and chunks:
                vectors = embedding_provider.embed_documents([chunk.text for chunk in chunks])
                if len(vectors) != len(chunks):
                    raise ValueError("embedding provider must return one vector per chunk")
                embeddings = {
                    chunk.id: tuple(float(value) for value in vector)
                    for chunk, vector in zip(chunks, vectors, strict=True)
                }
            if cancelled is not None and cancelled():
                raise IndexingCancelled(f"Indexing cancelled for {source_path}")

            replacement = IndexedDocument(
                document=stable_document,
                chunks=chunks,
                embeddings=embeddings,
                versions=versions,
            )
            with self._lock:
                if cancelled is not None and cancelled():
                    raise IndexingCancelled(f"Indexing cancelled for {source_path}")
                self._persist_record(replacement)
                self._records[source_path] = replacement
                self._revision += 1
                self._attempts[source_path] = IndexAttempt(
                    source_path=source_path,
                    status="ready",
                    started_at=started_at,
                    finished_at=datetime.now(UTC).isoformat(),
                )
        except Exception as exc:
            with self._lock:
                self._attempts[source_path] = IndexAttempt(
                    source_path=source_path,
                    status="failed",
                    started_at=started_at,
                    finished_at=datetime.now(UTC).isoformat(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            if isinstance(exc, IndexingCancelled):
                raise
            raise IndexingError(f"Indexing failed for {source_path}: {exc}") from exc

        if current is None:
            action: Literal["added", "updated", "rebuilt"] = "added"
        elif current.document.checksum != document.checksum:
            action = "updated"
        else:
            action = "rebuilt"
        return IndexResult(
            source_path=source_path,
            action=action,
            chunk_count=len(chunks),
            embedding_count=len(embeddings),
        )

    def update_document(
        self,
        document: NormalizedDocument,
        *,
        policy: ChunkingPolicy,
        pipeline_version: str,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model_version: str | None = None,
    ) -> IndexResult:
        """Apply the same idempotent lifecycle logic to a changed source file."""
        return self.index_document(
            document,
            policy=policy,
            pipeline_version=pipeline_version,
            embedding_provider=embedding_provider,
            embedding_model_version=embedding_model_version,
        )

    def delete_document(self, source_path: str) -> IndexResult:
        """Remove a source and every chunk, lexical entry, and vector derived from it."""
        with self._lock:
            current = self._records.get(source_path)
            timestamp = datetime.now(UTC).isoformat()
            self._attempts[source_path] = IndexAttempt(
                source_path=source_path,
                status="deleted",
                started_at=timestamp,
                finished_at=timestamp,
            )
            if current is None:
                return IndexResult(source_path, "missing", 0, 0)
            self._delete_persisted_record(source_path)
            self._records.pop(source_path)
            self._revision += 1
            return IndexResult(
                source_path,
                "deleted",
                len(current.chunks),
                len(current.embeddings),
            )


class PersistentDocumentIndex(InMemoryDocumentIndex):
    """Persist existing document, chunk, provenance, and dense-vector records locally."""

    schema_version = 1

    def __init__(
        self,
        file_index_dir: Path,
        embedding_dir: Path,
        *,
        embedding_dtype: str = "float16",
    ) -> None:
        """Load every previously completed per-file record from the two index directories."""
        super().__init__()
        self._file_index_dir = Path(file_index_dir)
        self._embedding_dir = Path(embedding_dir)
        self._embedding_dtype = np.dtype(embedding_dtype)
        self._file_index_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_dir.mkdir(parents=True, exist_ok=True)
        self._load_records()

    @staticmethod
    def _source_key(source_path: str) -> str:
        """Create a filesystem-safe stable key without changing source identity."""
        return hashlib.sha256(source_path.encode("utf-8")).hexdigest()

    def _record_path(self, source_path: str) -> Path:
        """Return the JSON record path for one original source path."""
        return self._file_index_dir / f"{self._source_key(source_path)}.json"

    @staticmethod
    def _embedding_key(record: IndexedDocument) -> str:
        """Name a dense matrix by the exact chunks and embedding model that produced it."""
        identity = {
            "chunk_ids": [chunk.id for chunk in record.chunks],
            "embedding_model": record.versions.embedding_model,
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
        """Replace one metadata record only after its complete JSON is on disk."""
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_numpy_write(path: Path, matrix: np.ndarray) -> None:
        """Replace one dense matrix only after NumPy finishes writing every row."""
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as file:
                np.save(file, matrix, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_records(self) -> None:
        """Restore completed records while ignoring interrupted or invalid artifacts."""
        for record_path in sorted(self._file_index_dir.glob("*.json")):
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != self.schema_version:
                    continue
                document = NormalizedDocument.model_validate(payload["document"])
                chunks = tuple(Chunk.model_validate(item) for item in payload["chunks"])
                versions = IndexVersions(**payload["versions"])
                chunk_ids = payload.get("embedding_chunk_ids", [])
                embedding_file = payload.get("embedding_file")
                embeddings: dict[str, tuple[float, ...]] = {}
                if embedding_file is not None:
                    matrix_path = self._embedding_dir / str(embedding_file)
                    matrix = np.load(matrix_path, allow_pickle=False)
                    if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids):
                        raise ValueError("dense matrix rows do not match stored chunk IDs")
                    embeddings = {
                        chunk_id: tuple(float(value) for value in row)
                        for chunk_id, row in zip(chunk_ids, matrix, strict=True)
                    }
                if set(embeddings) - {chunk.id for chunk in chunks}:
                    raise ValueError("dense matrix contains an unknown chunk ID")
                self._records[document.source_path] = IndexedDocument(
                    document=document,
                    chunks=chunks,
                    embeddings=embeddings,
                    versions=versions,
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self._revision = len(self._records)

    def _persist_record(self, record: IndexedDocument) -> None:
        """Atomically publish a dense matrix before its metadata starts referring to it."""
        record_path = self._record_path(record.document.source_path)
        old_embedding_file: str | None = None
        if record_path.exists():
            try:
                old_payload = json.loads(record_path.read_text(encoding="utf-8"))
                old_embedding_file = old_payload.get("embedding_file")
            except (OSError, json.JSONDecodeError):
                pass

        embedding_chunk_ids = [
            chunk.id for chunk in record.chunks if chunk.id in record.embeddings
        ]
        embedding_file: str | None = None
        if embedding_chunk_ids:
            embedding_file = f"{self._embedding_key(record)}.npy"
            matrix_path = self._embedding_dir / embedding_file
            matrix = np.asarray(
                [record.embeddings[chunk_id] for chunk_id in embedding_chunk_ids],
                dtype=self._embedding_dtype,
            )
            self._atomic_numpy_write(matrix_path, matrix)

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            # Chunks already contain the searchable text, so do not duplicate normalized blocks.
            "document": record.document.model_copy(update={"blocks": []}).model_dump(mode="json"),
            "chunks": [chunk.model_dump(mode="json") for chunk in record.chunks],
            "versions": asdict(record.versions),
            "embedding_chunk_ids": embedding_chunk_ids,
            "embedding_file": embedding_file,
            "embedding_dtype": self._embedding_dtype.name,
        }
        self._atomic_json_write(record_path, payload)
        if old_embedding_file and old_embedding_file != embedding_file:
            (self._embedding_dir / old_embedding_file).unlink(missing_ok=True)

    def _delete_persisted_record(self, source_path: str) -> None:
        """Delete one source record and the dense matrix referenced only by that record."""
        record_path = self._record_path(source_path)
        embedding_file: str | None = None
        if record_path.exists():
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
                embedding_file = payload.get("embedding_file")
            except (OSError, json.JSONDecodeError):
                pass
        record_path.unlink(missing_ok=True)
        if embedding_file:
            (self._embedding_dir / embedding_file).unlink(missing_ok=True)


def index_document(
    store: InMemoryDocumentIndex,
    document: NormalizedDocument,
    *,
    policy: ChunkingPolicy,
    pipeline_version: str,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_model_version: str | None = None,
) -> IndexResult:
    """Index one document through the shared lifecycle store."""
    return store.index_document(
        document,
        policy=policy,
        pipeline_version=pipeline_version,
        embedding_provider=embedding_provider,
        embedding_model_version=embedding_model_version,
    )


def update_document(
    store: InMemoryDocumentIndex,
    document: NormalizedDocument,
    *,
    policy: ChunkingPolicy,
    pipeline_version: str,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_model_version: str | None = None,
) -> IndexResult:
    """Replace stale artifacts for one existing source file."""
    return store.update_document(
        document,
        policy=policy,
        pipeline_version=pipeline_version,
        embedding_provider=embedding_provider,
        embedding_model_version=embedding_model_version,
    )


def delete_document(store: InMemoryDocumentIndex, source_path: str) -> IndexResult:
    """Delete one source and all derived artifacts from the store."""
    return store.delete_document(source_path)
