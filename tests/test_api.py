import base64

from fastapi.testclient import TestClient

from backend.main import app


def test_health() -> None:
    """Verify that the health endpoint reports a running service."""
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_search_exposes_source_provenance() -> None:
    """Verify that API search hits retain their chunk and source identifiers."""
    with TestClient(app) as client:
        response = client.get("/api/search", params={"q": "Korean address"})

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_method"] == "bm25"
    assert body["hits"][0]["chunk_id"] == "doc-seoul-address:000"
    assert body["hits"][0]["source_path"].endswith("seoul_address_note.txt")


def test_search_can_report_a_testing_stage() -> None:
    """Verify that the dashboard can choose and identify a retrieval stage."""
    with TestClient(app) as client:
        app.state.retrievers["hybrid"] = app.state.retriever
        response = client.get(
            "/api/search",
            params={"q": "orange chart", "stage": "hybrid"},
        )

    assert response.status_code == 200
    assert response.json()["retrieval_method"] == "hybrid"


def test_search_exposes_query_processing_and_applies_explicit_filters() -> None:
    """Verify that the API traces variants and applies one filter across retrieval stages."""
    with TestClient(app) as client:
        response = client.get(
            "/api/search",
            params={"q": "find the Python file where I implemented retries", "stage": "bm25"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query_variants"][0] == "find the Python file where I implemented retries"
    assert any(
        filter_["field"] == "file_extension" and filter_["value"] == ".py"
        for filter_ in body["metadata_filters"]
    )
    assert [hit["chunk_id"] for hit in body["hits"]] == ["doc-retry-example:000"]


def test_search_can_use_browser_selected_local_files() -> None:
    """Verify that a selected file becomes a temporary corpus used by search."""
    text = b"My private telescope calibration phrase is amber nebula seven."
    payload = {
        "files": [
            {
                "relative_path": "personal/astronomy-notes.txt",
                "media_type": "text/plain",
                "content_base64": base64.b64encode(text).decode("ascii"),
            }
        ]
    }

    with TestClient(app) as client:
        corpus_response = client.post("/api/testing/corpus", json=payload)
        assert corpus_response.status_code == 200
        corpus = corpus_response.json()
        search_response = client.get(
            "/api/search",
            params={
                "q": "amber nebula seven",
                "stage": "bm25",
                "corpus_id": corpus["corpus_id"],
            },
        )

    assert corpus["files_indexed"] == 1
    assert corpus["chunks_indexed"] == 1
    assert search_response.status_code == 200
    assert search_response.json()["hits"][0]["source_path"] == "personal/astronomy-notes.txt"


def test_bge_dense_corpus_survives_backend_restart_without_reembedding() -> None:
    """Verify a restarted API loads saved chunks and embeds only the next query."""
    text = b"A persistent observatory calibration record."
    payload = {
        "files": [
            {
                "relative_path": "personal/persistent-notes.txt",
                "media_type": "text/plain",
                "content_base64": base64.b64encode(text).decode("ascii"),
            }
        ]
    }

    with TestClient(app) as client:
        corpus = client.post("/api/testing/corpus", json=payload).json()
        assert app.state.dense_embedding_provider.document_calls == 1

    with TestClient(app) as restarted_client:
        response = restarted_client.get(
            "/api/search",
            params={
                "q": "observatory calibration",
                "stage": "bge_dense",
                "corpus_id": corpus["corpus_id"],
            },
        )

    assert response.status_code == 200
    assert response.json()["hits"][0]["source_path"] == "personal/persistent-notes.txt"
    assert app.state.dense_embedding_provider.document_calls == 1
    assert app.state.dense_embedding_provider.query_calls == 1


def test_unknown_local_corpus_is_rejected() -> None:
    """Verify that expired in-memory corpus identifiers fail clearly."""
    with TestClient(app) as client:
        response = client.get(
            "/api/search",
            params={"q": "anything", "corpus_id": "missing-corpus"},
        )

    assert response.status_code == 404


def test_local_corpus_can_be_extended_one_file_at_a_time() -> None:
    """Verify that progress-friendly uploads accumulate in one searchable corpus."""
    first_file = {
        "relative_path": "notes/first.txt",
        "media_type": "text/plain",
        "content_base64": base64.b64encode(b"first lighthouse note").decode("ascii"),
    }
    second_file = {
        "relative_path": "notes/second.txt",
        "media_type": "text/plain",
        "content_base64": base64.b64encode(b"second observatory note").decode("ascii"),
    }

    with TestClient(app) as client:
        first_response = client.post("/api/testing/corpus", json={"files": [first_file]})
        corpus_id = first_response.json()["corpus_id"]
        second_response = client.post(
            "/api/testing/corpus",
            json={"corpus_id": corpus_id, "files": [second_file]},
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["corpus_id"] == corpus_id
    assert second_response.json()["files_received"] == 2
    assert second_response.json()["files_indexed"] == 2
    assert second_response.json()["chunks_indexed"] == 2


def test_new_file_snapshot_removes_sources_no_longer_selected() -> None:
    """Verify a complete browser path snapshot prunes persisted files that disappeared."""
    first_file = {
        "relative_path": "notes/keep.txt",
        "media_type": "text/plain",
        "content_base64": base64.b64encode(b"keep this lighthouse note").decode("ascii"),
    }
    removed_file = {
        "relative_path": "notes/remove.txt",
        "media_type": "text/plain",
        "content_base64": base64.b64encode(b"remove this observatory note").decode("ascii"),
    }

    with TestClient(app) as client:
        first_response = client.post(
            "/api/testing/corpus",
            json={"files": [first_file], "source_paths": ["notes/keep.txt", "notes/remove.txt"]},
        )
        corpus_id = first_response.json()["corpus_id"]
        client.post(
            "/api/testing/corpus",
            json={"corpus_id": corpus_id, "files": [removed_file]},
        )
        pruned_response = client.post(
            "/api/testing/corpus",
            json={
                "corpus_id": corpus_id,
                "files": [first_file],
                "source_paths": ["notes/keep.txt"],
            },
        )
        removed_search = client.get(
            "/api/search",
            params={"q": "observatory", "stage": "bm25", "corpus_id": corpus_id},
        )

    assert pruned_response.json()["files_indexed"] == 1
    assert removed_search.json()["hits"] == []


def test_empty_file_is_reported_without_breaking_the_indexing_session() -> None:
    """Verify that an empty folder entry is skipped and later files still index."""
    empty_file = {
        "relative_path": "notes/empty.txt",
        "media_type": "text/plain",
        "content_base64": "",
    }
    valid_file = {
        "relative_path": "notes/useful.txt",
        "media_type": "text/plain",
        "content_base64": base64.b64encode(b"useful local note").decode("ascii"),
    }

    with TestClient(app) as client:
        empty_response = client.post("/api/testing/corpus", json={"files": [empty_file]})
        corpus_id = empty_response.json()["corpus_id"]
        valid_response = client.post(
            "/api/testing/corpus",
            json={"corpus_id": corpus_id, "files": [valid_file]},
        )

    assert empty_response.status_code == 200
    assert empty_response.json()["files_indexed"] == 0
    assert "file contains no searchable text" in empty_response.json()["skipped_files"][0]
    assert valid_response.status_code == 200
    assert valid_response.json()["files_indexed"] == 1


def test_skipped_file_does_not_block_or_enter_the_corpus() -> None:
    """Verify server-side cancellation rejects the skipped file and accepts the next one."""
    skipped_file = {
        "relative_path": "notes/slow.pdf",
        "media_type": "application/pdf",
        "content_base64": base64.b64encode(b"not parsed because it was cancelled").decode("ascii"),
    }
    next_file = {
        "relative_path": "notes/next.txt",
        "media_type": "text/plain",
        "content_base64": base64.b64encode(b"next file indexed normally").decode("ascii"),
    }

    with TestClient(app) as client:
        corpus = client.post("/api/testing/corpus", json={"files": []}).json()
        skip_response = client.post(
            f"/api/testing/corpus/{corpus['corpus_id']}/skip",
            json={"relative_path": skipped_file["relative_path"]},
        )
        skipped_response = client.post(
            "/api/testing/corpus",
            json={"corpus_id": corpus["corpus_id"], "files": [skipped_file]},
        )
        next_response = client.post(
            "/api/testing/corpus",
            json={"corpus_id": corpus["corpus_id"], "files": [next_file]},
        )

    assert skip_response.status_code == 202
    assert skipped_response.json()["files_indexed"] == 0
    assert "skipped by user" in skipped_response.json()["skipped_files"][0]
    assert next_response.json()["files_indexed"] == 1
