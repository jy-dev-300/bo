from fastapi.testclient import TestClient

from backend.main import app


def test_health() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_search_exposes_source_provenance() -> None:
    with TestClient(app) as client:
        response = client.get("/api/search", params={"q": "Korean address"})

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_method"] == "bm25"
    assert body["hits"][0]["chunk_id"] == "doc-seoul-address:000"
    assert body["hits"][0]["source_path"].endswith("seoul_address_note.txt")

