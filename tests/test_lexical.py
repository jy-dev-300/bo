from retrieval.lexical import BM25Index, tokenize


def test_tokenizer_preserves_identifiers_and_korean_text() -> None:
    assert tokenize("fetch_with_retries 서울특별시") == ["fetch_with_retries", "서울특별시"]


def test_bm25_returns_exact_visual_memory_match(sample_chunks) -> None:
    results = BM25Index(sample_chunks).search("orange chart March", limit=2)

    assert results[0].chunk.id == "doc-march-report:000"
    assert results[0].rank == 1
    assert set(results[0].details["matched_terms"]) == {"orange", "chart", "march"}


def test_bm25_returns_no_results_for_unseen_terms(sample_chunks) -> None:
    assert BM25Index(sample_chunks).search("volcano astronomy") == []

