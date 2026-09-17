# Interview Drills

Use these as whiteboard, live-coding, and incident-debugging prompts. Explain the prediction before changing code, then identify the metric that would falsify it.

1. Change chunk size from 300 to 900 tokens. Predict effects on recall, reranker quality, citation precision, latency, and context diversity.
2. Add file type and date-range metadata filters. Where should filtering happen for lexical and vector retrieval, and what happens if metadata is missing?
3. A vague-memory query retrieves nothing useful. Trace extraction, chunking, query terms, both retrievers, fusion, and filters to locate the failure.
4. Change retrieval top-k from 10 to 50 while the context budget stays fixed. What gets better, worse, or merely more expensive?
5. Implement query rewriting without losing exact filenames, addresses, or code identifiers. How will you evaluate regressions?
6. Two chunks contain the same paragraph because adjacent chunks overlap. Where and how should they be deduplicated?
7. Replace the embedding model. Which stored artifacts become stale, and how do you migrate without mixing vector spaces?
8. Explain BM25 versus vector search using “Python retries” and “that file where failures try again” as contrasting queries.
9. Reduce p95 RAG latency. Which stages can run concurrently, cache safely, or use smaller candidate sets—and what quality metrics guard each change?
10. Debug an answer whose citation opens the correct document but the wrong page. Which contracts and offsets do you inspect?
11. Vector recall is high but final answer quality drops after reranking. Design an isolation experiment.
12. The model answers confidently when no selected evidence supports the claim. Where should abstention be enforced and tested?

## Five oral-exam questions for the first checkpoint

1. Why are BM25 and cosine-similarity scores unsafe to add directly?
2. What metadata must a chunk retain for a citation to be inspectable?
3. How can increasing top-k reduce grounded-answer quality?
4. Why must retrieval success and answer faithfulness be evaluated separately?
5. When should OCR run, and how would low OCR confidence affect ranking or UI evidence?

