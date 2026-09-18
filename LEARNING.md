# Learning Log

These notes follow tasks A-I in `STUDENT_TASKS.md`. Each note says what an idea does and why we use it.

## A. Document-to-chunk transformation

- **A chunk is a small piece of a document that search can return.** `chunk_document` takes one document whose text has already been extracted and returns its chunks in order. A chunk should make sense on its own, but it does not need the whole document.
- **We use Chonkie to split ordinary text.** Its `SentenceChunker` tries to keep sentences whole and packs them into chunks that fit the size limit. It can repeat a little text between neighboring chunks. This is our default because it is predictable and does not need an embedding model. It replaced our handwritten splitter, which counted tokens and overlap incorrectly.
- **Chonkie's `SemanticChunker` can split by topic.** It uses a local model (`minishlab/potion-base-32M` by default) to find places where the subject changes. It needs token mode, and its first use may download model files. Use it when sentence boundaries give poor chunks.
- **Chonkie's `TokenChunker` handles text that is too large.** We use it when one sentence or piece of code exceeds the limit. We also use it when we cannot find useful code boundaries.
- **We use Tree-sitter, through `tree-sitter-language-pack`, for code.** It finds structures such as functions and classes so related code can stay together. If a structure is too large, `TokenChunker` splits it further. If Tree-sitter cannot detect or process the language, we use `TokenChunker` as a fallback.
- **Pydantic checks the input rules and data.** For example, it rejects an overlap as large as the whole chunk. Gather the required fields before creating a Pydantic model because it checks them immediately.
- **Our code keeps track of each chunk's source.** After Chonkie or Tree-sitter chooses a boundary, we map the text back to its original page, section, line, or character position. We also make a chunk ID that stays the same when the document, text, position, and rules stay the same. The `ordinal` number gives the chunk's order; putting that number in the ID is only a convenience.
- **We do not use Docling's `HybridChunker` here.** That tool expects a `DoclingDocument`, but this function receives text blocks that have already been prepared. Converting them again would repeat parsing and could lose source locations. Late chunking is a possible future embedding technique; it is not how this function chooses boundaries.
- **A size limit stops chunks from growing forever.** Character mode counts written characters. Token mode uses the tokenizer named in the policy: `word` works offline, while a Hugging Face tokenizer can match a later model. A token may be a word, part of a word, punctuation, or space. The BM25 search tokenizer finds terms; it cannot give the exact token count for an embedding or answer model.
- **Five hundred tokens is roughly 350-400 ordinary English words.** This is only a rough guide. Code and other languages can have very different token counts.
- **Overlap repeats text at the edge of neighboring chunks.** This helps when an idea crosses a boundary. Count the repeated text inside each chunk's size limit. Do not accidentally add a full overlap to both ends of a middle chunk.
- **A long sentence can be split into fixed-size pieces.** Move forward by `target_size - overlap` each time so neighboring pieces share the requested amount. In Python, a slice from `start` to `start + target_size` never returns more than `target_size` items, and the last slice can be shorter. There is no need to keep cutting the text in half.
- **Sentence splitting can get punctuation wrong.** If we split after `.`, `?`, or `!`, split at the following space so the punctuation stays with the sentence. A wrong boundary may be harmless, but it can also move an important phrase to the edge of a chunk.
- **Never lose the source location while splitting text.** A plain string does not tell us which page or line it came from. Keep the text with its location so search results and citations can point back to the source.
- **An image needs text extraction first.** OCR or a vision model can turn a JPG into text blocks with source locations. Then the normal chunker can split that text. It cannot split image bytes as if they were words.
- **The caller owns the chunking settings.** Do not change the policy object inside the function without telling the caller; hidden changes make results hard to predict and test.

## B. Hybrid retrieval

Tools in one line: **BM25** finds exact words and names; **BGE-M3** finds similar meaning and detailed term matches; **metadata search** finds source clues such as paths; **RRF** combines those result lists fairly; the **BGE reranker** puts the strongest matches first; and optional **SPLADE++** gives us another English term-based search method to test when the others miss something.

These are practical starting picks, not proven winners over alternatives: we use the existing BM25 code instead of running a search server for this small corpus; BGE-M3 instead of three separate models because one local multilingual model produces dense, sparse, and token-level results; simple metadata matching instead of another model for file paths; RRF instead of adding incompatible scores or tuning weights without enough labeled examples; the BGE reranker because the same library provides a local multilingual cross-encoder; and SPLADE++ only as a separate sparse model to test against BGE-M3's sparse results.

- **BM25 is our starting point.** It searches for matching words, so it is useful for exact filenames, addresses, error messages, and code names. Our `BM25Index` runs locally without downloading a model.
- **BGE-M3 adds three ways to search.** Its dense method looks for similar meaning when the words differ. Its sparse method gives weight to useful terms. Its ColBERT-style method compares individual query and document token vectors, which can catch details lost when a whole chunk has only one vector. BGE-M3 is `BAAI/bge-m3` and runs through FlagEmbedding.
- **We keep BM25 alongside BGE-M3.** A meaning-based match may help with a paraphrase, while BM25 may be best for an exact code name. We also have a small search list for matches in metadata, such as a source path.
- **RRF combines the search lists.** Reciprocal Rank Fusion gives a chunk `1 / (k + rank)` points for each list where it appears, then adds those points. It uses positions because a BM25 score and a vector score are different kinds of numbers. A chunk appears once in the final list even if several methods found it. We keep each method's rank and score so we can inspect the result. RRF cannot rescue a relevant chunk that no method found.
- **The reranker checks a short list more carefully.** The BGE cross-encoder (`BAAI/bge-reranker-v2-m3`) reads the question together with each candidate and scores their match. We run it after search because it is more expensive. A later step reduces near-duplicate results. Reranking can move a found chunk higher; it cannot find a missing one.
- **SPLADE++ is an optional extra method for English.** We use `prithivida/Splade_PP_en_v2` through Sentence Transformers `SparseEncoder`. It makes its own sparse list, separate from BGE-M3's sparse list. It may repeat some of the same work, so keep it only if it helps enough to justify its cost.
- **Test one change at a time.** First test BM25 alone (`bm25`). Next add BGE-M3 (`hybrid`). Then add reranking (`rerank`). Finally add SPLADE++ (`specialists`). Ask the same questions each time and check whether the new step helped, hurt, or changed nothing. If BGE-M3 changes the results, test its dense, sparse, and ColBERT-style methods one at a time to find which caused the change. Our current comparison command tests the four full stages; it cannot test those three BGE-M3 methods separately yet.
- **Use the same measures for every test.** Run `python -m evaluation.compare_retrieval --stages bm25 hybrid rerank specialists --k 3`. Look at the chunk IDs each question returns. Recall@3 asks whether the needed chunks appear in the first three results. MRR@3 rewards putting a needed chunk closer to the top. Also note how long the first search takes, because it may load a model, and the typical time for later searches. Record results and failures in `RESULTS.md`.
- **Our current questions are too easy to choose a winner.** The three-question gold set gives BM25 Recall@3 = 1.000 and MRR@3 = 1.000. Add questions with paraphrases, exact identifiers, multiple languages, and misleading matches. The model-backed stages have not yet been measured for quality, speed, or memory use. The current code scores an in-memory collection; it is a small reference implementation.
- **The all-in-one system is only one option.** Compare `rerank` with `specialists` to see whether SPLADE++ adds value. Keep the simplest setup that finds and orders the right evidence without an unreasonable speed or memory cost. More methods do not automatically mean better search.

## C. Reranking and context selection

- **Search finds possible evidence; reranking puts the best candidates first.** A cross-encoder reads the question and one candidate together, so use it on a short list rather than every chunk.
- **Reranking cannot fix a missing candidate.** If search never found the needed chunk, changing the order will not help.
- **Choose varied evidence that fits the answer model's space limit.** Near-identical chunks waste space. Keep the source location with every selected chunk so the answer can cite it later.

## D. Grounded citation assembly

- **A citation needs to point to evidence actually used.** Keep each chunk's ID and source location through search, reranking, and context selection. Then check that a cited chunk really belongs to the selected evidence.

## E. Retrieval evaluation

- **First ask whether search found the needed chunk.** Recall@k measures that within the first `k` results. If search found it, MRR or nDCG can help judge how high it appeared.
- **Compare systems on the same questions.** Test BM25, hybrid search, and reranking with the same labeled answers. Record search quality, speed, and memory use before deciding whether another model is worth keeping.

## F. RAG evaluation

- No notes yet.

## G. Query processing

- **Keep the user's original question.** You can add a careful variation to help search, but do not remove an exact filename, path, address, or code name from the original.
- **Apply source filters before cutting down the candidate list when possible.** If you take only a few results first and filter afterward, you might throw away all useful matches.

## H. Index lifecycle and updates

- **Save document embeddings so repeated searches do not have to make them again.** An embedding is made from a chunk. Rebuild it when the chunk text or embedding model changes.
- **Load large models only when needed.** This makes normal startup lighter, but the first search that uses a model may be slow or need to download files.
- **Our current code scores every chunk in memory.** That is useful for learning and small examples. A larger collection needs saved indexes for vector, sparse, and ColBERT-style search.

## I. End-to-end failure diagnosis

- **Find the first place a wrong result went wrong.** If search never returned the needed chunk, investigate search. If it found the chunk but placed it too low, investigate ranking. Fix that stage, then rerun the same question to check whether the change helped.

## Across tasks: working approach

- **Try the main reasoning step yourself first.** Keep normal syntax and type suggestions in the editor, but turn off AI-written code suggestions while you first work through a core task.
- **Separate routine code from the hard decision.** Imports, copying already-chosen fields into a `Chunk`, and saving results are routine. Deciding where chunks begin and end is the part to understand.
- **Practice one small idea at a time.** If Python syntax and system design feel overwhelming together, try the idea in a tiny runnable example before putting it into the full project.
