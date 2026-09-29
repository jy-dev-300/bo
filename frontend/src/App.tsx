import { useEffect, useMemo, useState } from "react";

type Stage = "bm25" | "bge_dense" | "hybrid" | "rerank_bge" | "rerank_jev";
type ReviewStatus = "not-reviewed" | "pass" | "fail";

type SearchHit = {
  chunk_id: string;
  document_id: string;
  source_path: string;
  text: string;
  score: number;
  rank: number;
  page: number | null;
  section: string | null;
  matched_terms: string[];
};

type SearchResponse = { query: string; retrieval_method: string; hits: SearchHit[] };
type LocalCorpus = {
  corpus_id: string;
  files_received: number;
  files_indexed: number;
  chunks_indexed: number;
  skipped_files: string[];
};
type TestDefinition = { id: string; label: string; description: string; defaultQuery: string };
type TestSuite = {
  id: string;
  name: string;
  stage: Stage;
  description: string;
  output: string;
  scoreLabel: string;
  scoreFormula: string;
  scoreVariables: string[];
  scoreGuide: string[];
};
type RunResult = { loading: boolean; response?: SearchResponse; error?: string; elapsedMs?: number };
type HumanReview = { status: ReviewStatus; expectedFile: string; notes: string };
type IndexProgress = {
  processed: number;
  total: number;
  currentFile: string;
  status?: "indexing" | "pausing" | "paused" | "complete";
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const REVIEW_STORAGE_KEY = "bo-evaluation-reviews-v1";
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = new Set([
  ".pdf", ".docx", ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv",
  ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
  ".html", ".htm", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt",
  ".kts", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb",
  ".php", ".swift", ".sql", ".sh", ".ps1", ".css", ".scss", ".xml",
]);
const IGNORED_FOLDERS = new Set([
  ".git", ".cache", ".next", ".venv", "venv", "node_modules", "dist", "build",
  "__pycache__",
]);
const TESTS: TestDefinition[] = [
  { id: "literal", label: "Literal", description: "Checks whether an exact request finds the obvious matching file.", defaultQuery: "find the PDF file that contains my tax returns" },
  { id: "vague", label: "Somewhat vague", description: "Checks whether related wording can still find the intended notes.", defaultQuery: "find files that talk about startup ideas" },
  { id: "abstract", label: "Abstractly vague", description: "Checks whether a broad personal goal surfaces genuinely useful files.", defaultQuery: "find files that give useful information for improving my life" },
];
const SUITES: TestSuite[] = [
  { id: "bm25", name: "Task A + B - BM25 retrieval", stage: "bm25", description: "The exact-word baseline retrieves files without loading an AI model.", output: "BM25-retrieved files", scoreLabel: "Okapi BM25", scoreFormula: "sum IDF(q) * tf(q,d)(k1+1) / [tf(q,d) + k1(1-b+b*|d|/avgdl)]", scoreVariables: ["q: each distinct search term; d: the candidate chunk", "tf(q,d): number of times q occurs in d", "IDF(q) = ln(1 + (N-df(q)+0.5)/(df(q)+0.5)); N: chunk count; df(q): chunks containing q", "|d|: terms in d; avgdl: average chunk length; k1=1.5; b=0.75"], scoreGuide: ["Range: 0 or greater, with no fixed maximum. BO omits zero-score chunks because they contain none of the search terms.", "Higher means a stronger exact-word match for this search in this corpus. Rare matching terms help more than common ones.", "Do not compare BM25 values across different searches or corpora. There is no universal good-score threshold.", "Performance means whether relevant files rank near the top; measure that across labelled searches with Recall@k and MRR, not from one BM25 value."] },
  { id: "bge-dense", name: "Task A + B - BGE dense retrieval", stage: "bge_dense", description: "BGE-M3 retrieves files by comparing the meaning of the search request and chunks.", output: "BGE dense-retrieved files", scoreLabel: "Cosine similarity", scoreFormula: "cosine(q,d) = (q dot d) / (||q|| * ||d||); here BGE L2-normalizes both, so score = q dot d", scoreVariables: ["q: BGE-M3 dense vector for the search request", "d: BGE-M3 dense vector for the candidate chunk", "||q|| and ||d||: vector lengths, both normalized to 1"], scoreGuide: ["Range: -1 to 1. A value near 1 means the vectors point in a similar direction; 0 means little directional similarity; negative means opposing directions.", "Higher means BGE considers the meanings more similar, but it does not prove the chunk answers the request.", "Treat scores as relative within the same search. Real scores cluster differently by corpus and request, so there is no safe universal cutoff.", "Evaluate performance with labelled relevant files using Recall@k, MRR, or nDCG across many searches."] },
  { id: "retrieval", name: "Task A + B - BM25 + BGE retrieval", stage: "hybrid", description: "BM25 and BGE-M3 retrieve and combine candidate chunks.", output: "Retrieved files", scoreLabel: "Reciprocal Rank Fusion (RRF)", scoreFormula: "RRF(d) = sum_i 1 / (60 + rank_i(d))", scoreVariables: ["d: the candidate chunk", "i: each retrieval list containing d", "rank_i(d): d's one-based position in list i", "60: the smoothing constant; raw BM25 and BGE scores are not added together"], scoreGuide: ["Range: greater than 0 up to L/61, where L is the number of contributing lists. With five lists and one request, the theoretical maximum is about 0.082.", "Higher means the chunk appeared near the top of more retrieval lists. It is not a probability or relevance percentage.", "Compare RRF scores only when the same lists and smoothing constant are used.", "Good performance means relevant files consistently receive high fused ranks; verify that with Recall@k, MRR, or nDCG on labelled searches."] },
  { id: "bge-rerank", name: "Task C1 - BGE reranking", stage: "rerank_bge", description: "The same hybrid candidates are reordered by the BGE cross-encoder.", output: "BGE-ranked files", scoreLabel: "BGE normalized cross-encoder score", scoreFormula: "score = sigmoid(z) = 1 / (1 + exp(-z))", scoreVariables: ["z: BGE cross-encoder relevance logit for the request-chunk pair", "sigmoid(z): converts the logit to a 0-to-1 normalized relevance score"], scoreGuide: ["Range: 0 to 1. Above 0.5 means the underlying logit is positive; below 0.5 means it is negative.", "Higher means the reranker sees stronger request-chunk relevance after reading them together.", "This sigmoid value is not guaranteed to be a calibrated probability, so 0.8 does not automatically mean an 80% chance of relevance.", "Judge improvement by comparing ranking metrics before and after reranking on the same labelled searches."] },
  { id: "jev-rerank", name: "Task C1 - Jev reranking", stage: "rerank_jev", description: "The same hybrid candidates are reordered by TypeSafe Jev.", output: "Jev-ranked files", scoreLabel: "Jev Noul relevance probability", scoreFormula: "score = P(relevant = true | search request, candidate chunk)", scoreVariables: ["relevant=true: Jev judges that the chunk directly helps answer the request", "search request: the user's test request", "candidate chunk: one passage returned by hybrid retrieval"], scoreGuide: ["Range: 0 to 1. Near 1 means strong support for relevant=true; near 0 means strong support for false; near 0.5 means uncertainty.", "The value measures Jev's judgement of one candidate, not whether retrieval found every relevant file.", "Check calibration on your own labelled files before treating a number such as 0.8 as a reliable 80% success rate.", "Compare Jev with BGE using the same candidate sets and labelled searches, then measure MRR or nDCG plus latency and cost."] },
];
const EMPTY_REVIEW: HumanReview = { status: "not-reviewed", expectedFile: "", notes: "" };
const RAG_CHECKS = ["Required evidence retrieved", "Answer used selected evidence", "Claims supported by cited chunks", "Citation locations correct"];
const indexingControl: {
  pauseRequested: boolean;
  skipRequested: boolean;
  request?: AbortController;
  corpusId?: string;
  currentPath?: string;
} = { pauseRequested: false, skipRequested: false };

function testKey(suiteId: string, testId: string) {
  return `${suiteId}:${testId}`;
}

function loadReviews(): Record<string, HumanReview> {
  try {
    return JSON.parse(localStorage.getItem(REVIEW_STORAGE_KEY) ?? "{}") as Record<string, HumanReview>;
  } catch {
    return {};
  }
}

async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const pieces: string[] = [];
  for (let start = 0; start < bytes.length; start += 0x8000) {
    pieces.push(String.fromCharCode(...bytes.subarray(start, start + 0x8000)));
  }
  return btoa(pieces.join(""));
}

function selectedPath(file: File): string {
  return file.webkitRelativePath || file.name;
}

function fileSkipReason(file: File): string | undefined {
  const path = selectedPath(file);
  const pathParts = path.replaceAll("\\", "/").split("/");
  const ignoredFolder = pathParts.slice(0, -1).find((part) => IGNORED_FOLDERS.has(part));
  if (ignoredFolder) return `ignored folder: ${ignoredFolder}`;
  if (path.length > 1000) return "path is longer than the 1,000-character limit";
  if (file.size === 0) return "empty file";
  if (file.size > MAX_FILE_BYTES) return "larger than the 20 MB per-file limit";
  const lastDot = file.name.lastIndexOf(".");
  const extension = lastDot >= 0 ? file.name.slice(lastDot).toLowerCase() : "";
  if (!SUPPORTED_EXTENSIONS.has(extension) && !file.type.startsWith("text/")) {
    return `unsupported file type: ${extension || file.type || "unknown"}`;
  }
  return undefined;
}

function describeApiError(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const record = item as { loc?: unknown; msg?: unknown };
      const location = Array.isArray(record.loc) ? record.loc.join(" > ") : "request";
      return `${location}: ${String(record.msg ?? "invalid value")}`;
    });
    return messages.join("; ");
  }
  return detail ? JSON.stringify(detail) : fallback;
}

async function readResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

/** Render the local BO retrieval and reranking test dashboard. */
export default function App() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [corpus, setCorpus] = useState<LocalCorpus>();
  const [corpusError, setCorpusError] = useState<string>();
  const [indexing, setIndexing] = useState(false);
  const [indexProgress, setIndexProgress] = useState<IndexProgress>();
  const [queries, setQueries] = useState<Record<string, string>>(() => Object.fromEntries(SUITES.flatMap((suite) => TESTS.map((test) => [testKey(suite.id, test.id), test.defaultQuery]))));
  const [runs, setRuns] = useState<Record<string, RunResult>>({});
  const [reviews, setReviews] = useState<Record<string, HumanReview>>(loadReviews);

  useEffect(() => localStorage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(reviews)), [reviews]);
  const reviewedCount = useMemo(() => Object.values(reviews).filter((review) => review.status !== "not-reviewed").length, [reviews]);
  const completedCount = Object.values(runs).filter((run) => run.response).length;
  const selectedBytes = selectedFiles.reduce((total, file) => total + file.size, 0);
  const fileSelection = useMemo(() => {
    const indexable: File[] = [];
    const skipped: string[] = [];
    for (const file of selectedFiles) {
      const reason = fileSkipReason(file);
      if (reason) skipped.push(`${selectedPath(file)}: ${reason}`);
      else indexable.push(file);
    }
    return { indexable, skipped };
  }, [selectedFiles]);

  function selectFiles(files: FileList | null) {
    setSelectedFiles(files ? Array.from(files) : []);
    setCorpus(undefined);
    setCorpusError(undefined);
    setIndexProgress(undefined);
    setRuns({});
  }

  async function indexSelectedFiles() {
    if (!selectedFiles.length) return;
    if (!fileSelection.indexable.length) {
      setCorpusError("None of the selected files are supported, non-empty files under 20 MB.");
      return;
    }
    setIndexing(true);
    setCorpusError(undefined);
    indexingControl.pauseRequested = false;
    indexingControl.skipRequested = false;
    const existingProgress = indexProgress?.total === fileSelection.indexable.length ? indexProgress : undefined;
    const startIndex = corpus && existingProgress?.status === "paused" ? existingProgress.processed : 0;
    if (startIndex === 0) setCorpus(undefined);
    setIndexProgress({ processed: startIndex, total: fileSelection.indexable.length, currentFile: "", status: "indexing" });
    let latestCorpus: LocalCorpus | undefined = startIndex > 0 ? corpus : undefined;
    const manuallySkipped: string[] = [];
    try {
      let activeCorpusId: string | undefined = latestCorpus?.corpus_id;
      indexingControl.corpusId = activeCorpusId;
      for (let fileIndex = startIndex; fileIndex < fileSelection.indexable.length; fileIndex += 1) {
        const file = fileSelection.indexable[fileIndex];
        const relativePath = selectedPath(file);
        indexingControl.skipRequested = false;
        indexingControl.currentPath = relativePath;
        setIndexProgress({ processed: fileIndex, total: fileSelection.indexable.length, currentFile: relativePath, status: "indexing" });
        try {
          const contentBase64 = await fileToBase64(file);
          if (indexingControl.skipRequested) {
            manuallySkipped.push(`${relativePath}: skipped by user`);
          } else {
            indexingControl.request = new AbortController();
            const response = await fetch(`${API_URL}/api/testing/corpus`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              signal: indexingControl.request.signal,
              body: JSON.stringify({
                corpus_id: activeCorpusId,
                files: [{ relative_path: relativePath, media_type: file.type || null, content_base64: contentBase64 }],
                source_paths: fileIndex === 0 && startIndex === 0
                  ? fileSelection.indexable.map(selectedPath)
                  : undefined,
              }),
            });
            const body = await readResponseBody(response);
            if (!response.ok) throw new Error(describeApiError(body, `Indexing failed (${response.status})`));
            const serverCorpus = body as LocalCorpus;
            latestCorpus = {
              ...serverCorpus,
              skipped_files: Array.from(new Set([...fileSelection.skipped, ...manuallySkipped, ...serverCorpus.skipped_files])),
            };
            activeCorpusId = latestCorpus.corpus_id;
            indexingControl.corpusId = activeCorpusId;
            setCorpus(latestCorpus);
          }
        } catch (caught) {
          if (indexingControl.skipRequested) {
            manuallySkipped.push(`${relativePath}: skipped by user`);
          } else {
            throw caught;
          }
        } finally {
          indexingControl.request = undefined;
          indexingControl.currentPath = undefined;
        }
        const processed = fileIndex + 1;
        setIndexProgress({ processed, total: fileSelection.indexable.length, currentFile: relativePath, status: indexingControl.pauseRequested ? "paused" : "indexing" });
        if (indexingControl.pauseRequested) break;
      }
      if (!latestCorpus?.files_indexed && !indexingControl.pauseRequested) {
        throw new Error("None of the selected files contained supported searchable text.");
      }
      setIndexProgress((current) => current ? {
        ...current,
        currentFile: "",
        status: indexingControl.pauseRequested ? "paused" : "complete",
      } : current);
      setRuns({});
    } catch (caught) {
      setCorpus(latestCorpus);
      setCorpusError(caught instanceof Error ? caught.message : "The selected files could not be indexed.");
    } finally {
      setIndexing(false);
    }
  }

  function pauseIndexing() {
    indexingControl.pauseRequested = true;
    setIndexProgress((current) => current ? { ...current, status: "pausing" } : current);
  }

  async function skipCurrentFile() {
    indexingControl.skipRequested = true;
    const request = indexingControl.request;
    const corpusId = indexingControl.corpusId;
    const relativePath = indexingControl.currentPath;
    try {
      if (corpusId && relativePath) {
        await fetch(`${API_URL}/api/testing/corpus/${corpusId}/skip`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ relative_path: relativePath }),
        });
      }
    } finally {
      request?.abort();
    }
  }

  async function runTest(suite: TestSuite, test: TestDefinition) {
    if (selectedFiles.length && !corpus?.files_indexed) {
      setCorpusError("Index the selected files before running a test.");
      return;
    }
    const key = testKey(suite.id, test.id);
    const searchRequest = queries[key]?.trim();
    if (!searchRequest) return;
    setRuns((current) => ({ ...current, [key]: { loading: true } }));
    const startedAt = performance.now();
    try {
      const params = new URLSearchParams({ q: searchRequest, limit: "10", stage: suite.stage });
      if (corpus) params.set("corpus_id", corpus.corpus_id);
      const response = await fetch(`${API_URL}/api/search?${params.toString()}`);
      const body = await readResponseBody(response);
      if (!response.ok) throw new Error(describeApiError(body, `Test failed (${response.status})`));
      setRuns((current) => ({ ...current, [key]: { loading: false, response: body as SearchResponse, elapsedMs: performance.now() - startedAt } }));
    } catch (caught) {
      setRuns((current) => ({ ...current, [key]: { loading: false, error: caught instanceof Error ? caught.message : "The test could not run.", elapsedMs: performance.now() - startedAt } }));
    }
  }

  async function runSuite(suite: TestSuite) {
    for (const test of TESTS) await runTest(suite, test);
  }

  function updateReview(key: string, update: Partial<HumanReview>) {
    setReviews((current) => ({ ...current, [key]: { ...(current[key] ?? EMPTY_REVIEW), ...update } }));
  }

  return (
    <main>
      <header className="tool-header">
        <h1>BO testing</h1>
        <div className="progress-line"><span>{completedCount} runs</span><span>{reviewedCount} reviews</span></div>
      </header>

      <section className="local-corpus" aria-labelledby="local-corpus-title">
        <div><h2 id="local-corpus-title">Local test corpus</h2><p>Select files or a folder, then index them before running tests.</p></div>
        <div className="file-actions">
          <label className="file-button">Choose files<input type="file" multiple onChange={(event) => selectFiles(event.target.files)} /></label>
          <label className="file-button">Choose folder<input type="file" multiple ref={(node) => node?.setAttribute("webkitdirectory", "")} onChange={(event) => selectFiles(event.target.files)} /></label>
          {!indexing ? <button className="index-button" disabled={!selectedFiles.length} onClick={() => void indexSelectedFiles()}>{indexProgress?.status === "paused" ? "Resume indexing" : "Index selected files"}</button> : null}
          {indexing ? <button className="file-button" disabled={indexProgress?.status === "pausing"} onClick={pauseIndexing}>{indexProgress?.status === "pausing" ? "Finishing current file..." : "Pause & run tests"}</button> : null}
          {indexing ? <button className="file-button" onClick={() => void skipCurrentFile()}>Skip current file</button> : null}
        </div>
        <div className="corpus-status" aria-live="polite">
          {!selectedFiles.length && !corpus ? <span>Using the bundled sample corpus.</span> : null}
          {selectedFiles.length && !corpus ? <span>{selectedFiles.length} selected ({(selectedBytes / 1024 / 1024).toFixed(1)} MB): {fileSelection.indexable.length} indexable, {fileSelection.skipped.length} skipped before reading.</span> : null}
          {indexing && indexProgress ? <div className="index-progress"><div><strong>{indexProgress.status === "pausing" ? "Pausing after this file" : `Indexing ${indexProgress.processed} of ${indexProgress.total}`}</strong><span>{indexProgress.total - indexProgress.processed} files left</span></div><progress value={indexProgress.processed} max={indexProgress.total} /><small>{indexProgress.status === "pausing" ? "Tests will unlock when the current file finishes." : indexProgress.currentFile || "Preparing files..."}</small></div> : null}
          {!indexing && indexProgress?.status === "paused" ? <div className="paused-status"><strong>Indexing paused — tests are ready.</strong><span>{indexProgress.total - indexProgress.processed} files left. Run tests now, then choose Resume indexing when you are ready.</span></div> : null}
          {corpus ? <span><strong>{corpus.files_indexed} files</strong> indexed into <strong>{corpus.chunks_indexed} chunks</strong>.</span> : null}
          {corpusError ? <span className="run-error compact-error">{corpusError}</span> : null}
          {(corpus?.skipped_files.length || (!corpus && fileSelection.skipped.length)) ? <details><summary>{corpus?.skipped_files.length ?? fileSelection.skipped.length} files skipped</summary><ul>{(corpus?.skipped_files ?? fileSelection.skipped).map((item, index) => <li key={`${index}:${item}`}>{item}</li>)}</ul></details> : null}
        </div>
        <p className="privacy-note">Files are read only after you select them and are kept in this local app's memory. Running Jev sends candidate text to TypeSafe for reranking.</p>
      </section>

      <details className="rag-status">
        <summary><span><span className="section-kicker">RAG evaluation</span><strong>RAG checks</strong></span><span>4 checks not run</span></summary>
        <div className="rag-status-body"><p>These remain not run until BO returns a generated answer with citations.</p><ul>{RAG_CHECKS.map((check) => <li key={check}><span>{check}</span><strong>Not run</strong></li>)}</ul></div>
      </details>

      <section className="testing-section" aria-labelledby="testing-title">
        <div className="testing-heading"><div><p className="section-kicker">Retrieval testing</p><h2 id="testing-title">Tests</h2></div><span>{corpus ? `${corpus.files_indexed} indexed files` : "Sample corpus"}</span></div>
        <div className="suite-tabs" role="tablist" aria-label="Retrieval test type">
          {SUITES.map((suite) => <button key={suite.id} role="tab" aria-selected={(queries.__activeSuite ?? SUITES[0].id) === suite.id} className={(queries.__activeSuite ?? SUITES[0].id) === suite.id ? "active" : ""} onClick={() => setQueries((current) => ({ ...current, __activeSuite: suite.id }))}>{suite.name.replace(/^Task [A-Z0-9 +]+ - /, "")}</button>)}
        </div>
        <div className="suite-list">
        {SUITES.filter((suite) => suite.id === (queries.__activeSuite ?? SUITES[0].id)).map((suite) => (
          <section className="suite" key={suite.id} aria-labelledby={`${suite.id}-title`}>
            <div className="suite-header">
              <div className="suite-heading"><p className="stage-label">{suite.stage.replaceAll("_", " ")}</p><h2 id={`${suite.id}-title`}>{suite.name}</h2><p>{suite.description}</p></div>
              <button className="run-suite" disabled={selectedFiles.length > 0 && !corpus?.files_indexed} onClick={() => void runSuite(suite)}>Run all 3</button>
            </div>
            <div className="test-grid">
              {TESTS.map((test) => {
                const key = testKey(suite.id, test.id);
                const run = runs[key];
                const review = reviews[key] ?? EMPTY_REVIEW;
                return (
                  <article className="test-card" key={key}>
                    <div className="test-card-header"><div><span className="test-type">{test.label}</span><h3>{test.label} search request</h3><p>{test.description}</p></div><button className="run-test" disabled={run?.loading || (selectedFiles.length > 0 && !corpus?.files_indexed)} onClick={() => void runTest(suite, test)}>{run?.loading ? "Running..." : "Run test"}</button></div>
                    <label className="query-field">Test search request<textarea rows={2} value={queries[key]} onChange={(event) => setQueries((current) => ({ ...current, [key]: event.target.value }))} /></label>
                    <div className="output-panel" aria-live="polite">
                      <div className="output-heading"><span>{suite.output}</span>{run?.elapsedMs !== undefined ? <span>{Math.round(run.elapsedMs)} ms</span> : null}</div>
                      <details className="score-method"><summary><strong>{suite.scoreLabel}</strong><span>Formula, variables, and score guide</span></summary><code>{suite.scoreFormula}</code><h5>Variables</h5><ul>{suite.scoreVariables.map((variable) => <li key={variable}>{variable}</li>)}</ul><h5>How to read the score</h5><ul>{suite.scoreGuide.map((guidance) => <li key={guidance}>{guidance}</li>)}</ul></details>
                      {!run ? <p className="empty-state">Run this test to see BO's output.</p> : null}
                      {run?.loading ? <p className="empty-state">BO is searching...</p> : null}
                      {run?.error ? <p className="run-error">{run.error}</p> : null}
                      {run?.response && run.response.hits.length === 0 ? <p className="empty-state">No files matched this search request.</p> : null}
                      {run?.response?.hits.length ? <ol className="result-list">{run.response.hits.map((hit) => <li key={hit.chunk_id}><details className="ranked-file"><summary><span className="rank">{hit.rank}</span><span className="result-title"><strong>{hit.source_path}</strong><small>{[hit.section, hit.page ? `Page ${hit.page}` : null].filter(Boolean).join(" - ") || hit.document_id}</small></span><span className="score"><small>{suite.scoreLabel}</small>{hit.score.toFixed(3)}</span></summary><p>{hit.text}</p></details></li>)}</ol> : null}
                    </div>
                    <details className="review-panel">
                      <summary className="review-heading"><h4>Human review</h4><span>Saved in this browser</span></summary>
                      <div className="review-fields">
                      <div className="review-row">
                        <label>Verdict<select value={review.status} onChange={(event) => updateReview(key, { status: event.target.value as ReviewStatus })}><option value="not-reviewed">Not reviewed</option><option value="pass">Pass</option><option value="fail">Fail</option></select></label>
                        <label>Expected file or path<input value={review.expectedFile} placeholder="e.g. notes/startup-ideas.pdf" onChange={(event) => updateReview(key, { expectedFile: event.target.value })} /></label>
                      </div>
                      <label>Testing notes<textarea rows={3} value={review.notes} placeholder="What looked right, wrong, surprising, or missing?" onChange={(event) => updateReview(key, { notes: event.target.value })} /></label>
                      </div>
                    </details>
                  </article>
                );
              })}
            </div>
          </section>
        ))}
        </div>
      </section>
    </main>
  );
}
