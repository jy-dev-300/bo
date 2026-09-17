import { FormEvent, useState } from "react";

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

type SearchResponse = {
  query: string;
  retrieval_method: string;
  hits: SearchHit[];
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("that PDF with the orange chart from March");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error(`Search failed (${response.status})`);
      setResult((await response.json()) as SearchResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">Local-first retrieval baseline</p>
        <h1>Find the thing you half remember.</h1>
        <p className="intro">
          Search pre-indexed evidence now. Grounded answers become available only after the
          student-owned retrieval and citation stages are complete.
        </p>
      </header>

      <form onSubmit={submit}>
        <label htmlFor="query">What do you remember?</label>
        <div className="search-row">
          <input
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="the screenshot with the Korean address"
          />
          <button disabled={loading || !query.trim()}>{loading ? "Searching…" : "Search"}</button>
        </div>
      </form>

      {error ? <p role="alert" className="error">{error}</p> : null}

      {result ? (
        <section aria-live="polite">
          <div className="result-summary">
            <h2>{result.hits.length} evidence matches</h2>
            <span>{result.retrieval_method.toUpperCase()}</span>
          </div>
          {result.hits.length === 0 ? (
            <p>No lexical match. A semantic retriever may help later.</p>
          ) : null}
          <ol className="results">
            {result.hits.map((hit) => (
              <li key={hit.chunk_id}>
                <div className="provenance">
                  <strong>{hit.source_path}</strong>
                  <span>{[hit.section ? `Section: ${hit.section}` : null, hit.page ? `Page ${hit.page}` : null].filter(Boolean).join(" · ")}</span>
                </div>
                <p>{hit.text}</p>
                <small>rank {hit.rank} · score {hit.score.toFixed(3)} · {hit.chunk_id}</small>
              </li>
            ))}
          </ol>
        </section>
      ) : null}
    </main>
  );
}
