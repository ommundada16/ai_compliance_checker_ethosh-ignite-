import { useState } from "react";
import { api, type SearchHit } from "../lib/api";

/**
 * Retrieval without auditing.
 *
 * Most RAG debugging is really "what did it actually retrieve", and answering
 * that should not require reading a log. The rerank toggle makes the
 * two-stage architecture visible: the same query, with and without the
 * cross-encoder, usually returns a noticeably different top 5.
 */
export default function SearchPanel() {
  const [query, setQuery] = useState(
    "no clinical investigation was performed for this implantable device",
  );
  const [rerank, setRerank] = useState(true);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    const started = performance.now();
    try {
      const response = await api.search(query, 8, rerank);
      setHits(response.results);
      setElapsed(performance.now() - started);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <form className="card" onSubmit={run}>
        <h2>Inspect retrieval</h2>
        <p className="dim" style={{ marginTop: 0, fontSize: 13 }}>
          Paste any text from a clinical evaluation report and see exactly which
          MDR clauses the auditor would be shown.
        </p>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Text from a CER section…"
        />
        <div className="row" style={{ marginTop: 10 }}>
          <label className="row" style={{ gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={rerank}
              onChange={(event) => setRerank(event.target.checked)}
            />
            <span>Cross-encoder rerank</span>
          </label>
          <button className="primary" type="submit" disabled={busy}>
            {busy ? "Searching…" : "Search"}
          </button>
          {elapsed !== null && !busy && (
            <span className="dim mono">{(elapsed / 1000).toFixed(2)}s</span>
          )}
        </div>
      </form>

      {error && (
        <div className="card" style={{ borderColor: "var(--critical)" }}>
          <strong style={{ color: "var(--critical)" }}>Search failed.</strong>{" "}
          <span className="mono">{error}</span>
        </div>
      )}

      {hits?.length === 0 && <div className="empty">No clauses matched.</div>}

      {hits?.map((hit, index) => (
        <div key={hit.clause_id} className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div className="row">
              <span className="dim mono">#{index + 1}</span>
              <span className="clause-ref">{hit.clause_id}</span>
              <span className="dim" style={{ fontSize: 12 }}>{hit.path}</span>
            </div>
            <span className="dim mono">
              {/* Whichever stage decided the order reports the score, so this
                  number always agrees with the ranking it is shown beside. */}
              {rerank ? "rerank" : "RRF"} {hit.score.toFixed(4)}
            </span>
          </div>
          <p style={{ marginBottom: 0, fontSize: 13 }}>
            {hit.text.length > 420 ? `${hit.text.slice(0, 420)}…` : hit.text}
          </p>
        </div>
      ))}
    </div>
  );
}
