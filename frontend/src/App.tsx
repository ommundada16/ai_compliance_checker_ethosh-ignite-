import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, streamAudit, type StreamProgress } from "./lib/api";
import PassageView from "./components/PassageView";
import MetricsPanel from "./components/MetricsPanel";
import SearchPanel from "./components/SearchPanel";

type Tab = "document" | "search" | "metrics";

function Health() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, retry: false });
  if (health.isLoading) return <span className="dim sub">checking…</span>;
  if (health.isError)
    return (
      <span className="sub">
        <span className="status-dot bad" /> API unreachable
      </span>
    );

  const data = health.data!;
  return (
    <span className="sub row" style={{ gap: 6 }}>
      <span className={`status-dot ${data.ok ? "ok" : "bad"}`} />
      {data.corpus.clauses} clauses · {data.corpus.passages} sections
      {!data.qdrant.ok && " · vector store down"}
    </span>
  );
}

/** Whole-document audit, streamed. Each passage arrives as it completes. */
function DocumentRun({ onSelect }: { onSelect: (id: string) => void }) {
  const [progress, setProgress] = useState<StreamProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [stop, setStop] = useState<(() => void) | null>(null);
  const [hits, setHits] = useState<Array<{ id: string; title: string; n: number }>>([]);
  const [error, setError] = useState<string | null>(null);

  function start() {
    setRunning(true);
    setHits([]);
    setError(null);
    const abort = streamAudit(
      {
        onPassage: (p) => {
          setProgress(p);
          if (p.result.findings.length > 0) {
            setHits((previous) => [
              ...previous,
              {
                id: p.result.passage_id,
                title: p.result.section,
                n: p.result.findings.length,
              },
            ]);
          }
        },
        onDone: () => setRunning(false),
        onError: (message) => {
          setError(message);
          setRunning(false);
        },
      },
      { limit: 0 },
    );
    // Keep the abort function so navigating away can close the stream; an
    // EventSource left open keeps the server working on results nobody reads.
    setStop(() => abort);
  }

  function halt() {
    stop?.();
    setRunning(false);
  }

  const percent = progress ? (progress.index / progress.total) * 100 : 0;

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Audit the whole document</h2>
        {running ? (
          <button className="ghost" onClick={halt}>Stop</button>
        ) : (
          <button className="primary" onClick={start}>Run</button>
        )}
      </div>

      {progress && (
        <>
          <div className="progress" style={{ marginTop: 10 }}>
            <div style={{ width: `${percent}%` }} />
          </div>
          <div className="dim mono" style={{ marginTop: 6 }}>
            {progress.index} / {progress.total} sections ·{" "}
            {progress.findings_total} findings
          </div>
        </>
      )}

      {error && (
        <p style={{ color: "var(--critical)", marginBottom: 0 }} className="mono">
          {error}
        </p>
      )}

      {hits.length > 0 && (
        <div style={{ marginTop: 10 }}>
          {hits.map((hit) => (
            <button
              key={hit.id}
              className="section-item"
              onClick={() => onSelect(hit.id)}
            >
              <span className="num">{hit.title}</span>
              <span className="title">{hit.id}</span>
              <span className="badge">{hit.n}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("document");
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const passages = useQuery({ queryKey: ["passages"], queryFn: api.passages });
  const rows = (passages.data ?? []).filter((p) => {
    const needle = filter.toLowerCase();
    return (
      !needle ||
      p.section_title.toLowerCase().includes(needle) ||
      p.section.includes(needle)
    );
  });

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>The Auditor</h1>
          <Health />
        </div>
        <nav className="tabs" role="tablist">
          {(["document", "search", "metrics"] as Tab[]).map((name) => (
            <button
              key={name}
              role="tab"
              className="tab"
              aria-selected={tab === name}
              onClick={() => setTab(name)}
            >
              {name === "document" ? "Document" : name === "search" ? "Retrieval" : "Results"}
            </button>
          ))}
        </nav>
      </header>

      <div className="main">
        {tab === "document" && (
          <aside className="sidebar">
            <div style={{ padding: 10 }}>
              <input
                type="search"
                placeholder="Filter sections…"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              />
            </div>
            {passages.isLoading && <div className="empty">Loading…</div>}
            {rows.map((row) => (
              <button
                key={row.passage_id}
                className="section-item"
                aria-current={selected === row.passage_id}
                onClick={() => setSelected(row.passage_id)}
              >
                <span className="num">{row.section}</span>
                <span className="title">{row.section_title}</span>
              </button>
            ))}
          </aside>
        )}

        <main className="content">
          {tab === "document" &&
            (selected ? (
              <PassageView passageId={selected} />
            ) : (
              <>
                <DocumentRun onSelect={setSelected} />
                <div className="empty">
                  Select a section on the left to audit it on its own, or run
                  the whole document above.
                </div>
              </>
            ))}
          {tab === "search" && <SearchPanel />}
          {tab === "metrics" && <MetricsPanel />}
        </main>
      </div>
    </div>
  );
}
