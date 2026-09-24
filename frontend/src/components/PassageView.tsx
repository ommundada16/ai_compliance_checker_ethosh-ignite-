import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Finding, type PassageAudit } from "../lib/api";

/**
 * Render the passage with each finding's evidence highlighted in place.
 *
 * The highlight uses the character offsets the guardrail resolved server-side,
 * not a text search in the browser. Searching again here would re-solve a
 * problem that was already solved exactly, and would silently fail on the
 * whitespace-normalised and fuzzy matches -- which are precisely the cases
 * where a reviewer most wants to see what the model was actually looking at.
 */
function HighlightedText({
  text,
  findings,
  activeId,
  onSelect,
}: {
  text: string;
  findings: Finding[];
  activeId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const segments = useMemo(() => {
    const spans = findings
      .map((f, i) => ({ start: f.quote_start, end: f.quote_end, id: `${f.clause_id}-${i}` }))
      .filter((s) => s.end > s.start)
      .sort((a, b) => a.start - b.start);

    // Overlapping spans would produce nested <mark> elements and duplicated
    // text. Keep the first and skip anything that starts before it ends.
    const kept: typeof spans = [];
    let cursor = 0;
    for (const span of spans) {
      if (span.start >= cursor) {
        kept.push(span);
        cursor = span.end;
      }
    }

    const out: Array<{ text: string; id?: string }> = [];
    let position = 0;
    for (const span of kept) {
      if (span.start > position) out.push({ text: text.slice(position, span.start) });
      out.push({ text: text.slice(span.start, span.end), id: span.id });
      position = span.end;
    }
    if (position < text.length) out.push({ text: text.slice(position) });
    return out;
  }, [text, findings]);

  return (
    <div className="passage-text">
      {segments.map((segment, index) =>
        segment.id ? (
          <mark
            key={index}
            className={`evidence${activeId === segment.id ? " active" : ""}`}
            onClick={() => onSelect(activeId === segment.id ? null : segment.id!)}
            title="Evidence for a finding"
          >
            {segment.text}
          </mark>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </div>
  );
}

function FindingCard({ finding, id, active, onSelect }: {
  finding: Finding; id: string; active: boolean; onSelect: () => void;
}) {
  return (
    <div
      className={`card finding ${finding.severity}`}
      style={active ? { outline: "2px solid var(--accent)" } : undefined}
      onClick={onSelect}
    >
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className={`pill ${finding.severity}`}>{finding.severity}</span>
        <span className="dim mono">
          confidence {finding.confidence.toFixed(2)}
          {finding.judged && " · judged"}
        </span>
      </div>

      <h2 style={{ marginTop: 8 }}>{finding.violating_statement}</h2>

      <div className="row">
        <span className="clause-ref">{finding.clause_id}</span>
        <span className="dim" style={{ fontSize: 12 }}>{finding.clause_path}</span>
      </div>

      <blockquote className="quote">{finding.source_quote}</blockquote>

      <h3>Why this breaches the clause</h3>
      <p style={{ margin: 0 }}>{finding.explanation}</p>

      <h3>Suggested correction</h3>
      <p style={{ margin: 0 }}>{finding.suggested_correction}</p>

      <div className="dim mono" style={{ marginTop: 10 }}>
        {finding.category} · page {finding.page} · grounding{" "}
        {finding.grounding_score.toFixed(2)}
      </div>
    </div>
  );
}

export default function PassageView({ passageId }: { passageId: string }) {
  const [audit, setAudit] = useState<PassageAudit | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);

  const passage = useQuery({
    queryKey: ["passage", passageId],
    queryFn: () => api.passage(passageId),
  });

  async function runAudit() {
    setRunning(true);
    setError(null);
    try {
      setAudit(await api.auditPassage(passageId));
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    } finally {
      setRunning(false);
    }
  }

  if (passage.isLoading) return <div className="empty">Loading…</div>;
  if (passage.isError) return <div className="empty">Could not load {passageId}</div>;

  const data = passage.data!;
  const findings = audit?.findings ?? [];

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 16 }}>
            <span className="mono dim">{data.section}</span> {data.section_title}
          </h2>
          <span className="dim" style={{ fontSize: 12 }}>
            pages {data.page_start}–{data.page_end} · {data.n_words} words
          </span>
        </div>
        <button className="primary" onClick={runAudit} disabled={running}>
          {running ? "Auditing…" : "Audit this section"}
        </button>
      </div>

      {error && (
        <div className="card" style={{ borderColor: "var(--critical)" }}>
          <strong style={{ color: "var(--critical)" }}>Audit failed.</strong>{" "}
          <span className="mono">{error}</span>
        </div>
      )}

      <HighlightedText
        text={data.text}
        findings={findings}
        activeId={activeId}
        onSelect={setActiveId}
      />

      {audit && (
        <>
          <h3 style={{ marginTop: 20 }}>
            {findings.length === 0
              ? "No findings — this section appears compliant"
              : `${findings.length} finding${findings.length === 1 ? "" : "s"}`}
          </h3>

          {findings.map((finding, index) => {
            const id = `${finding.clause_id}-${index}`;
            return (
              <FindingCard
                key={id}
                id={id}
                finding={finding}
                active={activeId === id}
                onSelect={() => setActiveId(activeId === id ? null : id)}
              />
            );
          })}

          {audit.dropped.length > 0 && (
            <div className="card">
              <h2>Rejected by guardrails ({audit.dropped.length})</h2>
              <p className="dim" style={{ marginTop: 0, fontSize: 13 }}>
                Findings the model produced that did not survive verification.
                Shown because the reasons are themselves a measurement of how
                the model fails.
              </p>
              {audit.dropped.map((dropped, index) => (
                <div key={index} className="row" style={{ marginBottom: 4 }}>
                  <span className="pill">{dropped.reason}</span>
                  <span className="dim mono">{dropped.detail}</span>
                </div>
              ))}
            </div>
          )}

          <div className="dim mono" style={{ marginTop: 8 }}>
            retrieved: {audit.retrieved_clauses.join(", ") || "nothing"} ·{" "}
            {audit.llm_model} · {audit.latency_s.toFixed(1)}s
          </div>
        </>
      )}
    </div>
  );
}
