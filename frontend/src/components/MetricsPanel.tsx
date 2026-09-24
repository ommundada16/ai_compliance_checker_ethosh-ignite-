import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";

/**
 * The v1 vs v2 comparison, read from the committed evaluation results.
 *
 * Nothing here is computed in the browser. The dashboard shows exactly the
 * numbers the repository records, so a reader can check any figure against
 * eval_data/results/*.json. A UI that recomputes its own metrics is a UI that
 * can disagree with the evidence behind it.
 */

const RETRIEVAL_ROWS: Array<{ key: string; label: string; better: "up" }> = [
  { key: "scope_recall", label: "Scope recall", better: "up" },
  { key: "ndcg", label: "nDCG", better: "up" },
  { key: "mrr", label: "MRR", better: "up" },
  { key: "map", label: "MAP", better: "up" },
  { key: "context_precision", label: "Context precision", better: "up" },
];

function pct(from: number, to: number): string {
  if (!from) return to ? "new" : "—";
  const change = ((to - from) / from) * 100;
  return `${change >= 0 ? "+" : ""}${change.toFixed(0)}%`;
}

export default function MetricsPanel() {
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.metrics });

  if (metrics.isLoading) return <div className="empty">Loading results…</div>;
  if (metrics.isError) return <div className="empty">Could not load results</div>;

  const data = metrics.data ?? {};
  const v1 = data.v1_baseline;
  const ablation = data.v2_ablation?.results ?? {};
  const audit = data.audit_eval ?? {};

  const systems = ["v1_baseline", ...Object.keys(ablation)];
  const summaryFor = (name: string) =>
    name === "v1_baseline" ? v1?.summary?.["5"] : ablation[name]?.summary?.["5"];

  const truncation = v1?.encoder_truncation;

  return (
    <div>
      <div className="card">
        <h2>Retrieval quality at k = 5</h2>
        <p className="dim" style={{ marginTop: 0, fontSize: 13 }}>
          Identical frozen gold set, identical metric code, same 75 passages.
          Each v2 row adds exactly one thing to the row above it.
        </p>
        <table className="metrics">
          <thead>
            <tr>
              <th>Metric</th>
              {systems.map((s) => (
                <th key={s}>{s.replace("_", " ")}</th>
              ))}
              <th>v1 → best</th>
            </tr>
          </thead>
          <tbody>
            {RETRIEVAL_ROWS.map((row) => {
              const values = systems.map((s) => summaryFor(s)?.[row.key] ?? 0);
              const best = Math.max(...values);
              return (
                <tr key={row.key}>
                  <td>{row.label}</td>
                  {values.map((value, index) => (
                    <td key={index} className={value === best && best > 0 ? "best" : ""}>
                      {value.toFixed(3)}
                    </td>
                  ))}
                  <td className="delta">{pct(values[0], best)}</td>
                </tr>
              );
            })}
            <tr>
              <td>Context words sent to the LLM</td>
              {systems.map((s) => (
                <td key={s}>{(summaryFor(s)?.mean_context_words ?? 0).toFixed(0)}</td>
              ))}
              <td className="delta">
                {pct(
                  summaryFor("v1_baseline")?.mean_context_words ?? 0,
                  Math.min(
                    ...systems.map((s) => summaryFor(s)?.mean_context_words ?? Infinity),
                  ),
                )}
              </td>
            </tr>
          </tbody>
        </table>
        <p className="dim" style={{ fontSize: 12, marginBottom: 0 }}>
          Scope recall is the headline: did the retriever surface the provisions
          that actually govern the passage. Lower context words is better — it
          means less noise reaching the model, for fewer tokens.
        </p>
      </div>

      {truncation && (
        <div className="card">
          <h2>Why v1 could not have worked</h2>
          <p style={{ marginTop: 0 }}>
            v1 embedded 800-word chunks with a model that accepts 256
            word-pieces. Measured, not assumed: only{" "}
            <strong>{truncation.embedded_words_per_chunk}</strong> of every{" "}
            <strong>{truncation.chunk_words}</strong> words were ever encoded.
          </p>
          <table className="metrics">
            <tbody>
              <tr>
                <td>Fraction of each chunk embedded</td>
                <td>{(truncation.fraction_of_chunk_embedded * 100).toFixed(0)}%</td>
              </tr>
              <tr>
                <td>Clauses ever embedded</td>
                <td>
                  {truncation.clauses_ever_embedded} / {truncation.clauses_total}
                </td>
              </tr>
              <tr>
                <td>Hard recall ceiling</td>
                <td className="mono">
                  {(truncation.recall_ceiling * 100).toFixed(1)}%
                </td>
              </tr>
            </tbody>
          </table>
          <p className="dim" style={{ fontSize: 12, marginBottom: 0 }}>
            No value of k could beat that ceiling: the remaining clauses were
            never in the index in any form. v2 feeds clause-sized units to a
            model with a 430-word window and truncates zero gold clauses.
          </p>
        </div>
      )}

      {Object.keys(audit).length > 0 && (
        <div className="card">
          <h2>Audit quality</h2>
          <p className="dim" style={{ marginTop: 0, fontSize: 13 }}>
            Recall counts violations known to be present, so it is a lower
            bound. The false-positive rate is the share of clean passages that
            drew at least one finding — the metric that decides whether a
            reviewer can trust the output.
          </p>
          <table className="metrics">
            <thead>
              <tr>
                <th>System</th>
                <th>Findings</th>
                <th>Recall</th>
                <th>F1</th>
                <th>FP rate</th>
                <th>Hallucination</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(audit).map(([name, value]: [string, any]) => (
                <tr key={name}>
                  <td>{name.replace("_", " ")}</td>
                  <td>{value.findings}</td>
                  <td>{value.scores.recall.toFixed(3)}</td>
                  <td>{value.scores.f1.toFixed(3)}</td>
                  <td>{value.scores.false_positive_rate.toFixed(3)}</td>
                  <td>{value.scores.hallucination_rate.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
