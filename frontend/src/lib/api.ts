/**
 * Typed client for the auditor API.
 *
 * Types are written by hand against the FastAPI schema rather than generated.
 * The surface is small, and a hand-written type that disagrees with the server
 * is caught the first time a field is read, whereas a generated one silently
 * drifts whenever the generator is not re-run.
 */

export type Severity = "Low" | "Medium" | "High" | "Critical";

export interface PassageSummary {
  passage_id: string;
  section: string;
  section_title: string;
  page_start: number;
  page_end: number;
  n_words: number;
}

export interface Passage extends PassageSummary {
  text: string;
  text_sha1: string;
}

export interface Clause {
  clause_id: string;
  path: string;
  text: string;
  page_start: number;
  n_words: number;
  article: number | null;
  annex: string | null;
}

export interface SearchHit {
  clause_id: string;
  score: number;
  path: string;
  text: string;
  page_start: number;
  n_words: number;
}

export interface Finding {
  passage_id: string;
  clause_id: string;
  clause_path: string;
  category: string;
  severity: Severity;
  violating_statement: string;
  source_quote: string;
  explanation: string;
  suggested_correction: string;
  confidence: number;
  page: number;
  /** Character offsets into the passage text, so the quote can be highlighted
   *  exactly rather than searched for again in the browser. */
  quote_start: number;
  quote_end: number;
  grounding_score: number;
  judged: boolean;
}

export interface DroppedFinding {
  reason: string;
  detail: string;
  raw: Record<string, unknown>;
}

export interface PassageAudit {
  passage_id: string;
  section: string;
  page: number;
  findings: Finding[];
  dropped: DroppedFinding[];
  retrieved_clauses: string[];
  llm_provider: string;
  llm_model: string;
  latency_s: number;
  error: string | null;
}

export interface Health {
  ok: boolean;
  api: string;
  qdrant: { ok: boolean; points?: number; error?: string };
  provider: { ok: boolean; chain?: string; error?: string };
  corpus: { passages: number; clauses: number };
}

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    // Surface the server's message. "Request failed" tells the user nothing
    // and sends them to the network tab.
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* body was not JSON; the status text is all there is */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => json<Health>("/health"),
  passages: () => json<PassageSummary[]>("/api/passages"),
  passage: (id: string) => json<Passage>(`/api/passages/${encodeURIComponent(id)}`),
  clause: (id: string) => json<Clause>(`/api/clauses/${encodeURIComponent(id)}`),
  search: (query: string, k = 5, rerank = true) =>
    json<{ results: SearchHit[] }>("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, k, rerank }),
    }),
  auditPassage: (passage_id: string, enable_judge = true) =>
    json<PassageAudit>("/api/audit/passage", {
      method: "POST",
      body: JSON.stringify({ passage_id, enable_judge }),
    }),
  metrics: () => json<Record<string, any>>("/api/metrics"),
};

export interface StreamProgress {
  index: number;
  total: number;
  findings_total: number;
  result: PassageAudit;
}

/**
 * Stream a whole-document audit.
 *
 * Returns an abort function. Auditing 75 passages takes minutes, and a user
 * who navigates away must be able to stop it -- an EventSource left open keeps
 * the server working on results nobody will read.
 */
export function streamAudit(
  handlers: {
    onStart?: (total: number) => void;
    onPassage?: (progress: StreamProgress) => void;
    onDone?: (findingsTotal: number) => void;
    onError?: (message: string) => void;
  },
  options: { limit?: number; enableJudge?: boolean } = {},
): () => void {
  const params = new URLSearchParams();
  if (options.limit) params.set("limit", String(options.limit));
  params.set("enable_judge", String(options.enableJudge ?? true));

  const source = new EventSource(`/api/audit/stream?${params}`);

  source.addEventListener("start", (event) => {
    handlers.onStart?.(JSON.parse((event as MessageEvent).data).total);
  });
  source.addEventListener("passage", (event) => {
    handlers.onPassage?.(JSON.parse((event as MessageEvent).data) as StreamProgress);
  });
  source.addEventListener("done", (event) => {
    handlers.onDone?.(JSON.parse((event as MessageEvent).data).findings_total);
    source.close();
  });
  source.onerror = () => {
    // EventSource reconnects automatically on a transient drop, but a closed
    // connection after the server finished is not an error. Only report when
    // the stream is genuinely dead.
    if (source.readyState === EventSource.CLOSED) {
      handlers.onError?.("connection to the audit stream was lost");
    }
  };

  return () => source.close();
}
