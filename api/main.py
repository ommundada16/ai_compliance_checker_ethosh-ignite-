"""FastAPI service over the v2 auditor.

Endpoints:

    GET  /health              dependency status, per service
    GET  /api/passages        the parsed CER sections
    GET  /api/passages/{id}   one passage with its full text
    GET  /api/clauses/{id}    one MDR clause
    POST /api/search          retrieval only, for inspecting what the LLM sees
    POST /api/audit/passage   audit one passage, synchronously
    GET  /api/audit/stream    audit the document, streaming findings over SSE
    GET  /api/metrics         the committed evaluation results

Auditing a 75-passage document takes minutes on local inference, so the
document-level endpoint streams. A request that blocks for ten minutes is a
request that times out somewhere in the middle of the stack, and a progress bar
that only moves at the end is not a progress bar.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from api.deps import (
    RESULTS_DIR,
    build_retriever,
    get_judge,
    get_provider,
    get_store,
    load_clauses,
    load_passages,
)

app = FastAPI(
    title="The Auditor",
    version="2.0.0",
    description="Audits Clinical Evaluation Reports against EU MDR 2017/745.",
)

# The React dev server runs on a different origin; production serves the built
# assets from the same one, so this is permissive only for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- models ---------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=25)
    rerank: bool = True


class AuditRequest(BaseModel):
    passage_id: str
    k: int = Field(default=5, ge=1, le=25)
    enable_judge: bool = True
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


# --- health ---------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    """Per-dependency status.

    Reports each service separately rather than a single boolean, because
    "unhealthy" without a reason sends whoever is on call to read logs.
    """
    status: dict[str, Any] = {"api": "ok"}

    try:
        status["qdrant"] = {"ok": True, "points": get_store().count()}
    except Exception as exc:  # noqa: BLE001
        status["qdrant"] = {"ok": False, "error": str(exc)[:200]}

    try:
        status["provider"] = {"ok": True, "chain": get_provider().describe()}
    except Exception as exc:  # noqa: BLE001
        status["provider"] = {"ok": False, "error": str(exc)[:200]}

    passages = load_passages()
    status["corpus"] = {"passages": len(passages), "clauses": len(load_clauses())}
    status["ok"] = bool(passages) and status.get("qdrant", {}).get("ok", False)
    return status


# --- data -----------------------------------------------------------------

@app.get("/api/passages")
def list_passages() -> list[dict]:
    """Section list. Text is omitted deliberately -- the full document is ~19k
    words and the sidebar only needs titles."""
    return [
        {
            "passage_id": p["passage_id"],
            "section": p["section"],
            "section_title": p["section_title"],
            "page_start": p["page_start"],
            "page_end": p["page_end"],
            "n_words": p["n_words"],
        }
        for p in load_passages()
    ]


@app.get("/api/passages/{passage_id}")
def get_passage(passage_id: str) -> dict:
    for p in load_passages():
        if p["passage_id"] == passage_id:
            return p
    raise HTTPException(404, f"unknown passage {passage_id}")


@app.get("/api/clauses/{clause_id}")
def get_clause(clause_id: str) -> dict:
    clause = load_clauses().get(clause_id)
    if clause is None:
        raise HTTPException(404, f"unknown clause {clause_id}")
    return clause


# --- retrieval ------------------------------------------------------------

@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    """Retrieval without auditing.

    Exposed so the exact context the LLM will see can be inspected. Most RAG
    debugging is really "what did it actually retrieve", and answering that
    should not require reading a log.
    """
    retrieve = build_retriever(use_reranker=request.rerank)
    try:
        hits = retrieve(request.query, request.k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"retrieval failed: {exc}") from exc
    return {
        "query": request.query,
        "reranked": request.rerank,
        "results": [
            {
                "clause_id": h.clause_id,
                "score": h.score,
                "path": h.path,
                "text": h.text,
                "page_start": h.page_start,
                "n_words": h.n_words,
            }
            for h in hits
        ],
    }


# --- auditing -------------------------------------------------------------

def _pipeline(k: int, enable_judge: bool, min_confidence: float):
    from auditor.audit.pipeline import AuditConfig, AuditPipeline

    return AuditPipeline(
        get_provider(),
        build_retriever(),
        AuditConfig(top_k=k, enable_judge=enable_judge, min_confidence=min_confidence),
        judge=get_judge() if enable_judge else None,
    )


@app.post("/api/audit/passage")
def audit_passage(request: AuditRequest) -> dict:
    passage = next(
        (p for p in load_passages() if p["passage_id"] == request.passage_id), None
    )
    if passage is None:
        raise HTTPException(404, f"unknown passage {request.passage_id}")

    pipeline = _pipeline(request.k, request.enable_judge, request.min_confidence)
    result = pipeline.audit_passage(passage)
    return result.model_dump(mode="json")


@app.get("/api/audit/stream")
async def audit_stream(
    limit: int = Query(default=0, ge=0, le=200),
    k: int = Query(default=5, ge=1, le=25),
    enable_judge: bool = Query(default=True),
) -> EventSourceResponse:
    """Audit the document, emitting one event per passage.

    Each passage is run in a worker thread: the pipeline is synchronous and
    CPU/IO bound, and running it inline would block the event loop so nothing
    would actually stream.
    """
    passages = load_passages()
    if limit:
        passages = passages[:limit]
    pipeline = _pipeline(k, enable_judge, 0.35)

    async def events():
        yield {"event": "start", "data": json.dumps({"total": len(passages)})}
        findings_total = 0
        for index, passage in enumerate(passages, start=1):
            try:
                result = await asyncio.to_thread(pipeline.audit_passage, passage)
                payload = result.model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001 - one passage must not kill the stream
                payload = {
                    "passage_id": passage["passage_id"],
                    "section": passage["section"],
                    "findings": [],
                    "dropped": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            findings_total += len(payload.get("findings", []))
            yield {
                "event": "passage",
                "data": json.dumps(
                    {
                        "index": index,
                        "total": len(passages),
                        "findings_total": findings_total,
                        "result": payload,
                    }
                ),
            }
        yield {"event": "done", "data": json.dumps({"findings_total": findings_total})}

    return EventSourceResponse(events())


# --- evaluation results ---------------------------------------------------

@app.get("/api/metrics")
def metrics() -> dict:
    """The committed evaluation results, served as-is.

    The UI shows the same numbers the repository records; there is no separate
    live computation that could disagree with them.
    """
    out: dict[str, Any] = {}
    for name in ("v1_baseline", "v2_ablation", "audit_eval"):
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        # per_query and full reports are large and the dashboard never reads
        # them; strip so the endpoint stays a few kilobytes.
        if isinstance(data, dict):
            data.pop("per_query", None)
            for value in data.get("results", {}).values():
                if isinstance(value, dict):
                    value.pop("per_query", None)
            for value in data.values():
                if isinstance(value, dict):
                    value.pop("report", None)
        out[name] = data
    return out
