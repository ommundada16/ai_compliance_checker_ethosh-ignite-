# The Auditor

Audits Clinical Evaluation Reports against **EU MDR 2017/745** with a RAG
pipeline, and — the part that matters — **measures whether it actually works**.

The regulation is the corpus (1320 clauses). A section of the document under
audit is the query. That inversion of the usual RAG arrangement drives most of
the design decisions below.

---

## The result in one table

Retrieval quality at k = 5, identical frozen gold set, identical metric code:

| Metric | v1 | v2 | |
|---|---|---|---|
| Scope recall | 0.113 | **0.200** | **+77%** |
| nDCG | 0.063 | **0.150** | **+138%** |
| MRR | 0.076 | **0.228** | **+200%** |
| Context precision | 0.054 | **0.144** | **+167%** |
| Context words sent to the LLM | 4000 | **1456** | **−64%** |

Better answers from a third of the context.

Full numbers, including the ablation and the negative results:
**[docs/COMPARISON.md](docs/COMPARISON.md)**

---

## The finding the project is actually about

v1 scored 2.9% recall. That looked like a harness bug, so it was investigated
before being reported. It was not a bug:

> `all-MiniLM-L6-v2` accepts 256 word-pieces. Regulatory English runs well over
> two pieces per word, so only **106 of every 800-word chunk** was ever encoded
> — 13%. Silently. No error, no warning.

Measured consequences:

- **334 of 1320 clauses** were ever embedded at all
- a hard **recall ceiling of ~31%** — no value of `k` could beat it, because the
  rest was not in the index in any form

v1's problem was never ranking. **87% of the regulation it was auditing against
was invisible to its retriever.** `measure_embedding_window()` proves this by
binary-searching the shortest prefix whose embedding is *identical* to the full
chunk's, and the result is written into the baseline JSON so the claim travels
with the numbers.

v2 truncates **zero** gold clauses.

---

## Architecture

```
  PDF
   |
   +-- PARSE      layout-aware; tables rendered as "header: value" rows
   |              (flattened, a table reads "Categor Characteris Device 1...")
   |
   +-- CHUNK      clause-level units with hierarchical IDs (Art.61.3.a)
   |
   +-- EMBED      bge-base-en-v1.5 dense + BM25 sparse, ONNX on CPU
   |
   +-- INDEX      Qdrant, named vectors, persisted
                    |
  Query (a CER section)
   |
   +-- HYBRID     dense + sparse, fused with RRF
   +-- RERANK     bge-reranker-base cross-encoder, top-25 -> top-5
   |
   +-- AUDIT      LLM with clause IDs it must cite
   |
   +-- GUARDRAILS schema -> abstention -> citation check -> grounding -> judge
   |
   +-- FastAPI --SSE--> React
```

The GPU budget is 4 GB and it belongs to the LLM, so embedding and reranking run
on CPU via ONNX. There is no torch in the v2 dependency set.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                 # add a GROQ_API_KEY, or use Ollama
```

Qdrant runs **embedded by default** — no Docker required. `docker-compose.yml`
is there for when you want a real server.

```bash
python tools/score_v2.py --reindex     # build the index and score retrieval
uvicorn api.main:app --reload          # http://localhost:8000/docs
cd frontend && npm install && npm run dev
```

---

## Evaluation

Three layers, because they fail independently.

| Layer | Metrics | Question |
|---|---|---|
| Retrieval | Scope recall, nDCG, MRR, MAP, context precision | Did the right clauses come back, high up? |
| Audit | Recall, precision, F1, **false-positive rate**, hallucination rate | Were the right violations reported, and only those? |
| Systems | p50/p95 latency, context words, index time | Is it usable and affordable? |

The gold set is **frozen**: 1320 clauses, 75 passages, an expert MDR mapping
authored from the regulation, and graded relevance (primary / secondary).

CI **rebuilds the gold set from the source PDFs on every push and fails if a
single byte differs.** The artefacts are frozen but the parser is live code; a
cleaning tweak silently redefines what a clause *is* while the labels keep
pointing at the old IDs. Nothing would break — the numbers would just quietly
stop meaning what they did.

---

## What this does **not** show

Stated here rather than buried, because these are the first questions a careful
reader should ask:

- **The audit gold set is not exhaustive.** Four authored violations, fifteen
  clean passages. Audit recall is a lower bound, not an estimate.
- **"Clean" means a reviewer would not expect a finding**, not "provably
  compliant".
- **Retrieval labels are an expert rubric** cross-checked by an independent
  model — not verified by a regulatory professional.
- **Hybrid search made things worse** at low k, and is kept in the ablation
  precisely because it is a negative result. BM25 assumes short keyword
  queries; these are 250-word passages, so the sparse arm matches common legal
  vocabulary and injects noise that fusion then rewards for "agreement".
- **Absolute numbers are low.** Picking 2–3 governing provisions out of 1320
  into a top-5 is genuinely hard. The *relative* improvement is the claim.

---

## Guardrails

Cheapest first, so the expensive judge only sees findings that are already
grounded and correctly cited. Every rejection records a typed reason, because
the distribution of those reasons says which failure mode the model actually
has — and whether a guardrail removes more false positives than true ones.

| Guardrail | v1 | v2 |
|---|---|---|
| Schema | Categories listed in the prompt only | Enforced enums |
| Abstention | None | Confidence threshold |
| Citation | Free text | Must be a clause that was **retrieved** |
| Grounding | First 40 characters, substring | Character spans: exact → whitespace → bounded fuzzy |
| Judge | None | A **different** model verifies |
| Injection | None | The PDF is untrusted input |

The citation check matters most. The model cannot have read a clause it was
never shown, so a citation outside the retrieved set comes from training data
rather than from the regulation in front of it — the failure mode most likely
to look convincing and be wrong. v1 could not detect it at all.

---

## Layout

```
src/auditor/
  parsing/      MDR and CER structural parsers (shared with the gold set)
  chunking      clause-level, in parsing/
  embedding.py  dense + sparse, ONNX on CPU
  retrieval/    Qdrant store, RRF fusion, cross-encoder rerank
  llm/          provider interface: Groq, Ollama, failover
  audit/        schema, prompts, guardrails, pipeline
  evaluation/   metrics, gold-set resolution, audit scoring
  baselines/    frozen v1, kept runnable so v2 has something to beat
api/            FastAPI, SSE streaming
frontend/       React + Vite + TypeScript
tools/          gold-set builders and scoring scripts
tests/          172 tests (170 pass, 2 skip without a live service)
```

---

## A note on running this locally

The LLM provider is pluggable on purpose. With `GROQ_API_KEY` set, inference
happens remotely and the local footprint is ~2 GB. Running `llama3.1:8b`
locally instead adds ~3.5 GB, and a 4.92 GB model does not fit in 4 GB of VRAM,
so it spills into RAM.

The evaluation scripts print a resource plan before they start and abort
cleanly if free memory drops below a floor. That is not defensive
over-engineering: three of these jobs left running at once exhausted a 16 GB
machine, sent Windows into swap, and froze the desktop. A batch job that cannot
finish should stop and say so.
