# HANDOFF — The Auditor (v2)

Everything another engineer or AI assistant needs to continue this work.
Every number is read from a committed file; none is estimated.

**Repository:** https://github.com/ommundada16/ai_compliance_checker_ethosh-ignite-
**Branch:** `main` · **Commits:** 34 · **Tests:** 172 collected, 170 passing (2 skipped: integration tests that need a live service) · **Lint:** clean
**Last updated:** 2026-09-25

---

## 1. GOAL

Audit **Clinical Evaluation Reports** (CERs) for medical devices against
**EU MDR 2017/745** using an LLM and a RAG pipeline — and, critically, **prove
numerically that v2 is better than v1** rather than asserting it.

Two real documents drive everything:

| File | What it is |
|---|---|
| `data/guideline.pdf` | EU MDR 2017/745, the full regulation — 175 pages, 101k words |
| `data/source_file.pdf` | A real CER for a Double J ureteral stent (BIORAD MEDISYS) — 91 pages |

### The one insight that shapes every design decision

**This is not question-answering RAG, and treating it as such is why v1 failed.**

Normally a short human question retrieves against a document corpus. Here it is
inverted:

- the **corpus** is the regulation (1,320 clauses)
- the **query** is a ~250-word section of the document being audited

Consequences that recur throughout the codebase:

- A long, topically-mixed query embeds to a mushy average vector, so chunking
  has outsized impact.
- "Relevant" means *"this clause governs this passage"*, not *"this answers the
  question"* — a different labelling task entirely.
- Exact identifiers (`Annex XIV`, `Article 61(4)`, `PMCF`) matter because the
  literal string IS the meaning. That is why there is a sparse retrieval arm.

### Non-negotiable methodology

The evaluation harness was built **before** any improvement, and v1 was scored
with it. Without that, "v2 is better" is unfalsifiable. Preserve this: measure
first, change one variable per ablation row, keep negative results, state
limitations up front. **Never loosen the gold set to improve a number.**

---

## 2. CURRENT STATE

### Done and measured

| Area | Status |
|---|---|
| Frozen gold set (1,320 clauses, 75 passages, 159 primary labels) | complete |
| Retrieval metrics + v1 baseline | complete |
| v2 retriever: Qdrant, hybrid, RRF, cross-encoder rerank | complete, ablated |
| Table-aware parsing | complete |
| Audit pipeline + 6 layered guardrails | complete |
| Audit gold set + audit metrics | complete |
| FastAPI + SSE streaming | complete |
| React + Vite + TypeScript UI | complete, typechecks + builds |
| Docker, CI, docs | complete |

### Retrieval quality at k = 5

| Metric | v1 | v2 dense | v2 hybrid | **v2 rerank** | v1 → best |
|---|---|---|---|---|---|
| **Scope recall** | 0.113 | 0.138 | 0.107 | **0.200** | **+76%** |
| nDCG | 0.063 | 0.110 | 0.074 | **0.150** | **+137%** |
| MRR | 0.076 | 0.174 | 0.131 | **0.228** | **+198%** |
| MAP | 0.028 | 0.068 | 0.040 | **0.069** | **+147%** |
| Context precision | 0.054 | 0.099 | 0.061 | **0.144** | **+164%** |
| Hit rate | 0.187 | 0.267 | 0.213 | **0.387** | **+107%** |
| Context words to LLM | 4000 | 2398 | 3662 | **1456** | **−64%** |
| Latency / query | 17 ms | 339 ms | 353 ms | 14,061 ms | — |

### Recall at a matched context budget (the fair comparison)

Equal `k` is not equal information: v1 returns 800-word chunks, v2 returns
~66-word clauses.

| System | 200w | 400w | 800w | 2400w | 4000w |
|---|---|---|---|---|---|
| v1 baseline | 0.019 | 0.019 | 0.019 | 0.035 | 0.063 |
| **v2 rerank** | **0.054** | **0.080** | **0.111** | **0.126** | **0.144** |

At 800 words: **0.111 vs 0.019 — 5.8×**.

### Audit quality (v2_full, 19 passages)

| Metric | Value |
|---|---|
| Findings reported | 15 |
| Recall over known violations | **0.250** (1 of 4) |
| Precision | 0.067 |
| F1 | 0.105 |
| **False-positive rate** | **0.533** (8 of 15 clean passages flagged) |
| **Hallucination rate** | **0.000** |
| Guardrail rejections | 5 |
| Runtime | 576 s |

**These are poor and are reported as poor.** Section 5 explains why.

### Environment

- **Python 3.12** venv at `C:\Users\ommun\.venvs\auditor` (deliberately outside
  the OneDrive-synced project folder)
- **No torch.** Embedding and reranking run on ONNX Runtime, CPU only
- **Qdrant runs embedded** (in-process, `qdrant_local/`) — Docker not required
- **LLM:** Groq `openai/gpt-oss-120b`, judge `openai/gpt-oss-20b`; local Ollama
  `llama3.1:8b` as failover

---

## 3. FILES TOUCHED

10,228 lines of Python, 1,108 of TypeScript/CSS.

### New — the v2 pipeline (`src/auditor/`)

| File | Purpose |
|---|---|
| `parsing/mdr.py` | MDR → 1,320 clauses with hierarchical IDs (`Art.61.3.a`) |
| `parsing/cer.py` | CER → 75 passages; table-aware, bullet repair, boilerplate stripping |
| `embedding.py` | bge-base dense + BM25 sparse, ONNX/CPU |
| `retrieval/qdrant_store.py` | Named dense+sparse vectors; server **or** embedded |
| `retrieval/fusion.py` | Reciprocal Rank Fusion |
| `retrieval/rerank.py` | bge-reranker-base cross-encoder |
| `llm/base.py` | `JSONProvider` interface, JSON recovery, typed errors |
| `llm/providers.py` | Groq (token-paced), Ollama (native API), failover chain |
| `audit/schema.py` | Enforced enums, mandatory quote, drop reasons |
| `audit/guardrails.py` | Span grounding, citation check, injection sanitising |
| `audit/pipeline.py` | retrieve → prompt → parse → guard → judge |
| `evaluation/metrics.py` | Recall@k, nDCG, MRR, MAP, context precision |
| `evaluation/gold.py` | Scope resolution, `scope_recall` |
| `evaluation/audit_metrics.py` | Recall, FP rate, hallucination rate |
| `baselines/v1_retriever.py` | Frozen v1, runnable; `measure_embedding_window()` |
| `baselines/coverage.py` | Maps v1 chunks → clause IDs so v1 can be scored |
| `resources.py` | Memory guard + resource plan |

### New — gold set and scoring (`tools/`)

`build_clause_corpus.py`, `build_protocol_passages.py`, `mdr_section_map.py`
(the expert mapping), `build_gold_retrieval.py`, `audit_expectations.py`,
`propose_labels_llm.py`, `score_baseline_v1.py`, `score_v2.py`,
`run_audit_eval.py`, `build_comparison_report.py`

### New — service and UI

`api/main.py`, `api/deps.py`, `frontend/src/**` (App, PassageView,
MetricsPanel, SearchPanel, api.ts, styles.css)

### New — tests (`tests/`, 172 collected, 170 passing)

`test_metrics.py`, `test_coverage.py`, `test_fusion_and_gold.py`,
`test_guardrails.py`, `test_audit_metrics.py`, `test_llm_providers.py`,
`test_clause_corpus.py`, `test_protocol_passages.py`, `test_gold_retrieval.py`,
`test_parsers_match_frozen.py`, `test_api.py`

### New — infra and docs

`Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`,
`.gitattributes`, `pyproject.toml`, `requirements{,-dev,-v1}.txt`,
`README.md`, `docs/COMPARISON.md`, `docs/HANDOFF.md`

### Frozen artefacts (`eval_data/`, committed)

`clauses.jsonl` (1,320), `passages.jsonl` (75), `gold_retrieval.jsonl` (75),
`results/v1_baseline.json`, `results/v2_ablation.json`, `results/audit_eval.json`

### Untouched — v1, kept as the baseline

`app.py`, `auditor.py`, `ingest.py`, `retriever.py`, `run_audit.py`,
`schema.py`, `report.py`. **Do not modify these.** They are the reference
point every comparison is made against.

---

## 4. WHAT CHANGED

### Architecture

| | v1 | v2 |
|---|---|---|
| Parsing | `text.split()` over concatenated pages | Layout-aware; tables → key-value rows |
| Chunking | Fixed 800 words / 100 overlap | Clause-level, hierarchical IDs |
| Embedding | all-MiniLM-L6-v2 (256 word-pieces) | bge-base-en-v1.5 (512) |
| Sparse | none | BM25 |
| Index | numpy in RAM, rebuilt each run | Qdrant, persisted |
| Retrieval | dense, top-3 | hybrid → RRF → cross-encoder |
| Grounding | first 40 chars, substring | character spans: exact → whitespace → fuzzy |
| Citations | free text | clause ID verified against what was retrieved |
| LLM | Ollama hardcoded | pluggable provider + failover |
| Evaluation | **none** | frozen gold set, 3 metric layers, ablation |

### Guardrails (cheapest first, so the judge only sees clean candidates)

| Guardrail | Rejects |
|---|---|
| Schema | Invented categories/severities, missing quote |
| Abstention | Confidence below threshold |
| **Citation** | A clause that was never retrieved |
| Grounding | A quote not locatable in the passage |
| Judge | A finding a **different** model won't support |
| Injection | Instruction-like text inside the untrusted PDF |

Every rejection records a **typed reason**, so the guardrails are themselves
measurable — a guardrail that removes more true positives than false ones is a
bad guardrail, and without the reasons there is no way to tell.

### The single most important finding

v1 scored 2.9% recall. That looked like a harness bug, so it was investigated
before being reported. It was not a bug:

> `all-MiniLM-L6-v2` accepts 256 word-pieces. Regulatory English exceeds two
> pieces per word, so only **106 words of every 800-word chunk** were ever
> encoded — 13%. Silently. No error, no warning.

| Measurement | Value |
|---|---|
| Words embedded per chunk | **106** of 800 (13%) |
| Clauses ever embedded | **334 / 1,320** |
| **Hard recall ceiling** | **30.1%** — no `k` could beat it |
| v2 gold clauses truncated | **0** |

`measure_embedding_window()` proves it by binary-searching the shortest prefix
whose embedding is *identical* to the full chunk's. The result is written into
the baseline JSON so the claim travels with the numbers.

### Gold-set design decisions worth preserving

- **Graded relevance**: primary (2) = the clause a regulator cites first;
  secondary (1) = legitimately related. Recall counts primary only; nDCG uses
  both.
- **Labels name a SCOPE**: `Annex.VIII` is satisfied by `Annex.VIII.5`;
  `Art.61.1` has no children and matches only itself.
- **Non-circular labelling**: the expert map is hand-authored from the
  regulation; the LLM cross-check selects from a COMPLETE ENUMERATION of the
  regulation's structure and never sees retrieval output. It may only add
  secondary labels, never create a primary one. A test enforces this.
- **v1 is scored generously** (its chunks are credited with every clause they
  contain), making the v2 delta a conservative claim.

---

## 5. WHAT FAILED

### 5.1 The audit results are poor — and the breakdown says why

| Expected finding | Retrieved? | Reported? | Verdict |
|---|---|---|---|
| `CER.4.3.2.1` → Art. 61(4) | yes | yes | **found** |
| `CER.2.11` → Annex I §23 | **yes** | **no** | LLM miss |
| `CER.2.19` → Annex II §1 | no | — | retrieval miss |
| `CER.4.3.2.2#1` → Art. 83 | no | — | retrieval miss |

**Two retrieval failures, one reasoning failure.** A better prompt would not
have recovered the first two; better retrieval would not have recovered the
third. This attribution is the entire reason the two layers are measured apart.

### 5.2 The false-positive rate is the binding problem — and it is architectural

**8 of 15 clean passages drew a finding. Precision 0.067.**

The pipeline audits each passage **in isolation**. The model is shown section
2.1 ("Identification of device(s)", 54 administrative words) together with
Annex XIV §1(a), which lists what a clinical evaluation plan must contain. It
correctly observes that this section contains no such plan and reports it — but
the content is in **section 4**, which it never sees.

Those findings carried confidence **0.88–0.97**, so raising the abstention
threshold will not separate them. This is not a prompt defect.

### 5.3 Hybrid search made retrieval worse — a kept negative result

Scope recall at k=5: hybrid **0.107** vs dense **0.138**. BM25 assumes short
keyword queries; these are 250-word passages, so the sparse arm matches common
legal vocabulary and injects noise that RRF then rewards for "cross-arm
agreement". The reranker recovers it. Kept in the ablation **because** it is
negative.

### 5.4 Operational failures, and what fixed them

| Failure | Cause | Fix |
|---|---|---|
| Desktop froze, disk 100% | Three ML jobs at once on 15.7 GB; Windows swapped | Memory guard + resource plan + embedded Qdrant lock |
| 18 of 19 passages lost to HTTP 429 | Token pacer existed in one tool only | Pacing moved **into** `GroqProvider` |
| Audit run took 50+ min then died | Longest MDR clause is 2,351 words → one request cost ~25,700 tokens = 3.2 min of pacing | Clause text capped at 250 words in the prompt → 4,589 tokens; run now 576 s |
| qwen3:4b returned empty responses | Ollama's OpenAI shim silently drops `think:False` | Switched to the native `/api/chat` endpoint |
| Gold labels marked correct retrievals wrong | `Annex.VIII.1` was written meaning "Annex VIII"; that ID is "DURATION OF USE" | Scope-based matching, applied identically to v1 and v2 |
| API returned scores contradicting its own order | With rerank off, order came from RRF but raw dense scores were returned | Return the score of whichever stage decided the order |

**Beware the Groq dashboard.** Its "Rate Limit" line reads 80.5K, which looks
like 10× the available headroom. It is the per-minute limit scaled to the
graph's 10-minute buckets. The API header is authoritative:
`x-ratelimit-limit-tokens: 8000`.

### 5.5 The 250-word clause cap loses more context than it first appears

The cap that made the audit run fast (see 5.4) is a real information trade-off,
and the first framing of it understated the cost.

| | |
|---|---|
| Clauses over 250 words in the corpus | 46 of 1,320 (**3.5%**) |
| Clauses over 250 words **actually retrieved** in the audit run | 41 of 95 (**43%**) |

Long clauses are retrieved far more often than their share of the corpus,
because more text means more chance of matching a query. So the cap touches
**roughly half the context the model is shown**, not 3.5% of it.

Consequence: a violation described beyond word 250 of a clause cannot be found.
Whether that actually costs recall is **unmeasured** — see Next Steps §1.

The clauses over 250 words that were shown during the run:
`Annex.I.10.h`, `Annex.I.11.d`, `Annex.I.23.s#1`, `Annex.IX.2.e`, `Annex.IX.4`,
`Annex.VI.A.2`, `Annex.VII.4.b#2`, `Annex.VII.4.d#3`, `Annex.VII.4.e#1`,
`Annex.VIII.4`, `Annex.VIII.5`, `Annex.VIII.7`, `Annex.XIV.A.3`, `Annex.XV.2`,
`Annex.XV.2#1`, `Art.117`, `Art.2`, `Art.2.c#1`

### 5.6 Is the v1 vs v2 comparison fair on model choice?

A reasonable objection: v1 originally ran on a local 8B model while v2 runs on
Groq's 120B. Checked rather than assumed:

**The retrieval comparison uses no LLM at all.** `tools/score_baseline_v1.py`
and `tools/score_v2.py` contain zero LLM calls — verified by grep for
`complete_json`, `GroqProvider`, `OllamaProvider`, `chat.completions`. Both are
pure embedding retrieval over the same gold set with the same metric code.

So **scope recall +76%, nDCG +137%, MRR +198% are unaffected by model choice.**

**The audit comparison is designed to hold the LLM constant**: `v1_retrieval`
and `v2_full` both use the same Groq model, and only the *retriever* differs.
But `v1_retrieval` has not actually been run, so **there is currently no audit
comparison at all** — fair or otherwise. That is the gap, not the model choice.

### 5.7 Known limitations — state these before anyone asks

- **Audit gold set is not exhaustive** — 4 authored violations, 15 clean
  passages. Audit recall is a **lower bound**, not an estimate.
- **"Clean" means a reviewer would not expect a finding**, not "provably
  compliant".
- **Retrieval labels are an expert rubric**, not verified by a regulatory
  professional.
- **Absolute retrieval numbers are low** (0.200 scope recall). Selecting 2–3
  governing provisions from 1,320 into a top-5 is hard; random ≈ 0.4%. The
  **relative** improvement is the claim.
- **Reranking costs ~14 s/query on CPU** — too slow for interactive use.
- **Clause text is capped at 250 words in the prompt**, which touches ~43%
  of the context actually shown. The cost in recall is unmeasured (see 5.5).
- **Only `v2_full` has audit numbers.** `v1_retrieval` and `v2_no_guards` have
  not been run, so the audit table has no ablation.
- **LLM label cross-check not merged** — `tools/propose_labels_llm.py` works
  and is rate-limit-safe, but its output is not folded into the gold set.
- **UI not verified in a browser** — it typechecks and builds, but has not been
  run against the live API and screenshotted.

---

## 6. NEXT STEPS

In descending order of value.

### 1. Measure what the 250-word clause cap costs

Settle §5.5 with data instead of an assumption. Run the audit twice, changing
only `MAX_CLAUSE_WORDS_IN_PROMPT` in `src/auditor/audit/pipeline.py`:

```bash
# baseline, current setting
python tools/run_audit_eval.py --provider groq --configs v2_full \
  --out eval_data/results/audit_cap250.json

# edit MAX_CLAUSE_WORDS_IN_PROMPT = 600, then:
python tools/run_audit_eval.py --provider groq --configs v2_full \
  --out eval_data/results/audit_cap600.json
```

Compare recall and false-positive rate.

- **If they are the same**, the cap is free and the current setting stays.
- **If recall improves at 600**, the cap is costing real findings; raise it and
  accept the slower run, or cap by *tokens* rather than words so only the
  genuinely huge clauses are trimmed.

~10 min per run, one at a time. Do this **before** trusting the audit numbers
as a baseline for anything else.

### 2. Run `v1_retrieval` so the audit comparison actually exists

Right now only `v2_full` has audit numbers, so there is no audit comparison.
This is the run that produces one, and it holds the LLM constant so the
difference is attributable to retrieval alone:

```bash
python tools/run_audit_eval.py --provider groq --configs v1_retrieval
python tools/run_audit_eval.py --provider groq --configs v2_no_guards
```

~10 min each. `v2_no_guards` additionally quantifies what the guardrails are
worth in *findings*, not just in retrieval metrics.

### 3. Decide: Groq or local for evaluation runs

Both are viable; the trade-off is measured, not obvious:

| | Groq (paced) | Local `llama3.1:8b` |
|---|---|---|
| 19 passages | **576 s** (measured) | ~300 s (estimated at ~16 s/call) |
| Rate limit | 8,000 tokens/min | none |
| Local RAM | ~0 | **+3.5 GB** |
| Model | 120B | 8B |

Local may now be **faster**, because there is no rate limit to pace against.
The costs are 3.5 GB of RAM and a much weaker model. Use
`--provider local` to try it. If quality holds, local removes the rate-limit
problem entirely and makes the clause cap unnecessary.

### 4. Document-level reconciliation — fixes the biggest problem

Before reporting "X is missing", check whether X appears elsewhere in the
document. Most of the 8 false positives would disappear. Approach: after the
per-passage pass, for each "missing X" finding, run a retrieval query for X
over the **CER's own passages**; if a strong match exists, downgrade or drop.

Expected: FP rate 0.53 → well under 0.2. **Do this first.**

### 5. Tell the model what kind of section it is reading

Pass the section title and a coarse type (identification / description /
analysis / evidence). An identification section cannot breach a
clinical-evaluation-plan requirement. Cheap, and complements step 1.

### 6. Complete the audit ablation

Run `v1_retrieval` and `v2_no_guards` (~10 min each, Groq). This turns one
audit row into a real comparison and quantifies what retrieval quality and the
guardrails are each worth in *findings*, not just in retrieval metrics.

```bash
python tools/run_audit_eval.py --provider groq --configs v1_retrieval
python tools/run_audit_eval.py --provider groq --configs v2_no_guards
```

### 7. Fix the two retrieval misses

`CER.2.19` (Annex II §1) and `CER.4.3.2.2#1` (Art. 83) were never retrieved.
Investigate: query expansion, or a section-title-aware query, or raising `k`
for the audit path specifically.

### 8. Reranker latency

14 s/query is too slow interactively. Options: batch the cross-encoder, use
`jinaai/jina-reranker-v1-turbo-en` (0.15 GB vs 1.04 GB), or rerank only the
top-10 rather than top-25.

### 9. Verify the UI in a browser and screenshot it

`uvicorn api.main:app --reload` plus `cd frontend && npm run dev`.

### 10. Merge the LLM label cross-check

```bash
python tools/propose_labels_llm.py
python tools/build_gold_retrieval.py --merge eval_data/llm_proposals.jsonl
```

Then re-run both scoring scripts, since the gold set will have changed.

---

## Reproducing every number

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements-dev.txt
cp .env.example .env            # add GROQ_API_KEY

python tools/build_clause_corpus.py       # 1320 clauses
python tools/build_protocol_passages.py   # 75 passages
python tools/build_gold_retrieval.py      # graded labels
python tools/score_baseline_v1.py         # v1 numbers
python tools/score_v2.py --reindex        # v2 ablation
python tools/run_audit_eval.py --provider groq --configs v2_full
python tools/build_comparison_report.py   # docs/COMPARISON.md
```

CI rebuilds the gold set from the source PDFs on every push and **fails if a
single byte differs**. The artefacts are frozen but the parser is live code; a
cleaning tweak silently redefines what a clause *is* while the labels keep
pointing at old IDs. Nothing would break — the numbers would just stop meaning
what they did.

---

## Machine constraints — read before running anything

The development machine is **15.7 GB RAM, 4 GB VRAM (RTX 3050 Laptop)**.

- **Run ONE ML or LLM job at a time.** Verify the previous one has exited; a
  completion notification is not the same as having checked. Three concurrent
  jobs exhausted memory, sent Windows into swap, and froze the desktop.
- **Prefer Groq over local Ollama** for batch work — remote inference costs
  ~0 local RAM and is faster than an 8B model that does not fit in 4 GB of VRAM.
- `llama3.1:8b` is 4.92 GB and spills into RAM (~3.5 GB extra).
- Evaluation scripts print a resource plan and abort below a memory floor.
  **Keep that behaviour.**
- **Never pipe a long background job through `grep`** without `--line-buffered`
  — it block-buffers and you lose all output. This happened twice.

## Context about the author

Final-year Computer Engineering student building this for a portfolio and
interviews. **Interview-defensibility matters more than flattering numbers.**
The methodology is the product: measuring before changing, one variable per
ablation row, keeping negative results, and stating limitations up front.
