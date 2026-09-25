# HANDOFF — The Auditor (v2)

Everything another engineer or AI assistant needs to continue this work.
Every number is read from a committed file; none is estimated.

**Repository:** https://github.com/ommundada16/ai_compliance_checker_ethosh-ignite-
**Branch:** `main` · **Commits:** 38 (not pushed) · **Tests:** 198 collected, 198 passing · **Lint:** clean · **Frontend:** typechecks and builds
**Last updated:** 2026-09-25 (end of session)

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
| `audit/pipeline.py` | retrieve → prompt → parse → guard → judge → reconcile; `retrieval_query()`, `DOCUMENT_CONTEXT` |
| `audit/reconcile.py` | Checks "X is missing" findings against the rest of the CER; verified-quote rule, fails open |
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
`run_audit_eval.py`, `build_comparison_report.py`,
`eval_reconcile_offline.py` (replays reconciliation over stored runs)

### New — service and UI

`api/main.py`, `api/deps.py`, `frontend/src/**` (App, PassageView,
MetricsPanel, SearchPanel, api.ts, styles.css)

### New — tests (`tests/`, 198 collected, 198 passing)

`test_metrics.py`, `test_coverage.py`, `test_fusion_and_gold.py`,
`test_guardrails.py`, `test_audit_metrics.py`, `test_llm_providers.py`,
`test_clause_corpus.py`, `test_protocol_passages.py`, `test_gold_retrieval.py`,
`test_parsers_match_frozen.py`, `test_api.py`, `test_reconcile.py`,
`test_prompt_context.py`

### New — infra and docs

`Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`,
`.gitattributes`, `pyproject.toml`, `requirements{,-dev,-v1}.txt`,
`README.md`, `docs/COMPARISON.md`, `docs/HANDOFF.md`

### Frozen artefacts (`eval_data/`, committed)

`clauses.jsonl` (1,320), `passages.jsonl` (75), `gold_retrieval.jsonl` (75),
`results/v1_baseline.json`, `results/v2_ablation.json`, `results/audit_eval.json`;
added 2026-09-25: `results/v2_query_ablation.json`, `results/reconcile_offline.json`,
`results/audit_cap250.json`, `results/audit_cap600.json`. `results/dryrun_local/` holds
local-model dry runs and is **not reportable** (see its README). `.retrieval_cache.json`
is regenerable and git-ignored

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
- **UI verified in a browser on 2026-09-25**, which found and fixed a real bug
  (see §1f). Section list, passage view, search, metrics and single-passage audit
  all work against the live API; the whole-document audit stream and the
  "Audit this section" button were not clicked, because the judge model's daily
  token budget was spent.

---

## 6. NEXT STEPS

### STATUS BOARD (updated 2026-09-25, end of session)

| # | Step | Status |
|---|---|---|
| 1 | Clause-cap experiment | **DONE** — keep 250 (§1a) |
| 4 | Document-level reconciliation | **DONE**, measured offline, modest (§1b) |
| 5 | Tell the model what it is reading | **BUILT** (`document_context`), **not yet measured end-to-end** (§1d) |
| 7 | Retrieval misses | **PARTLY DONE** — section-title query adopted (§1c); `CER.2.19` is a parsing defect, left as is |
| 8 | Reranker latency | Measured 8.2 s/query with free RAM (14 s was under memory pressure); pool-size test not run |
| 2, 6 | Audit ablation on Groq (`v1_retrieval`, `v2_no_guards`, `v2_title`, `v2_reconcile`, `v2_context`) | **PENDING — token-gated, see "Groq plan" below** |
| 3 | Groq vs local for evaluation | **DECIDED** — Groq for reported numbers, local only as a dry run (§1e) |
| 9 | UI check in a browser | see the end of this file for its status |
| 10 | Merge LLM label cross-check | **DELIBERATELY SKIPPED** — see below |

### Groq plan (the one remaining block of work)

`openai/gpt-oss-120b` on Groq's free tier allows **200,000 tokens per rolling 24 h**;
one audit run costs about **90,000**, so roughly **2 runs per day**. Today's runs
used the budget; it refills continuously, about 8K tokens per hour. The `v2_full`
baseline already exists (three samples: `audit_eval.json`, `audit_cap250.json`,
`audit_cap600.json`), so it need not be re-run.

**Both models matter.** The generator is `gpt-oss-120b`, but every config except
`v1_retrieval` and `v2_no_guards` also calls `gpt-oss-20b` as judge (and
`v2_reconcile`/`v2_context` call it again to reconcile), so **the 20b budget can
run out first** — it did, at the end of 2026-09-25, after the local dry runs
spent 199,220 of 200,000 judging findings. Both budgets refill continuously
(about 8K tokens per hour each) rather than all at once, so waiting a few hours
helps but not fully. Check both before each run (prints `Used N`):

```bash
python - <<'EOF'
import os; from dotenv import load_dotenv; load_dotenv('.env', override=True)
from groq import Groq
c = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
for m in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
    try: c.chat.completions.create(model=m, messages=[{"role":"user","content":"hi "*3000}], max_tokens=500); print(m, "has room")
    except Exception as e: print(m, str(e)[str(e).find("Limit"):][:110])
EOF
```

Run only when `Used` is below ~100,000 for the 120b and below ~120,000 for the 20b.
A run that prints `CONTAMINATED` means the judge budget ran out part-way: discard
it and rerun later. One config per call, in this order. The
retrieval cache is already built (`eval_data/.retrieval_cache.json`), so no
retrieval models load:

```bash
# day 1 -- the two rows that matter most
python tools/run_audit_eval.py --provider groq --configs v1_retrieval --out eval_data/results/audit_groq_v1.json
python tools/run_audit_eval.py --provider groq --configs v2_title     --out eval_data/results/audit_groq_title.json
# day 2
python tools/run_audit_eval.py --provider groq --configs v2_no_guards --out eval_data/results/audit_groq_noguards.json
python tools/run_audit_eval.py --provider groq --configs v2_reconcile --out eval_data/results/audit_groq_reconcile.json
# day 3, optional (dev-set tuned prompt)
python tools/run_audit_eval.py --provider groq --configs v2_context   --out eval_data/results/audit_groq_context.json
```

Rules for reading the results:

- **A run printing `CONTAMINATED` must not be reported.** The judge and reconciler
  fail open, so an exhausted 20b budget silently keeps findings that skipped a
  check; the tool now counts those failures and refuses to hide them.
- **One run per config is a sample, not a measurement.** `v2_full` alone scored FP
  rate 0.53 and 0.73 on identical settings. Do not claim a difference between two
  rows unless it is larger than that spread; there are only 4 known violations,
  so recall moves in steps of 0.25.
- When done, merge the JSON files into `eval_data/results/audit_eval.json`
  (same shape) and run `python tools/build_comparison_report.py`.

### Deliberately skipped (and why)

- **Step 10, merging LLM-proposed labels.** It needs ~68 passages x 2 stages of
  ~5K-token prompts (~700K tokens, far beyond the Groq cap; the local fallback
  `llama3.1:8b` is not installed). It would also add secondary labels to the
  frozen gold set, shifting nDCG for every system and invalidating comparisons
  already made. The cost and the risk outweigh the benefit.
- **Fixing `CER.2.19`'s passage boundary.** Its text is swallowed by the start of
  chapter 3, which dilutes the query. Fixing it means changing the frozen parser
  and therefore the gold set. Documented, not changed.

### 1c. RESULT (2026-09-25): a section-title query adopted under a pre-registered rule

Rule fixed **before** the run: adopt a query variant only if scope recall rises by
at least 0.02 absolute and nDCG does not fall. All 75 gold passages, k = 5,
`eval_data/results/v2_query_ablation.json`:

| System | Scope recall | nDCG | MRR | Context precision |
|---|---|---|---|---|
| v2 dense | 0.138 | 0.110 | 0.174 | 0.099 |
| v2 dense + title | 0.213 | 0.155 | 0.220 | 0.141 |
| v2 dense + first 100 words | 0.167 | 0.118 | 0.190 | 0.091 |
| v2 rerank (shipped) | 0.200 | 0.150 | 0.228 | 0.144 |
| **v2 rerank + title** | **0.227** | **0.211** | **0.280** | **0.213** |
| v2 rerank + first 100 words | 0.173 | 0.125 | 0.208 | 0.112 |

**Title prefix passes** (+0.027 scope recall, nDCG 0.150 -> 0.211) and is adopted:
`AuditConfig.title_in_query`, defined once in `retrieval_query()` so the query
that was measured is the query that ships; the API turns it on. **First-100-words
fails** (0.173 < 0.200) and is not adopted. The "mushy average vector" hypothesis
from §1 was tested directly and did not hold for this fix. Default stays `False`
in `AuditConfig` so earlier audit results remain reproducible. Recall at a
matched 800-word budget: 0.111 -> 0.148.

### 1d. Other changes made this session

- **`document_context` prompt option** (`DOCUMENT_CONTEXT` in `audit/pipeline.py`).
  Tells the model it is reading one section of a longer CER and that a CER is not
  a label or IFU. Most surviving false positives apply label/IFU requirements
  (Annex I §23, Annex VI) to a CER section. **Off by default. Written after
  reading the model's errors on the evaluation set, so any score with it is a
  development-set score and must be reported as such.** A test forbids naming a
  clause or passage in it. Its effect on recall is unmeasured; it may suppress
  true findings.
- **Truncated-reply salvage** (`llm/base.py::_salvage_findings`). A reply cut off
  at the token limit used to lose the whole passage (3 of 19 in a local run).
  Complete findings before the break are now kept and the reply is tagged
  `_salvaged`; a reply with nothing recoverable still raises.
- **Fail-open is now visible.** `AuditPipeline.judge_errors` counts judge and
  reconciler failures; `run_audit_eval.py` marks such runs `CONTAMINATED`.
- **Eval tooling:** `--repeats` (mean and range), `--seed`, `--judge`,
  `--local-model`, `--precompute-only`, a retrieval cache
  (`eval_data/.retrieval_cache.json`, verified identical to a real run on 19/19
  passages), results saved after every config, and a cumulative ablation chain
  `v1_retrieval -> v2_no_guards -> v2_full -> v2_title -> v2_reconcile -> v2_context`,
  one variable per row.

### 1e. Groq vs local, decided

Local `qwen3:4b` as generator works end to end (`v2_full`: 18 findings, recall
0.25, FP 0.73, ~650 s, no rate limit) but is **not** used for reported numbers:
it is a different and much weaker model, and Ollama's default 4,096-token context
is smaller than the ~4.6K-token audit prompt, so prompts may be silently
truncated. It is kept as a **free dry run** that catches code bugs before scarce
Groq tokens are spent (it found the truncated-JSON bug). Machine note: a loaded
`qwen3:4b` holds 4-8 GB of RAM depending on prompt length; on this 15.7 GB laptop
the memory guard stops runs unless Chrome and other heavy apps are closed. Run
`ollama stop qwen3:4b` before any job that loads the embedder. Keep the guard.

### 1f. Bug found by the browser check (fixed)

`api/deps.py` ignored `QDRANT_PATH` and always connected to a Qdrant **server**
on `localhost:6333`, although `.env.example`, `docker-compose.yml` and every
evaluation tool use the embedded index. On the documented no-Docker setup the UI
header said "vector store down" and search/audit could not work. It now defaults
to the embedded `qdrant_local` (empty `QDRANT_PATH` = server, as docker-compose
sets). Side effect: the two integration tests that used to skip ("needs a live
service") now run and pass, hence 198 passing and 0 skipped. Note the embedded
index takes an exclusive lock, so the API and an evaluation script cannot run at
the same time.

### 1g. Report generator no longer overwrites hand-written analysis

`docs/COMPARISON.md` holds hand-written sections (the audit failure breakdown and the
false-positive discussion) that `tools/build_comparison_report.py` never generated;
regenerating over it deleted 46 lines. The script now writes
`docs/COMPARISON.generated.md` (git-ignored) and leaves `COMPARISON.md` alone unless
given `--overwrite`. After the Groq runs, diff the two files and merge the new audit
table by hand.

### Older list (kept for reference; see the status board above)

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

### 1a. RESULT (2026-09-25): the cap experiment is done — keep 250

| | Cap 250 | Cap 600 |
|---|---|---|
| Findings | 19 | 15 |
| Recall | 0.25 (1/4) | 0.00 (0/4) |
| FP rate | 0.733 | 0.733 |
| Runtime | 628 s | 1,508 s |

Raising the cap did not recover any finding and made the run 2.4x slower, so
`MAX_CLAUSE_WORDS_IN_PROMPT` stays at 250. **Caveat:** with only 4 expected
violations, 1/4 vs 0/4 is one finding — within noise. The same v2_full/250
config gave FP rate 0.533 earlier today and 0.733 in the re-run, so **a single
audit run is not a stable measurement**. Any future audit comparison needs
repeated runs (or temperature 0) before a difference is claimed. Files:
`eval_data/results/audit_cap250.json`, `audit_cap600.json`.

### 1b. RESULT (2026-09-25): document-level reconciliation — built, measured, modest

`src/auditor/audit/reconcile.py`. For a finding that asserts an *absence*
("does not specify...", "lacks...", "no evidence..."), the rest of the CER is
searched (dense index over all 75 passages) and a second model is asked whether
another section already satisfies the requirement. The verdict counts **only**
if the model names a section it was shown *and* quotes text verifiably present
in it; any error keeps the finding. 14 tests in `tests/test_reconcile.py`.

Measured **offline on three frozen first-pass runs** (`tools/eval_reconcile_offline.py`,
`eval_data/results/reconcile_offline.json`). Replaying the filter over stored
output holds the generator constant, which end-to-end A/B cannot: the audit is
not bit-deterministic, and Groq's free tier caps `gpt-oss-120b` at 200,000
tokens/day (one audit run costs about half), so repeated end-to-end runs were
not possible.

| Stored run | Findings | FP rate | Precision | Recall |
|---|---|---|---|---|
| audit_eval (250-word cap) | 15 → 12 | 0.53 → 0.47 | 0.067 → 0.083 | 0.25 → 0.25 |
| audit_cap250 | 19 → 16 | 0.73 → 0.60 | 0.053 → 0.062 | 0.25 → 0.25 |
| audit_cap600 | 15 → 14 | 0.73 → 0.67 | 0.000 → 0.000 | 0.00 → 0.00 |

Consistent direction, **no true positive lost in any run**, but the effect is
small (1-3 findings per run). It did **not** bring the FP rate near the 0.2 the
first estimate hoped for. Reading the surviving false positives explains why:
most are not "answered elsewhere" at all — they apply requirements about a
*label or instructions for use* (Annex I §23, Annex VI) to a section of a CER,
which is not a label. That is a scope error, not missing context, and is
addressed separately by the `document_context` prompt option (§1c).

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
