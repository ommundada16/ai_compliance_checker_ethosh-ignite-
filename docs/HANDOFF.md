# The Auditor — project handoff

A complete briefing for another engineer or AI assistant picking this up.
Everything here is verifiable from the repository; no figure is estimated.

**Repository:** https://github.com/ommundada16/ai_compliance_checker_ethosh-ignite-
**Status:** v2 complete and measured. 30 commits. ~160 tests passing, lint clean.
**Last updated:** 2026-09-24

---

## 1. What this project is

An AI tool that audits **Clinical Evaluation Reports** (CERs) for medical
devices against **EU MDR 2017/745**, using an LLM plus a RAG pipeline.

Two real documents drive everything:

| File | What it is | Size |
|---|---|---|
| `data/guideline.pdf` | EU MDR 2017/745, the full regulation | 175 pages, 101k words |
| `data/source_file.pdf` | A real CER for a Double J ureteral stent (BIORAD MEDISYS) | 91 pages |

### The insight that shapes the whole design

**This is not normal question-answering RAG, and treating it as such is why v1
failed.**

In ordinary RAG a short human question retrieves against a document corpus.
Here it is inverted:

- the **corpus** is the regulation (1,320 clauses)
- the **query** is a ~250-word section of the document being audited

Consequences that drive design decisions throughout:

- A long, topically-mixed query embeds to a mushy average vector, so chunking
  strategy has outsized impact.
- "Relevance" means *"this clause governs this passage"*, not *"this answers
  the question"* — a different labelling task.
- Exact identifiers (`Annex XIV`, `Article 61(4)`, `PMCF`) matter, because the
  literal string IS the meaning. That motivates the sparse retrieval arm.

---

## 2. v1 versus v2

| | v1 (baseline) | v2 (current) |
|---|---|---|
| Parsing | `text.split()` over concatenated pages | Layout-aware; tables rendered as key–value rows |
| Chunking | Fixed 800 words, 100 overlap | Clause-level units, hierarchical IDs (`Art.61.3.a`) |
| Embedding | `all-MiniLM-L6-v2` (256 word-pieces) | `BAAI/bge-base-en-v1.5` (512) |
| Sparse | none | BM25 (`Qdrant/bm25`) |
| Index | numpy matrix in RAM, rebuilt every run | Qdrant, persisted, named dense + sparse vectors |
| Retrieval | dense only, top-3 | hybrid → RRF fusion → cross-encoder rerank |
| Reranker | none | `BAAI/bge-reranker-base` |
| Grounding | first 40 chars, substring match | character spans: exact → whitespace → bounded fuzzy |
| Citations | free text | clause ID verified against what was retrieved |
| LLM | Ollama only, hardcoded | pluggable provider with Groq → Ollama failover |
| Evaluation | **none** | frozen gold set, three metric layers, per-change ablation |

**No torch anywhere in v2.** Embedding and reranking run on ONNX Runtime on
CPU, because the dev machine has 4 GB of VRAM and that belongs to the LLM.

---

## 3. THE NUMBERS

All from `eval_data/results/*.json`. Same frozen gold set, same metric code,
same 75 passages for every system.

### 3.1 Retrieval quality at k = 5

| Metric | v1 | v2 dense | v2 hybrid | **v2 rerank** | v1 → best |
|---|---|---|---|---|---|
| **Scope recall** | 0.113 | 0.138 | 0.107 | **0.200** | **+76%** |
| nDCG | 0.063 | 0.110 | 0.074 | **0.150** | **+137%** |
| MRR | 0.076 | 0.174 | 0.131 | **0.228** | **+198%** |
| MAP | 0.028 | 0.068 | 0.040 | **0.069** | **+147%** |
| Context precision | 0.054 | 0.099 | 0.061 | **0.144** | **+164%** |
| Hit rate | 0.187 | 0.267 | 0.213 | **0.387** | **+107%** |
| Context words sent to LLM | 4000 | 2398 | 3662 | **1456** | **−64%** |
| Latency per query | 17 ms | 339 ms | 353 ms | 14,061 ms | — |

**Scope recall is the headline metric**: did the retriever surface the
provisions that actually govern this passage.

### 3.2 Recall at a matched context budget — the fair comparison

Equal `k` is not equal information: v1 returns 800-word chunks, v2 returns
~66-word clauses. Comparing at equal **words delivered to the LLM**:

| System | 200w | 400w | 800w | 2400w | 4000w |
|---|---|---|---|---|---|
| v1 baseline | 0.019 | 0.019 | 0.019 | 0.035 | 0.063 |
| v2 dense | 0.034 | 0.057 | 0.058 | 0.067 | 0.125 |
| v2 hybrid | 0.006 | 0.027 | 0.034 | 0.042 | 0.065 |
| **v2 rerank** | **0.054** | **0.080** | **0.111** | **0.126** | **0.144** |

At 800 words: **0.111 vs 0.019 — 5.8×**.

### 3.3 The root cause of v1's failure (the most important finding)

v1 scored 2.9% recall on the first measurement. That looked like a harness bug,
so it was investigated before being reported. It was not a bug:

> `all-MiniLM-L6-v2` accepts 256 word-pieces. Regulatory English runs well over
> two pieces per word, so only **106 words of every 800-word chunk** were ever
> encoded — 13%. Silently. No error, no warning.

| Measurement | Value |
|---|---|
| Words embedded per 800-word chunk | **106** (13%) |
| Clauses ever embedded at all | **334 / 1320** |
| Gold labels reachable | 243 / 808 |
| **Hard recall ceiling** | **30.1%** — no value of `k` could beat it |
| v2 gold clauses truncated | **0** |

v1's problem was never ranking. **87% of the regulation it was auditing against
was invisible to its retriever.** `measure_embedding_window()` in
`src/auditor/baselines/v1_retriever.py` proves this by binary-searching the
shortest prefix whose embedding is *identical* to the full chunk's, and the
result is written into the baseline JSON so the claim travels with the numbers.

### 3.4 Audit quality — PENDING

`eval_data/results/audit_eval.json` currently holds output from an **invalid
run** (18 of 19 passages failed on rate limits before token pacing was added).
Do not quote it. A corrected 3-config run was in progress at handoff time.

When it completes it reports: recall over known violations, precision, F1,
**false-positive rate on clean passages**, and hallucination rate.

---

## 4. The evaluation methodology

This is the part worth understanding — it is what makes the numbers mean
anything.

### 4.1 Build the measuring stick first

The gold set and metric harness were built **before** any improvement, and v1
was scored with them. Without that, "v2 is better" is unfalsifiable.

### 4.2 The frozen gold set

| Artefact | Contents |
|---|---|
| `eval_data/clauses.jsonl` | 1,320 MDR clauses, hierarchical IDs, page spans |
| `eval_data/passages.jsonl` | 75 CER passages across 46 sections |
| `eval_data/gold_retrieval.jsonl` | 159 primary + 136 secondary graded labels |
| `tools/mdr_section_map.py` | Expert CER-section → MDR-clause mapping, with rationale per entry |
| `tools/audit_expectations.py` | 4 authored violations + 15 clean passages |

**Graded relevance, not binary.** Primary (grade 2) is the clause a regulator
cites first; missing it is a real failure. Secondary (grade 1) legitimately
bears on the section but its absence is not. Recall counts primary only; nDCG
uses both. This stops a retriever scoring well by dredging up loose context
while missing the governing clause.

**Labels name a SCOPE, not always a leaf.** A retrieved clause satisfies label
`L` if its ID is `L` or sits beneath it. `Annex.VIII` is satisfied by
`Annex.VIII.5`; `Art.61.1` has no children and still matches only itself.

**Non-circular labelling.** The expert map is hand-authored from the
regulation. A second opinion comes from an LLM that selects from a COMPLETE
ENUMERATION of the regulation's structure (all 123 Article titles + 17 Annex
titles, then all clauses of the chosen instruments) — it never sees retrieval
output, so labels cannot smuggle in a retriever's ranking. The model may only
add secondary labels; it can never create a primary one. A test enforces this.

### 4.3 Three metric layers

| Layer | Metrics | Question |
|---|---|---|
| Retrieval | Scope recall, nDCG, MRR, MAP, context precision, hit rate | Did the right clauses come back, high up? |
| Audit | Recall, precision, F1, **FP rate**, hallucination rate | Were the right violations reported, and only those? |
| Systems | latency, context words, index time | Is it usable and affordable? |

### 4.4 Fairness rules

- Identical frozen gold set for every system, built before v2 existed
- Same metric code, same temperature 0, same seed
- Exactly **one** variable changes per ablation row
- v1's 800-word chunks are mapped to the clauses they contain — deliberately
  **generous to v1**, making the v2 delta a conservative claim
- Comparison reported at matched context budget, not just matched `k`

---

## 5. Architecture

```
  PDF
   |
   +-- PARSE      layout-aware; tables rendered "header: value"
   +-- CHUNK      clause-level units, hierarchical IDs
   +-- EMBED      bge-base dense + BM25 sparse, ONNX on CPU
   +-- INDEX      Qdrant (embedded by default; server optional)
                    |
  Query = a CER section
   |
   +-- HYBRID     dense + sparse
   +-- FUSE       Reciprocal Rank Fusion (k=60)
   +-- RERANK     bge-reranker-base cross-encoder, top-25 -> top-5
   +-- AUDIT      LLM, must cite a clause ID it was shown
   +-- GUARDRAILS schema -> abstention -> citation -> grounding -> judge
   +-- API        FastAPI + SSE
   +-- UI         React + Vite + TypeScript
```

### Guardrails, cheapest first

| Guardrail | What it rejects |
|---|---|
| Schema | Invented categories/severities; missing quote |
| Abstention | Confidence below threshold |
| **Citation** | A clause that was never retrieved |
| Grounding | A quote that cannot be located in the passage |
| Judge | A finding a *different* model does not support |
| Injection | Instruction-like text inside the untrusted PDF |

The citation check matters most: the model cannot have read a clause it was
never shown, so a citation outside the retrieved set comes from training data
rather than the regulation in front of it. v1 could not detect this at all.

Every rejection records a **typed reason**. The distribution of reasons is
itself a measurement — it says which failure mode the model has, and whether a
guardrail removes more false positives than true ones.

---

## 6. Repository layout

```
src/auditor/
  parsing/mdr.py          MDR -> 1320 clauses with hierarchical IDs
  parsing/cer.py          CER -> 75 passages, table-aware
  embedding.py            dense + sparse, ONNX/CPU
  retrieval/qdrant_store.py   server or embedded index
  retrieval/fusion.py     RRF, unit-tested against hand-computed values
  retrieval/rerank.py     cross-encoder
  llm/base.py             JSONProvider interface, JSON recovery
  llm/providers.py        Groq (paced) / Ollama (native API) / failover
  audit/schema.py         enforced enums, mandatory quote
  audit/guardrails.py     span grounding, injection sanitising
  audit/pipeline.py       retrieve -> prompt -> parse -> guard -> judge
  evaluation/metrics.py   Recall@k, nDCG, MRR, MAP, context precision
  evaluation/gold.py      scope resolution
  evaluation/audit_metrics.py  recall, FP rate, hallucination
  baselines/              frozen v1, kept runnable
  resources.py            memory guard
api/                      FastAPI + SSE
frontend/                 React + Vite + TypeScript
tools/                    gold-set builders, scoring scripts
tests/                    ~160 tests
```

---

## 7. How to reproduce every number

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements-dev.txt
cp .env.example .env            # add GROQ_API_KEY

python tools/build_clause_corpus.py       # 1320 clauses
python tools/build_protocol_passages.py   # 75 passages
python tools/build_gold_retrieval.py      # graded labels
python tools/score_baseline_v1.py         # v1 numbers
python tools/score_v2.py --reindex        # v2 ablation
python tools/run_audit_eval.py --provider groq
python tools/build_comparison_report.py   # docs/COMPARISON.md
```

Qdrant runs **embedded** by default — no Docker required.
`docker-compose.yml` exists for when a real server is wanted.

CI rebuilds the gold set from the source PDFs on every push and **fails if a
single byte differs**. The artefacts are frozen but the parser is live code; a
cleaning tweak silently redefines what a clause *is* while the labels keep
pointing at old IDs. Nothing would break — the numbers would just stop meaning
what they did.

---

## 8. Known limitations — state these before anyone asks

- **The audit gold set is not exhaustive.** Four authored violations, fifteen
  clean passages. Audit recall is a **lower bound**, not an estimate.
- **"Clean" means a reviewer would not expect a finding**, not "provably
  compliant".
- **Retrieval labels are an expert rubric**, cross-checked by an independent
  model but not verified by a regulatory professional.
- **Hybrid search made things worse** at low k (0.107 vs 0.138 scope recall at
  k=5) and is kept in the ablation *because* it is a negative result. BM25
  assumes short keyword queries; these are 250-word passages, so the sparse arm
  matches common legal vocabulary and injects noise that RRF then rewards for
  "cross-arm agreement".
- **Absolute numbers are low.** Selecting 2–3 governing provisions out of 1,320
  clauses into a top-5 is genuinely hard (random ≈ 0.4%). The **relative**
  improvement is the claim.
- **Reranking costs ~14 s/query on CPU.** Too slow for interactive use at that
  setting; needs batching or a smaller cross-encoder.
- **Groq free tier is 8,000 tokens/minute**, which makes a full audit run take
  ~40 minutes. This is the dominant cost of any end-to-end evaluation.

---

## 9. Bugs found and fixed — useful context

Each was found by checking output, not by a test failing:

1. **MiniLM truncation** — the finding above. 87% of the corpus unindexed.
2. **Gold labels at the wrong granularity** — `Annex.VIII.1` was written
   meaning "Annex VIII"; that ID is actually "DURATION OF USE". The retriever
   was returning the correct classification rule and scoring zero. Fixed with
   scope-based matching applied identically to v1 and v2.
3. **Clause locator scattering** — accepting the first candidate above a
   threshold let a 9-token clause match a sparse 34-token scatter. Now the
   tightest span wins and a ceiling rejects stretched matches.
4. **Flattened tables** — 17 of 68 gold queries were column-interleaved
   garbage. Fixed by rendering tables as key–value rows; +16% more text
   captured and all retrieval metrics rose.
5. **Duplicate section numbers in the source CER** — the document numbers two
   different sections `4.5.1`. Later occurrences now take a `~2` suffix.
6. **Ollama's OpenAI shim drops `think:False`** — qwen3 burned 2,000 tokens on
   a hidden reasoning block and returned empty content. Now on the native API.
7. **Rate limiting in the wrong layer** — the token pacer lived in one tool, so
   an audit run fired 19 requests back to back and lost 18 to HTTP 429. Pacing
   now lives inside `GroqProvider`.
8. **API returned scores that contradicted its own ordering** — with reranking
   off, ordering came from RRF but the response carried raw dense scores.

---

## 10. What is NOT done

- **Audit-quality numbers** — run was in progress at handoff; the committed
  `audit_eval.json` is from an invalid run and must not be quoted.
- **`docs/COMPARISON.md`** — generator exists (`tools/build_comparison_report.py`),
  needs the audit numbers.
- **Browser verification of the UI** — code typechecks and builds; not yet
  screenshotted running against the live API.
- **LLM label cross-check not merged** — `tools/propose_labels_llm.py` works
  and is rate-limit-safe, but its output has not been folded into the gold set.
- **Reranker latency** — 14 s/query needs addressing before interactive use.
- **No deployment** — Terraform/Ansible were deliberately dropped from scope.

---

## 11. If you are an AI assistant picking this up

Context you need that is not obvious from the code:

- **The user is a final-year Computer Engineering student** building this for a
  resume/portfolio. Interview-defensibility matters more than raw scores.
- **Their machine is 15.7 GB RAM / 4 GB VRAM.** Running three ML jobs at once
  exhausted it, sent Windows into swap, and froze the desktop. **Never run more
  than one LLM or ML job at a time**, and verify the previous one exited before
  starting another. Prefer Groq (remote, ~0 local RAM) over local Ollama.
- `llama3.1:8b` is 4.92 GB and does **not** fit in 4 GB of VRAM; it spills into
  RAM and adds ~3.5 GB.
- Evaluation scripts print a resource plan and abort below a memory floor. Keep
  that behaviour.
- **The methodology is the product.** Preserve: measuring before changing,
  one variable per ablation row, keeping negative results, and stating
  limitations up front. Do not "improve" a number by loosening the gold set.
