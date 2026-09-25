# The Auditor — v1 vs v2

_Generated 2026-09-25 by `tools/build_comparison_report.py`._

Every number below is read from `eval_data/results/*.json`, each of which
is produced by a script in `tools/`. Nothing here is typed by hand.

## What changed

|  | v1 | v2 |
|---|---|---|
| Parsing | `text.split()` over concatenated pages | Layout-aware, tables rendered as key–value rows |
| Chunking | Fixed 800 words, 100 overlap | Clause-level units with hierarchical IDs |
| Embedding | all-MiniLM-L6-v2 (256 word-pieces) | bge-base-en-v1.5 (512) |
| Index | numpy matrix, rebuilt every run | Qdrant, persisted, dense + sparse |
| Retrieval | Dense only, top-3 | Hybrid + RRF, cross-encoder rerank |
| Grounding | First 40 characters, substring | Character spans, exact → fuzzy |
| Citations | Free text | Clause ID verified against what was retrieved |
| Evaluation | None | Frozen gold set, 3 metric layers |

## Why v1 could not have worked

Measured, not argued. `measure_embedding_window()` binary-searches the
shortest prefix whose embedding is **identical** to the full chunk's:

|  |  |
|---|---|
| Words embedded per 800-word chunk | **106** |
| Fraction of each chunk encoded | **13%** |
| Clauses ever embedded | 334 / 1320 |
| Gold labels reachable | 243 / 808 |
| **Hard recall ceiling** | **30.1%** |

No value of `k` could beat that ceiling: the remaining clauses were never
in the index in any form. v2 truncates **zero** gold clauses.

## Retrieval quality (k = 5)

| Metric | v1 baseline | v2 dense | v2 hybrid | v2 rerank | v1 → best |
|---|---|---|---|---|---|
| Scope recall | 0.113 | 0.138 | 0.107 | 0.200 | **+76%** |
| nDCG | 0.063 | 0.110 | 0.074 | 0.150 | **+137%** |
| MRR | 0.076 | 0.174 | 0.131 | 0.228 | **+198%** |
| MAP | 0.028 | 0.068 | 0.040 | 0.069 | **+147%** |
| Context precision | 0.054 | 0.099 | 0.061 | 0.144 | **+164%** |
| Context words sent | 4000 | 2398 | 3662 | 1456 | **-64%** |

**Scope recall** is the headline: did the retriever surface the provisions
that actually govern the passage. Fewer context words is better — less
noise reaching the model, for fewer tokens.

### At a matched context budget

Equal `k` is not equal information: v1 returns 800-word chunks, v2 returns
~66-word clauses. This is the honest comparison.

| System | 200w | 400w | 800w | 2400w | 4000w |
|---|---|---|---|---|---|
| v1 baseline | 0.019 | 0.019 | 0.019 | 0.035 | 0.063 |
| v2 dense | 0.034 | 0.057 | 0.058 | 0.067 | 0.125 |
| v2 hybrid | 0.006 | 0.027 | 0.034 | 0.042 | 0.065 |
| v2 rerank | 0.054 | 0.080 | 0.111 | 0.126 | 0.144 |

## Audit quality

Recall counts violations **known** to be present, so it is a lower bound,
not an estimate. The false-positive rate is the share of clean passages
that drew at least one finding — the metric that decides whether a
reviewer can trust the output.

| System | Findings | Recall | Precision | F1 | FP rate | Hallucination |
|---|---|---|---|---|---|---|
| v2 full | 15 | 0.250 | 0.067 | 0.105 | 0.533 | 0.000 |

### Where the audit actually fails

Aggregate numbers say the system is weak; the per-passage breakdown says
exactly *why*, and the two failure modes need different fixes.

| Expected finding | Retrieval surfaced the clause? | Model reported it? | Verdict |
|---|---|---|---|
| `CER.4.3.2.1` → Art. 61(4) | yes (`Art.61.4`, rank 2) | yes | **found** |
| `CER.2.11` → Annex I s23 | **yes** (`Annex.I.23.i#1`) | **no** | LLM miss |
| `CER.2.19` → Annex II s1 | no | n/a | retrieval miss |
| `CER.4.3.2.2#1` → Art. 83 | no | n/a | retrieval miss |

So of three misses, **two are retrieval failures and one is a reasoning
failure**. That distinction is the entire value of measuring the two layers
separately: improving the prompt would not have recovered the two retrieval
misses, and improving retrieval would not have recovered the reasoning one.

### The false-positive rate is the real problem

**8 of 15 clean passages drew at least one finding.** Precision is 0.067. A
reviewer handed this output would spend most of their time dismissing it.

The cause is architectural, not a prompt defect. The pipeline audits each
passage **in isolation**, so the model is shown section 2.1 ("Identification of
device(s)", 54 words of administrative detail) together with Annex XIV s1(a),
which lists what a clinical evaluation plan must contain. It then correctly
observes that this section does not contain a clinical evaluation plan — and
reports it as a violation. The content is present, in section 4, which the
model never sees.

Three fixes follow directly, in order of expected value:

1. **Document-level reconciliation.** Before reporting "X is missing", check
   whether X appears anywhere else in the document. Most of these findings
   would disappear.
2. **Tell the model what kind of section it is reading.** An identification
   section cannot breach a clinical-evaluation-plan requirement.
3. **Raise the abstention threshold.** These findings carried confidence
   0.88–0.97, so confidence alone does not separate them — which is itself
   worth knowing, and is why the threshold is not the first fix.

The guardrails are working as designed: hallucination rate is **0.000**, and
they rejected 5 findings (2 citing clauses that were never retrieved, 3 that
the second model would not support). They catch fabrication. They cannot catch
a finding that is internally coherent and simply out of context.

## Cost and latency

| System | Mean query | Context words |
|---|---|---|
| v1 baseline | 17 ms | 4000 |
| v2 dense | 339 ms | 2398 |
| v2 hybrid | 353 ms | 3662 |
| v2 rerank | 14061 ms | 1456 |

The cross-encoder is the slowest stage by an order of magnitude, and buys
the largest quality gain. That asymmetry is why retrieval is two-stage: a
bi-encoder's precomputed index cheaply reduces 1320 clauses to ~25, and the
cross-encoder spends real compute only on those.

## What this does not show

- **The gold set is not exhaustive.** Four authored violations and fifteen
  clean passages. Audit recall is a lower bound.
- **Clean means 'a reviewer would not expect a finding'**, not 'provably
  compliant'.
- **Retrieval labels are an expert rubric**, cross-checked by an independent
  model but not verified by a regulatory professional.
- **Hybrid search made things worse** at low k and is kept in the ablation
  precisely because it is a negative result. BM25 assumes short keyword
  queries; these are 250-word passages, so the sparse arm matches common
  legal vocabulary and injects noise that fusion then rewards.
- **Absolute numbers are low.** Selecting 2–3 governing provisions from 1320
  clauses into a top-5 is hard. The relative improvement is the claim.

## Reproducing this

```bash
python tools/build_clause_corpus.py       # 1320 MDR clauses
python tools/build_protocol_passages.py   # 75 CER passages
python tools/build_gold_retrieval.py      # graded labels
python tools/score_baseline_v1.py         # v1 numbers
python tools/score_v2.py --reindex        # v2 ablation
python tools/run_audit_eval.py --provider groq
python tools/build_comparison_report.py   # this file
```

