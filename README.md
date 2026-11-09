# AI Compliance Checker — EU MDR Auditor

Audits a medical-device **Clinical Evaluation Report (CER)** against **EU MDR 2017/745** with an LLM
and a retrieval (RAG) pipeline, and — the part that matters — **measures whether it actually works**.

The repo holds two generations of the same tool, side by side, so the improvement can be reproduced
rather than asserted:

| Folder | What it is | Stack | Status |
|---|---|---|---|
| [`v1/`](v1/README.md) | The original app. Upload a PDF, get findings highlighted in the text. | Streamlit, `sentence-transformers`, local Ollama | Frozen baseline |
| [`v2/`](v2/README.md) | The rebuilt pipeline plus an evaluation harness that scores it against v1. | Qdrant, ONNX embeddings and reranker, Groq/Ollama, FastAPI, React | Current |
| [`data/`](data/) | The two source documents both versions audit: the EU MDR (`guideline.pdf`) and a sample CER (`source_file.pdf`). | PDF | Shared |
| [`docs/`](docs/) | Cross-version write-ups: measured comparison, engineering handoff, upgrade plan. | Markdown, PDF | Shared |

```
ai_compliance_checker/
├─ v1/        original Streamlit app (app.py, auditor.py, ingest.py, retriever.py, ...)
├─ v2/        src/  api/  frontend/  tools/  tests/  eval_data/  Dockerfile  pyproject.toml
├─ data/      guideline.pdf  source_file.pdf
├─ docs/      COMPARISON.md  HANDOFF.md  Auditor_v2_Upgrade_Plan.pdf
└─ .github/   CI (lint + tests, gold-set integrity, frontend typecheck/build)
```

---

## The idea that shapes the design

Normal RAG answers a short question from a document corpus. Here it is inverted: the **corpus is the
regulation** (1,320 clauses) and the **query is a ~250-word section of the report being audited**.
That changes what chunking, retrieval and "relevant" mean, and most of v2's design follows from it.

## The result

Retrieval quality at k = 5, on one frozen gold set with identical metric code:

| Metric | v1 | v2 | change |
|---|---|---|---|
| Scope recall | 0.1133 | **0.2000** | +76% |
| nDCG | 0.0634 | **0.1503** | +137% |
| MRR | 0.0764 | **0.2278** | +198% |
| Context precision | 0.0544 | **0.1440** | +164% |
| Context words sent to the LLM | 4000 | **1456** | −64% |

Better retrieval from about a third of the context. Full tables, the ablation and the negative results
are in [`docs/COMPARISON.md`](docs/COMPARISON.md).

## The finding the project is about

v1 scored 2.9% recall, which looked like a bug in the harness. It was investigated before being
reported, and it was not one: `all-MiniLM-L6-v2` accepts only 256 word-pieces, so just the first ~106
words of every 800-word chunk were ever encoded. Silently, with no error. **334 of 1,320 clauses**
were ever embedded, which caps recall at **30.1%** whatever `k` is. v1's problem was never ranking:
87% of the regulation it audited against was invisible to its retriever. v2 truncates no gold clause.

## What this does not show

Stated up front, because these are the first questions a careful reader should ask:

- The audit gold set is small (4 authored violations, 15 clean passages), so audit recall is a lower
  bound, not an estimate.
- On it, the shipped configuration finds 1 of 4 known violations and flags 53–73% of clean passages
  (the same settings gave both figures on different runs). **Zero findings were hallucinated.**
- Retrieval labels are a hand-authored expert rubric, not verified by a regulatory professional.
- Hybrid (dense + keyword) search made results *worse* here and is kept as a documented negative result.

The complete list is in [`v2/README.md`](v2/README.md#what-this-does-not-show).

---

## Quick start

The two versions have **separate dependencies** (v1 needs PyTorch via `sentence-transformers`; v2
deliberately has none), so give each its own virtual environment and run commands from its folder.

**v2 (current)**

```bash
cd v2
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                 # add a GROQ_API_KEY, or use Ollama
python tools/score_v2.py --reindex                   # build the index and score retrieval
uvicorn api.main:app --reload                        # http://localhost:8000/docs
cd frontend && npm install && npm run dev
pytest                                               # 198 tests
```

**v1 (original app)**

```bash
cd v1
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2:3b
cp .env.example .env
streamlit run app.py
```

Step-by-step detail, including Docker and Ollama setup, is in each folder's README.

## Where to read next

| If you want to... | Read |
|---|---|
| Understand the architecture and guardrails | [`v2/README.md`](v2/README.md) |
| See the v1 → v2 numbers and ablations | [`docs/COMPARISON.md`](docs/COMPARISON.md) |
| Continue the work: state, failures, next steps, how to reproduce every number | [`docs/HANDOFF.md`](docs/HANDOFF.md) |
| Run or understand the original app | [`v1/README.md`](v1/README.md) |

## Continuous integration

`.github/workflows/ci.yml` runs three jobs on every push: lint and tests; **gold-set integrity**, which
rebuilds the frozen evaluation set from the source PDFs and fails if a single byte differs (the parser
is live code, so a cleaning tweak could silently change what a "clause" is while the labels keep
pointing at the old IDs); and a frontend typecheck and build.
