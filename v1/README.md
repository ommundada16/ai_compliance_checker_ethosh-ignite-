# The Auditor — v1 (original Streamlit app)

> **Status: frozen baseline.** This is the first version of the project. It is kept, unmodified in
> behaviour, because every v1-vs-v2 number is measured against it. For the current pipeline see
> [`../v2/`](../v2/README.md); for the measured comparison see [`../docs/COMPARISON.md`](../docs/COMPARISON.md).

The Auditor checks a medical-device **Clinical Evaluation Report (CER)** against the **EU MDR
(Regulation 2017/745)**. You upload a PDF; the app reads it page by page, finds the most relevant
guideline clauses for each page, asks a local LLM to point out compliance gaps, and highlights each
issue directly in the document text, colour-coded by severity. It ends with a readiness score out of
100 and a downloadable HTML report.

---

## How it works

1. Upload a CER (PDF) in the web UI and choose how many pages to audit (5–10).
2. `ingest.py` extracts text page by page and skips pages that look like a cover, table of contents
   or abbreviations list, so no LLM call is spent on them.
3. `retriever.py` embeds the guideline (`../data/guideline.pdf`, chunked into 800-word windows with
   100 words of overlap) with `all-MiniLM-L6-v2` and returns the 3 chunks most similar to each page.
4. `auditor.py` sends the page plus those chunks to a local model through [Ollama](https://ollama.com)
   and asks for findings as strict JSON, validated with the Pydantic models in `schema.py`.
5. A finding whose quoted text cannot be located on the page is dropped rather than shown as an
   unanchored "phantom" issue.
6. `app.py` highlights the surviving findings in the page text. Applying a suggested fix re-verifies
   it against the guideline clause (a second LLM call) and then rewrites that part of the text.
7. `report.py` builds the downloadable HTML report.

The readiness score starts at 100 and subtracts a penalty for the worst finding in each category
(Low 5, Medium 10, High 15, Critical 20).

---

## Setup

Python 3.9 or newer is recommended. **Use a virtual environment of its own** — v1 depends on
`sentence-transformers`, which pulls in PyTorch, while v2 deliberately avoids it.

**1. Get the code** and enter this folder:

```bash
git clone https://github.com/ommundada16/ai_compliance_checker_ethosh-ignite-.git
cd ai_compliance_checker_ethosh-ignite-/v1
```

**2. Create a virtual environment and install the pinned dependencies:**

```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**3. Install Ollama and pull a model.** The model runs locally: no API key, no per-token cost,
nothing leaves your machine.

1. Install Ollama from [ollama.com/download](https://ollama.com/download) and make sure it is running.
2. Pull the default model:
   ```bash
   ollama pull llama3.2:3b
   ```
   The 3B model is the default because it is much faster per page on a single consumer GPU or
   laptop, at some cost in finding quality. With 8 GB+ of VRAM, `ollama pull llama3.1:8b` is more
   accurate but slower.
3. Copy the environment template and make `OLLAMA_MODEL` match the model you pulled:
   ```bash
   copy .env.example .env      # Mac/Linux: cp .env.example .env
   ```
   A `model not found` error means these two are out of sync. While an audit runs, `ollama ps`
   shows whether the model is on the GPU (`100% GPU`) or has fallen back to the much slower CPU.

**4. Run the app** from this folder:

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (by default at `http://localhost:8501`).

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app and UI |
| `auditor.py` | LLM audit call, fix verification and readiness score (local Ollama) |
| `ingest.py` | PDF text extraction, front-matter detection, chunking |
| `retriever.py` | Embedding search over the guideline chunks |
| `report.py` | HTML report generation |
| `schema.py` | Pydantic models for findings and reports |
| `run_audit.py` | Headless version of the pipeline; writes `report.json` |
| `sample_report.json` | Example report used while building the UI |
| `requirements.txt` | Frozen v1 dependency pins |
| `.env.example` | Template for `OLLAMA_MODEL` and `OLLAMA_BASE_URL` |

The two input PDFs are shared with v2 and live in the repo-level `../data/` folder
(`guideline.pdf`, the EU MDR; `source_file.pdf`, a sample CER). Both must be present.

---

## Notes and known limits

- It audits 5–10 pages per run to keep local inference time reasonable, not the whole report.
- Only PDFs with selectable text are supported; scanned documents need OCR first.
- A connection error during an audit almost always means Ollama is not running or the model named
  in `OLLAMA_MODEL` has not been pulled.
- **Why v2 exists:** v1's embedding model reads only 256 word-pieces, so just the first ~106 words
  of every 800-word guideline chunk were ever searchable. Most of the regulation was invisible to
  retrieval. The write-up and the numbers are in the [repo README](../README.md) and
  [`../docs/COMPARISON.md`](../docs/COMPARISON.md).
