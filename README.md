# The Auditor — EU MDR Compliance Checker

The Auditor is an AI-powered compliance tool that checks Medical Device Clinical Evaluation Reports (CER) against the EU MDR (Regulation 2017/745) guidelines. It is designed to help medical device manufacturers and regulatory auditors quickly identify compliance gaps, missing requirements, and safety issues — without manually going through hundreds of pages of regulations.

---

## What It Does

When you upload a Clinical Evaluation Report (PDF), the tool reads through it, compares each section against the official EU MDR guidelines, and highlights exactly where the document falls short. Every issue is tied to a specific regulation clause, given a severity rating, and comes with a suggested fix.

At the end, you get a readiness score out of 100 and a downloadable HTML report summarizing all findings.

---

## Key Features

- Checks your document against real EU MDR regulatory clauses
- Groups issues into categories like Risk Management, Clinical Evaluation, Post-Market Surveillance, and more
- Scores the document from 0 to 100 based on how many gaps were found
- Shows AI confidence level for each finding
- Generates a clean, downloadable audit report in HTML format
- Simple, modern web interface — no technical knowledge required to use

---

## How It Works

1. You upload a PDF Clinical Evaluation Report through the web interface and pick how many pages to audit (5-10).
2. The app splits the document into readable sections, page by page, and skips pages that look like a cover/table of contents/abbreviations list (no LLM call is wasted on those).
3. For each remaining page, it finds the most relevant EU MDR guideline clauses.
4. A local AI model, run through [Ollama](https://ollama.com), analyzes the page and identifies any violations or missing items.
5. Any finding whose flagged text can't actually be located on that page is dropped rather than shown as an unanchored "phantom" issue.
6. Results are highlighted directly in the document text, color-coded by severity. Expanding a highlighted issue shows the suggested fix; applying it re-verifies the correction against the guideline clause, then rewrites that part of the document text.

---

## Setup Instructions

These steps assume you have Python installed on your machine (version 3.9 or higher recommended).

**Step 1 — Clone the repository**

If you haven't already, download the project to your local machine:

```bash
git clone https://github.com/AyanMujawar/Invictus_NLP.git
cd Invictus_NLP
```

**Step 2 — Create a virtual environment**

This keeps the project's dependencies isolated from your system Python:

```bash
python -m venv venv
venv\Scripts\activate
```

On Mac or Linux, use `source venv/bin/activate` instead.

**Step 3 — Install the required libraries**

```bash
pip install -r requirements.txt
```

**Step 4 — Install Ollama and pull a model**

This project runs its AI model locally through [Ollama](https://ollama.com) — no API key, no rate limit, no per-token cost, nothing sent over the network.

1. Download and install Ollama for your OS from [ollama.com/download](https://ollama.com/download).
2. Start it (on Windows/Mac it runs as a background app after install; on Linux run `ollama serve`).
3. Pull the default model:
   ```bash
   ollama pull llama3.2:3b
   ```
   This 3B model is the default because it's noticeably faster per page than 8B-class models on a single consumer GPU/laptop, at some cost to finding quality. If you have a strong GPU (8GB+ VRAM) and want better accuracy and don't mind slower audits, you can use an 8B-class model instead:
   ```bash
   ollama pull llama3.1:8b
   ```
4. Create a file named `.env` in the root of the project folder:
   ```
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_BASE_URL=http://localhost:11434/v1
   ```
   `OLLAMA_MODEL` must exactly match whichever model you pulled in step 3 (a `model not found` error means these are out of sync).

   If audits still feel slow, run `ollama ps` while an audit is running — it shows whether the model is using your GPU (`100% GPU`) or has fallen back to CPU. CPU-only inference is 5-10x slower; if you see CPU usage, check that your GPU drivers are up to date and that Ollama detected your GPU (`ollama list` and the Ollama app's logs will mention it).

**Step 5 — Run the application**

```bash
streamlit run app.py
```

Once the server starts, Streamlit will automatically open the app in your default browser. If it does not open on its own, look for the Local URL printed in your terminal (it will look like `http://localhost:8501`) and open it manually.

---

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Main web application and UI |
| `auditor.py` | AI auditing logic using a local Ollama model |
| `ingest.py` | PDF text extraction and chunking |
| `retriever.py` | Semantic search over guideline chunks |
| `report.py` | HTML report generation |
| `schema.py` | Data models for findings and reports |
| `data/guideline.pdf` | The EU MDR guideline document used as reference |

---

## Notes

- The `data/guideline.pdf` file must be present for the audit to run. It contains the reference EU MDR guidelines the AI compares against.
- The document view currently audits 5-10 pages at a time (chosen via a slider on the upload screen) rather than a full report, to keep local inference time reasonable.
- Front-matter pages (cover, table of contents, abbreviations/glossary, revision history) are detected automatically and skipped entirely -- they don't count toward your page slider at all. "N pages" always means N pages of real content, scanned from wherever it actually starts in the PDF (even if that's non-contiguous, e.g. an abbreviations page appears again later in the document).
- If you see a connection error during an audit, make sure the Ollama app/service is actually running and that you've pulled the model named in `OLLAMA_MODEL`.
- Only PDFs with selectable text (not scanned images) are supported. If your PDF is scanned, run it through an OCR tool first.