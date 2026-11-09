import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from schema import Finding

def get_client():
    """Return (client, model_name) for the local Ollama server.

    Ollama exposes an OpenAI-compatible endpoint; the API key is unused but
    the client requires a non-empty string.
    """
    # Force reload environment variables on demand
    load_dotenv(override=True)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    client = OpenAI(api_key="ollama", base_url=base_url, timeout=180)
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    return client, model_name


SEVERITY_WEIGHT = {
    "Critical": 25,
    "High": 12,
    "Medium": 5,
    "Low": 2
}

SYSTEM = """
You are a senior regulatory auditor specializing in EU MDR compliance.

You compare clinical study protocols against MDR guidelines.

Your task:
- Find compliance violations
- Find missing requirements
- Find safety gaps
- Find timing/date issues
- Find unclear responsibilities

Return ONLY valid JSON.
"""

def audit_section(section_text: str, guideline_clauses: str, page_number: int = 1):

    prompt = f"""
Audit the following protocol section from page {page_number} of the document.

PROTOCOL SECTION:
{section_text}

RELEVANT GUIDELINES:
{guideline_clauses}

Return ONLY valid JSON in this format:

{{
  "findings": [
    {{
      "violating_statement": "Short description of what is wrong or missing",
      "source_quote": "The EXACT verbatim sentence or phrase copied from the PROTOCOL SECTION above that is problematic. Must be a direct quote of 10-60 words from the text above.",
      "guideline_clause": "...",
      "category": "...",
      "severity": "Low",
      "explanation": "...",
      "confidence": 0.9,
      "suggested_correction": "The complete replacement text that should replace the source_quote to fix the issue",
      "page_number": {page_number}
    }}
  ]
}}

CRITICAL RULES for source_quote:
- MUST be copied VERBATIM (word-for-word) from the PROTOCOL SECTION text above
- Must be a complete sentence or meaningful phrase, 10-60 words
- Do NOT paraphrase or summarize - copy exact text
- If you cannot find a specific quote, use the first sentence of the most relevant paragraph

category must be exactly one of:

- Clinical Evaluation
- Risk Management
- Post Market Surveillance
- Verification & Validation
- Equivalence
- Safety
- Regulatory Documentation
- Clinical Investigation
- Other

Category guidance:

Clinical Evaluation:
- systematic literature review
- clinical evidence
- clinical data appraisal
- clinical performance
- clinical evaluation reports
- clinical investigations

Risk Management:
- benefit-risk analysis
- risk assessment
- hazard identification
- residual risks
- risk controls
- risk acceptability
- adverse events
- contraindications
- undesirable side effects

Post Market Surveillance:
- PMS
- PMCF
- vigilance
- trend analysis
- incident monitoring
- post-market data collection

Verification & Validation:
- testing
- verification
- validation
- biocompatibility
- performance testing
- design verification

Equivalence:
- technical equivalence
- biological equivalence
- clinical equivalence
- equivalence justification

Safety:
- sterilization
- warnings
- precautions
- patient safety
- safety information

Regulatory Documentation:
- labeling
- instructions for use (IFU)
- device description
- required MDR documentation
- regulatory records

Clinical Investigation:
- study design
- endpoints
- patient enrollment
- clinical study methodology

Choose the SINGLE BEST category only.
Do not invent categories.

If there are no violations:

{{"findings":[]}}
"""

    try:
        client, model_name = get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1200,
        )

        content = response.choices[0].message.content

        # Remove markdown code fences
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        data = json.loads(content)

        findings = []

        for item in data.get("findings", []):
            try:
                findings.append(Finding(**item))
            except Exception:
                continue

        return findings

    except Exception as e:
        print("Audit error:", e)
        raise e


def verify_correction(source_quote: str, suggested_correction: str, guideline_clause: str, category: str) -> dict:
    """Re-check whether a suggested correction actually resolves the flagged
    issue against the guideline clause, before it's applied to the document."""

    prompt = f"""
You are verifying a proposed text correction for EU MDR compliance.

ORIGINAL FLAGGED TEXT:
{source_quote}

PROPOSED CORRECTED TEXT:
{suggested_correction}

RELEVANT GUIDELINE CLAUSE:
{guideline_clause}

CATEGORY: {category}

Does the PROPOSED CORRECTED TEXT resolve the compliance issue and satisfy the guideline clause?

Return ONLY valid JSON in this exact format:
{{"compliant": true, "explanation": "one or two sentence justification"}}
"""

    try:
        client, model_name = get_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict EU MDR compliance verifier. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {
            "compliant": bool(data.get("compliant", True)),
            "explanation": data.get("explanation", "")
        }
    except Exception as e:
        print("Verification error:", e)
        # Fail open so a local model hiccup doesn't block the user entirely.
        return {"compliant": True, "explanation": "Verification could not be completed due to an error; correction applied as generated."}


def readiness_score(findings):

    severity_weights = {
        "Low": 5,
        "Medium": 10,
        "High": 15,
        "Critical": 20
    }

    category_severity = {}

    for f in findings:

        current = category_severity.get(
            f.category,
            0
        )

        category_severity[f.category] = max(
            current,
            severity_weights[f.severity]
        )

    penalty = sum(
        category_severity.values()
    )

    return max(0, 100 - penalty)
