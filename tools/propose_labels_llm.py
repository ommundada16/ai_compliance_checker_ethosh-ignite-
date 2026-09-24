"""Phase 1c -- independent LLM cross-check of the expert clause mapping.

Output: eval_data/llm_proposals.jsonl, folded in by build_gold_retrieval.py.

Why this exists
---------------
The expert map in tools/mdr_section_map.py is one source of judgement. A second,
independent opinion catches clauses it missed, and disagreements are worth more
than agreements: they mark exactly where the ground truth is debatable.

Why it is not circular
----------------------
The model NEVER sees retrieval output. It chooses from a COMPLETE ENUMERATION
of the regulation's structure:

    stage 1   passage + all 123 Article titles + all 17 Annex titles
              -> which instruments govern this passage?
    stage 2   passage + every clause of the chosen instruments
              -> which specific clauses?

Selecting from a full list cannot smuggle in a retriever's ranking, so using
these labels to score a retriever does not measure that retriever against
itself. This is the whole reason the two-stage shape exists rather than just
embedding the corpus and asking the model to confirm the top hits.

What the model is NOT allowed to do
-----------------------------------
Proposals enter the gold set at grade 1 (secondary) only. The model can
corroborate an expert primary label, but it can never create one. A test
enforces that. The human rubric stays the ground truth; this is a second
reader, not a replacement.

Usage:
    python tools/propose_labels_llm.py
    python tools/propose_labels_llm.py --limit 5        # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "You are a regulatory affairs specialist auditing a Clinical Evaluation Report "
    "against EU MDR 2017/745. You answer only with valid JSON. You never invent "
    "identifiers: every identifier you return must appear verbatim in the list you "
    "were given."
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_instrument_catalogue(clauses: list[dict]) -> tuple[list[str], dict[str, list[dict]]]:
    """A one-line-per-instrument catalogue, plus clauses grouped by instrument.

    'Instrument' means one Article or one Annex. There are 140 of them, which
    fits comfortably in a prompt -- that is what makes selection over a complete
    enumeration practical here.
    """
    by_instrument: dict[str, list[dict]] = {}
    titles: dict[str, str] = {}
    for c in clauses:
        if c["article"]:
            key = f"Art.{c['article']}"
            titles.setdefault(key, c["article_title"] or "")
        elif c["annex"]:
            key = f"Annex.{c['annex']}"
            titles.setdefault(key, c["annex_title"] or "")
        else:
            continue
        by_instrument.setdefault(key, []).append(c)

    def sort_key(k: str) -> tuple:
        if k.startswith("Art."):
            return (0, int(k.split(".")[1]))
        roman = k.split(".")[1]
        order = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX",
                 "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII"]
        return (1, order.index(roman) if roman in order else 99)

    catalogue = [f"{k} - {titles[k]}" for k in sorted(titles, key=sort_key)]
    return catalogue, by_instrument


def call_json(client, model: str, prompt: str, effort: str, max_tokens: int = 2048) -> dict:
    """One JSON call, with backoff for free-tier rate limits.

    max_tokens has to cover BOTH the model's internal reasoning and its visible
    output. gpt-oss models reason first; too small a budget is spent entirely on
    reasoning and JSON mode then fails with an empty body, which looks like a
    prompt bug and is not.
    """
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                reasoning_effort=effort,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last = exc
            if attempt == 4:
                break
            time.sleep((2 ** attempt) + random.random())
    raise RuntimeError(f"giving up after 5 attempts: {last}")


def stage1(client, model, effort, passage: dict, catalogue: list[str]) -> list[str]:
    prompt = f"""Below is a section of a Clinical Evaluation Report, followed by the COMPLETE list of
Articles and Annexes of EU MDR 2017/745.

CER SECTION {passage['section']} -- {passage['section_title']}
\"\"\"
{passage['text']}
\"\"\"

COMPLETE LIST OF INSTRUMENTS:
{chr(10).join(catalogue)}

Which instruments impose obligations that this section must satisfy?
Choose at most 4, most relevant first. Use identifiers exactly as written above.

Return: {{"instruments": ["Art.61", "Annex.XIV"]}}"""
    data = call_json(client, model, prompt, effort)
    return [str(x) for x in data.get("instruments", [])][:4]


def stage2(client, model, effort, passage: dict, instruments: list[str],
           by_instrument: dict[str, list[dict]]) -> list[str]:
    blocks: list[str] = []
    allowed: set[str] = set()
    for key in instruments:
        for c in by_instrument.get(key, []):
            allowed.add(c["clause_id"])
            text = c["text"] if len(c["text"]) <= 260 else c["text"][:260] + "..."
            blocks.append(f"[{c['clause_id']}] {text}")
    if not blocks:
        return []

    prompt = f"""CER SECTION {passage['section']} -- {passage['section_title']}
\"\"\"
{passage['text']}
\"\"\"

CANDIDATE CLAUSES (these are ALL the clauses of the instruments you selected):
{chr(10).join(blocks)}

Which specific clauses impose an obligation this section must satisfy?
Choose at most 5. Return clause IDs exactly as shown in [brackets].

Return: {{"clause_ids": ["Art.61.1"]}}"""
    data = call_json(client, model, prompt, effort, max_tokens=3072)
    # Hard filter: only IDs that were actually offered. The system prompt asks
    # for this, but a prompt is not a guarantee and a hallucinated ID would
    # otherwise enter the gold set.
    return [cid for cid in (str(x) for x in data.get("clause_ids", [])) if cid in allowed][:5]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N passages.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "eval_data" / "llm_proposals.jsonl")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("error: GROQ_API_KEY is not set in .env", file=sys.stderr)
        return 1
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    effort = os.getenv("GROQ_REASONING_EFFORT", "low")

    from groq import Groq

    client = Groq(api_key=api_key)

    clauses = load_jsonl(PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    passages = load_jsonl(PROJECT_ROOT / "eval_data" / "passages.jsonl")
    if args.limit:
        passages = passages[: args.limit]
    catalogue, by_instrument = build_instrument_catalogue(clauses)

    print(f"model      : {model} (reasoning_effort={effort})")
    print(f"catalogue  : {len(catalogue)} instruments over {len(clauses)} clauses")
    print(f"passages   : {len(passages)}")
    print()

    results: list[dict] = []
    started = time.time()
    for idx, passage in enumerate(passages, start=1):
        try:
            instruments = stage1(client, model, effort, passage, catalogue)
            clause_ids = stage2(client, model, effort, passage, instruments, by_instrument)
        except Exception as exc:  # noqa: BLE001 - one bad passage must not lose the run
            print(f"  [{idx}/{len(passages)}] {passage['passage_id']}: FAILED {exc}",
                  file=sys.stderr)
            results.append({"passage_id": passage["passage_id"], "instruments": [],
                            "clause_ids": [], "error": str(exc)})
            continue
        results.append({
            "passage_id": passage["passage_id"],
            "section": passage["section"],
            "instruments": instruments,
            "clause_ids": clause_ids,
        })
        print(f"  [{idx}/{len(passages)}] {passage['passage_id']:<14} "
              f"{','.join(instruments) or '-':<28} -> {len(clause_ids)} clauses")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if not r.get("error"))
    total = sum(len(r["clause_ids"]) for r in results)
    print()
    print(f"wrote {len(results)} proposals ({ok} ok) -> {args.out.relative_to(PROJECT_ROOT)}")
    print(f"  {total} clause proposals, mean {total / max(ok, 1):.1f} per passage")
    print(f"  elapsed {time.time() - started:.0f}s")
    print()
    print("next: python tools/build_gold_retrieval.py --merge eval_data/llm_proposals.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
