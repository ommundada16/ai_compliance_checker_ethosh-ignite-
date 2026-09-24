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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auditor.llm.base import JSONProvider, RequestTooLarge  # noqa: E402

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


def build_provider(pacer_tpm: int) -> JSONProvider:
    """Groq first, local Ollama as the floor.

    The first full run lost 39 of 68 passages to Groq's 8000 tokens-per-minute
    free-tier ceiling. Two independent fixes, because either alone is
    insufficient:

      pacing    a rolling token budget, so requests wait for room BEFORE being
                sent. Reactive backoff cannot fix a per-MINUTE limit -- an
                exponential retry capped at ~16s just retries inside the same
                exhausted window, which is exactly how those 36 failures
                happened.
      failover  when Groq is benched anyway, the run continues on a local model
                instead of dying. Slower, but a batch that finishes on llama3.1
                beats one that stops at 57%.

    Using a local model for these labels is not circular. The system under test
    is an embedding retriever, not an LLM, so an LLM's opinion about which
    clause governs a passage is independent evidence either way. (That argument
    would NOT hold for audit-quality labels, where the thing being graded is
    itself an LLM.)
    """
    from auditor.llm.providers import FailoverProvider, GroqProvider, OllamaProvider

    chain: list = []
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        chain.append(
            GroqProvider(
                api_key,
                os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                os.getenv("GROQ_REASONING_EFFORT", "low"),
            )
        )
    chain.append(OllamaProvider(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b")))
    return FailoverProvider(chain)


class TokenPacer:
    """Rolling 60-second token budget for the metered provider."""

    def __init__(self, tokens_per_minute: int = 8000) -> None:
        self.budget = tokens_per_minute
        self.spent: list[tuple[float, int]] = []

    def _prune(self, now: float) -> None:
        self.spent = [(t, n) for t, n in self.spent if now - t < 60.0]

    def wait_for(self, tokens: int) -> None:
        while True:
            now = time.time()
            self._prune(now)
            if sum(n for _, n in self.spent) + tokens <= self.budget or not self.spent:
                return
            time.sleep(max(0.5, 60.0 - (now - self.spent[0][0]) + 0.5))

    def record(self, tokens: int) -> None:
        self.spent.append((time.time(), tokens))


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic: legal English runs dense, so ~3 chars/token.
    Overestimating costs throughput; underestimating costs a 429."""
    return len(text) // 3 + 256


def call_json(provider, prompt: str, pacer: TokenPacer, max_tokens: int = 1024) -> dict:
    estimated = estimate_tokens(prompt) + max_tokens
    last: Exception | None = None
    for attempt in range(4):
        pacer.wait_for(estimated)
        try:
            result = provider.complete_json(SYSTEM, prompt, max_tokens=max_tokens)
            pacer.record(result.usage.total_tokens or estimated)
            return result.data
        except RequestTooLarge as exc:
            # Retrying or failing over will not make the prompt smaller.
            raise RuntimeError(f"request too large: {str(exc)[:160]}") from exc
        except Exception as exc:  # noqa: BLE001
            last = exc
            pacer.record(estimated)
            if attempt == 3:
                break
            time.sleep(min(70.0, 20.0 * (attempt + 1)) + random.random())
    raise RuntimeError(f"giving up after 4 attempts: {last}")


def stage1(provider, pacer, passage: dict, catalogue: list[str]) -> list[str]:
    prompt = f"""Below is a section of a Clinical Evaluation Report, followed by the COMPLETE list of
Articles and Annexes of EU MDR 2017/745.

CER SECTION {passage['section']} -- {passage['section_title']}
\"\"\"
{passage['text']}
\"\"\"

COMPLETE LIST OF INSTRUMENTS:
{chr(10).join(catalogue)}

Which instruments impose obligations that this section must satisfy?
Choose at most 3, most relevant first. Use identifiers exactly as written above.

Return: {{"instruments": ["Art.61", "Annex.XIV"]}}"""
    data = call_json(provider, prompt, pacer)
    return [str(x) for x in data.get("instruments", [])][:3]


def stage2(provider, pacer, passage: dict, instruments: list[str],
           by_instrument: dict[str, list[dict]]) -> list[str]:
    blocks: list[str] = []
    allowed: set[str] = set()
    for key in instruments:
        for c in by_instrument.get(key, []):
            allowed.add(c["clause_id"])
            text = c["text"] if len(c["text"]) <= 180 else c["text"][:180] + "..."
            blocks.append(f"[{c['clause_id']}] {text}")
    # Hard cap. Annex XIV alone carries 26 clauses and Annex I s23 carries 59;
    # unbounded, a single stage-2 prompt reached 10288 tokens against an 8000
    # TPM ceiling and could never succeed no matter how often it was retried.
    blocks = blocks[:45]
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
    data = call_json(provider, prompt, pacer, max_tokens=1536)
    # Hard filter: only IDs that were actually offered. The system prompt asks
    # for this, but a prompt is not a guarantee and a hallucinated ID would
    # otherwise enter the gold set.
    return [cid for cid in (str(x) for x in data.get("clause_ids", [])) if cid in allowed][:5]


def rel(path: Path) -> str:
    """Display path, tolerant of outputs written outside the repo.

    Path.relative_to raises when the target is not under the base, which
    crashed a completed run at the final print line.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N passages.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "eval_data" / "llm_proposals.jsonl")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)
    provider = build_provider(int(os.getenv("GROQ_TPM", "8000")))

    clauses = load_jsonl(PROJECT_ROOT / "eval_data" / "clauses.jsonl")
    passages = load_jsonl(PROJECT_ROOT / "eval_data" / "passages.jsonl")
    if args.limit:
        passages = passages[: args.limit]
    catalogue, by_instrument = build_instrument_catalogue(clauses)

    print(f"providers  : {provider.describe()}")
    print(f"catalogue  : {len(catalogue)} instruments over {len(clauses)} clauses")
    print(f"passages   : {len(passages)}")
    print()

    pacer = TokenPacer(int(os.getenv('GROQ_TPM', '8000')))
    results: list[dict] = []
    started = time.time()
    for idx, passage in enumerate(passages, start=1):
        try:
            instruments = stage1(provider, pacer, passage, catalogue)
            clause_ids = stage2(provider, pacer, passage, instruments, by_instrument)
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
    print(f"served by  : {getattr(provider, 'counts', {})}")
    print(f"wrote {len(results)} proposals ({ok} ok) -> {rel(args.out)}")
    print(f"  {total} clause proposals, mean {total / max(ok, 1):.1f} per passage")
    print(f"  elapsed {time.time() - started:.0f}s")
    print()
    print("next: python tools/build_gold_retrieval.py --merge eval_data/llm_proposals.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
