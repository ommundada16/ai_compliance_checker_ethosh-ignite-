"""One interface over every LLM backend.

The pipeline never imports a vendor SDK. It asks a JSONProvider for structured
output and gets it, whether that came from a 4B model on the local GPU or a
120B model in someone else's datacentre.

This is not architecture for its own sake. It is what makes three things
possible that otherwise are not:

  * running the eval loop offline and for free, then switching one setting to
    get a stronger model's answer on the same inputs
  * failing over automatically when a free tier's per-minute budget runs out,
    instead of losing a 40-minute run
  * using a DIFFERENT model to judge than to generate, which the grounding
    guardrail in Phase 7 requires -- a model grading its own output measures
    self-consistency, not correctness

Every provider returns parsed JSON or raises. Callers should not be parsing
model prose, and a provider that cannot produce valid JSON is a broken
provider, not the caller's problem.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class RateLimited(Exception):
    """The backend refused because a quota is exhausted.

    Distinct from a generic failure so a failover chain can tell "try someone
    else" apart from "this request is malformed and will fail everywhere".
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RequestTooLarge(Exception):
    """The prompt alone exceeds the backend's limit.

    Retrying is pointless and failover probably will not help either; the
    caller has to send less.
    """


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class JSONResult:
    data: dict
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_payload(raw: str) -> dict:
    """Recover a JSON object from a model's reply.

    Three layers of defence, each for a failure seen in practice rather than
    imagined:

      * reasoning models (qwen3, gpt-oss) can emit a <think> block before the
        answer even when JSON mode is on
      * smaller models wrap output in markdown fences despite instructions
      * some prepend a sentence before the object

    If all three fail, raise rather than returning a half-parsed dict; a caller
    that silently accepts {} records a confident empty answer.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response body")

    text = _THINK.sub("", raw)
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            salvaged = _salvage_findings(text)
            if salvaged is not None:
                return salvaged
            raise
    raise ValueError(f"no JSON object in response: {raw[:200]!r}")


def _salvage_findings(text: str) -> dict | None:
    """Recover the complete findings from a reply that broke part-way through.

    A long answer cut off at the token limit, or a small model losing the thread
    on its eighth finding, used to cost the WHOLE passage: three of nineteen in
    a measured local run. The findings before the break are intact objects and
    are still evidence, so they are kept and the reply is marked `_salvaged` so
    that it stays visible in the results. Returns None when nothing complete can
    be recovered, and the caller then raises as before -- an unreadable reply is
    never quietly turned into "no findings".
    """
    anchor = text.find('"findings"')
    if anchor == -1:
        return None
    bracket = text.find("[", anchor)
    if bracket == -1:
        return None
    decoder = json.JSONDecoder()
    position, items = bracket + 1, []
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            obj, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        items.append(obj)
    return {"findings": items, "_salvaged": True} if items else None


class JSONProvider(ABC):
    """A backend that returns a JSON object for a prompt."""

    name: str = "provider"
    model: str = ""

    @abstractmethod
    def complete_json(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> JSONResult:
        """Return parsed JSON.

        Raises RateLimited when a quota is exhausted, RequestTooLarge when the
        prompt does not fit, and ValueError when the reply is not JSON.
        """

    def describe(self) -> str:
        return f"{self.name}:{self.model}"
