"""JSON recovery and failover logic, with no network involved.

Every case here is a failure actually observed while wiring up the providers,
not an imagined one.
"""

from __future__ import annotations

import pytest

from auditor.llm.base import (
    JSONProvider,
    JSONResult,
    RateLimited,
    RequestTooLarge,
    Usage,
    parse_json_payload,
)
from auditor.llm.providers import FailoverProvider

# --- JSON recovery --------------------------------------------------------

def test_plain_json() -> None:
    assert parse_json_payload('{"article": 61}') == {"article": 61}


def test_strips_markdown_fences() -> None:
    """Smaller models fence their output despite being told not to."""
    assert parse_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_payload('```\n{"a": 1}\n```') == {"a": 1}


def test_strips_reasoning_block() -> None:
    """qwen3 and gpt-oss can emit <think> before the answer even in JSON mode."""
    raw = "<think>The user wants the article number. Article 61.</think>\n{\"article\": 61}"
    assert parse_json_payload(raw) == {"article": 61}


def test_recovers_object_from_surrounding_prose() -> None:
    raw = 'Here is the result:\n{"clause_ids": ["Art.61.1"]}\nHope that helps.'
    assert parse_json_payload(raw) == {"clause_ids": ["Art.61.1"]}


def test_handles_fence_and_think_together() -> None:
    raw = '<think>reasoning</think>\n```json\n{"a": [1, 2]}\n```'
    assert parse_json_payload(raw) == {"a": [1, 2]}


def test_empty_body_raises_rather_than_returning_empty_dict() -> None:
    """An empty body is what a too-small max_tokens produces on a reasoning
    model. Returning {} would record a confident empty answer."""
    with pytest.raises(ValueError):
        parse_json_payload("")
    with pytest.raises(ValueError):
        parse_json_payload("   ")


def test_unparseable_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_payload("I cannot answer that.")


# --- failover -------------------------------------------------------------

class _Stub(JSONProvider):
    def __init__(self, name: str, behaviour) -> None:
        self.name = name
        self.model = "stub"
        self.behaviour = behaviour
        self.calls = 0

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0) -> JSONResult:
        self.calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        return JSONResult(data=self.behaviour, provider=self.name,
                          model=self.model, usage=Usage(1, 1, 2))


def test_first_provider_wins_when_healthy() -> None:
    a = _Stub("a", {"ok": 1})
    b = _Stub("b", {"ok": 2})
    chain = FailoverProvider([a, b])
    assert chain.complete_json("s", "p").data == {"ok": 1}
    assert b.calls == 0


def test_falls_over_on_rate_limit() -> None:
    a = _Stub("a", RateLimited("429 tokens per minute"))
    b = _Stub("b", {"ok": 2})
    chain = FailoverProvider([a, b])
    assert chain.complete_json("s", "p").data == {"ok": 2}


def test_rate_limited_provider_is_benched_not_retried_every_call() -> None:
    """A per-minute budget does not recover in milliseconds; hammering it just
    burns latency on a provider that is going to refuse again."""
    a = _Stub("a", RateLimited("429"))
    b = _Stub("b", {"ok": 2})
    chain = FailoverProvider([a, b], cooldown_s=300)
    for _ in range(4):
        chain.complete_json("s", "p")
    assert a.calls == 1, "benched provider was retried"
    assert b.calls == 4


def test_request_too_large_is_not_failed_over() -> None:
    """A prompt that does not fit will not fit elsewhere either. Failing over
    would waste the fallback's quota on a request that cannot succeed."""
    a = _Stub("a", RequestTooLarge("413"))
    b = _Stub("b", {"ok": 2})
    chain = FailoverProvider([a, b])
    with pytest.raises(RequestTooLarge):
        chain.complete_json("s", "p")
    assert b.calls == 0


def test_generic_failure_does_fall_over() -> None:
    a = _Stub("a", RuntimeError("connection reset"))
    b = _Stub("b", {"ok": 2})
    assert FailoverProvider([a, b]).complete_json("s", "p").data == {"ok": 2}


def test_all_benched_still_attempts_rather_than_abandoning_the_batch() -> None:
    a = _Stub("a", RateLimited("429"))
    chain = FailoverProvider([a], cooldown_s=300)
    with pytest.raises(RuntimeError):
        chain.complete_json("s", "p")
    with pytest.raises(RuntimeError):
        chain.complete_json("s", "p")
    assert a.calls == 2, "a fully benched chain must still try, not give up"


def test_counts_track_which_backend_served() -> None:
    a = _Stub("a", RateLimited("429"))
    b = _Stub("b", {"ok": 2})
    chain = FailoverProvider([a, b])
    chain.complete_json("s", "p")
    assert chain.counts == {"b:stub": 1}


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError):
        FailoverProvider([])
