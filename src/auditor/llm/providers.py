"""Concrete providers: Groq (cloud), Ollama (local), and a failover chain."""

from __future__ import annotations

import os
import random
import re
import time

from auditor.llm.base import (
    JSONProvider,
    JSONResult,
    RateLimited,
    RequestTooLarge,
    Usage,
    parse_json_payload,
)


def _classify(exc: Exception) -> Exception:
    """Turn a vendor exception into one of ours.

    Matching on the message rather than the exception class because the same
    condition arrives as different types across SDK versions, and the failover
    chain has to tell "quota exhausted, try elsewhere" from "this prompt is
    broken and will fail everywhere".
    """
    text = str(exc)
    lowered = text.lower()
    if "429" in text or "rate limit" in lowered or "tokens per minute" in lowered:
        return RateLimited(text)
    if "413" in text or "too large" in lowered or "context length" in lowered:
        return RequestTooLarge(text)
    return exc


class TokenPacer:
    """A rolling per-minute token budget, enforced BEFORE a request is sent.

    Groq's free tier allows 8000 tokens per minute. Reactive backoff cannot fix
    a per-MINUTE limit: an exponential retry capped at seconds simply retries
    inside the same exhausted window. Waiting until the window has room first is
    what actually works.

    This lives on the provider rather than in a script because it is a property
    of the provider's quota, not of any one caller. Having it in one tool and
    not another is exactly how an audit run fired 19 requests back to back and
    lost 18 of them to 429.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        self.budget = tokens_per_minute
        self.spent: list[tuple[float, int]] = []

    def wait_for(self, tokens: int) -> None:
        if self.budget <= 0:
            return
        while True:
            now = time.time()
            self.spent = [(t, n) for t, n in self.spent if now - t < 60.0]
            used = sum(n for _, n in self.spent)
            if used + tokens <= self.budget or not self.spent:
                return
            time.sleep(max(0.5, 60.0 - (now - self.spent[0][0]) + 0.5))

    def record(self, tokens: int) -> None:
        self.spent.append((time.time(), tokens))


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic: ~3 characters per token.

    Regulatory English is dense. Overestimating costs a little throughput;
    underestimating costs a 429 and a lost request.
    """
    return len(text) // 3 + 128


_RETRY_AFTER = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


class GroqProvider(JSONProvider):
    """Groq's free tier.

    Fast and far stronger than anything that fits on a 4 GB laptop GPU, but
    metered per minute, so it is a provider to fail OVER from rather than to
    depend on for a long batch run.
    """

    name = "groq"

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b",
                 reasoning_effort: str = "low",
                 tokens_per_minute: int | None = None,
                 max_retries: int = 4) -> None:
        from groq import Groq

        if not api_key:
            raise ValueError("GroqProvider needs an API key")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.pacer = TokenPacer(
            tokens_per_minute
            if tokens_per_minute is not None
            else int(os.getenv("GROQ_TPM", "8000"))
        )
        self._client = Groq(api_key=api_key)

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0) -> JSONResult:
        estimated = estimate_tokens(system) + estimate_tokens(prompt) + max_tokens
        last: Exception | None = None

        for attempt in range(self.max_retries):
            self.pacer.wait_for(estimated)
            started = time.time()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                    # Must cover the model's internal reasoning AND its visible
                    # output. gpt-oss reasons first; a tight budget is consumed
                    # entirely by reasoning and JSON mode then fails with an
                    # empty body, which reads like a prompt bug and is not.
                    max_tokens=max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:  # noqa: BLE001 - reclassified below
                classified = _classify(exc)
                # The prompt will not shrink on retry, and no other provider
                # would accept it either.
                if isinstance(classified, RequestTooLarge):
                    raise classified from exc
                last = classified
                # Charge the estimate anyway: as far as the server is concerned
                # a refused request still consumed quota.
                self.pacer.record(estimated)
                if attempt == self.max_retries - 1:
                    break
                # Groq states the wait in the error body; honour it rather than
                # guessing, with jitter so parallel callers do not synchronise
                # onto the same instant.
                match = _RETRY_AFTER.search(str(exc))
                delay = float(match.group(1)) + 1.0 if match else 20.0 * (attempt + 1)
                time.sleep(min(90.0, delay) + random.random())
                continue

            usage = response.usage
            self.pacer.record(usage.total_tokens)
            return JSONResult(
                data=parse_json_payload(response.choices[0].message.content or ""),
                provider=self.name,
                model=self.model,
                usage=Usage(usage.prompt_tokens, usage.completion_tokens,
                            usage.total_tokens),
                latency_s=time.time() - started,
            )

        raise last if last else RuntimeError("groq request failed")


class OllamaProvider(JSONProvider):
    """A local model over Ollama's NATIVE /api/chat endpoint.

    Unmetered and offline, which makes it the right floor for a long batch:
    slower per call, but it cannot run out of quota halfway through.

    Native rather than the OpenAI-compatible shim, for one concrete reason.
    Reasoning models (qwen3) emit a thinking block before their answer, and
    Ollama routes it to a separate `thinking` field. Through the OpenAI shim
    the `think` option is silently dropped, so qwen3:4b burns the entire token
    budget on reasoning and returns finish_reason="length" with EMPTY content
    -- measured at 2000 completion tokens and not one character of output. The
    native endpoint honours think=False and returns the answer.

    Measured on this machine (4 GB VRAM):
        qwen3:4b      ~20s   think=False required, else empty
        llama3.1:8b   ~16s   4.92 GB, spills to CPU

    No SDK: stdlib urllib, so nothing transitive has to be present.
    """

    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300, think: bool = False) -> None:
        self.model = model
        self.think = think
        self.timeout = timeout
        # Accept an OpenAI-style base_url and strip the shim suffix, so a
        # .env written for the compat endpoint keeps working.
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0) -> JSONResult:
        import json as _json
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "think": self.think,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = _json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise _classify(RuntimeError(f"{exc.code} {detail}")) from exc
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc) from exc

        message = payload.get("message", {})
        content = message.get("content", "")
        if not content and message.get("thinking"):
            # think=False was not honoured (older Ollama). Say so plainly --
            # this failure is otherwise indistinguishable from a bad prompt.
            raise ValueError(
                f"{self.model} returned only a reasoning block "
                f"({len(message['thinking'])} chars) and no content; "
                "this Ollama build appears to ignore think=False"
            )
        return JSONResult(
            data=parse_json_payload(content),
            provider=self.name,
            model=self.model,
            usage=Usage(
                payload.get("prompt_eval_count", 0) or 0,
                payload.get("eval_count", 0) or 0,
                (payload.get("prompt_eval_count", 0) or 0) + (payload.get("eval_count", 0) or 0),
            ),
            latency_s=time.time() - started,
        )


class FailoverProvider(JSONProvider):
    """Try providers in order; move on when one is rate-limited.

    Ordered fastest-and-best first, most-reliable last. Groq is quick and
    strong but metered; Ollama is slower but cannot run out of quota. A long
    batch therefore runs at cloud speed while the budget lasts and finishes
    locally instead of dying at 57% -- which is what happened on the first
    labelling run.

    A rate-limited provider is benched for `cooldown_s` rather than retried on
    every call, since a per-minute budget does not recover in milliseconds.
    RequestTooLarge is NOT failed over: a prompt that does not fit is the
    caller's problem and will not fit elsewhere either.
    """

    name = "failover"

    def __init__(self, providers: list[JSONProvider], cooldown_s: float = 60.0) -> None:
        if not providers:
            raise ValueError("FailoverProvider needs at least one provider")
        self.providers = providers
        self.cooldown_s = cooldown_s
        self._benched: dict[str, float] = {}
        self.model = providers[0].model
        self.counts: dict[str, int] = {}

    def _available(self) -> list[JSONProvider]:
        now = time.time()
        ready = [p for p in self.providers if self._benched.get(p.describe(), 0.0) <= now]
        # If everything is benched, use the one that recovers soonest rather
        # than giving up on the batch.
        return ready or [min(self.providers, key=lambda p: self._benched.get(p.describe(), 0.0))]

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0) -> JSONResult:
        last: Exception | None = None
        for provider in self._available():
            try:
                result = provider.complete_json(system, prompt, max_tokens, temperature)
            except RequestTooLarge:
                raise
            except RateLimited as exc:
                self._benched[provider.describe()] = time.time() + self.cooldown_s
                last = exc
                continue
            except Exception as exc:  # noqa: BLE001 - try the next backend
                last = exc
                continue
            self.counts[provider.describe()] = self.counts.get(provider.describe(), 0) + 1
            return result
        raise RuntimeError(f"all providers failed; last error: {last}")

    def describe(self) -> str:
        return " -> ".join(p.describe() for p in self.providers)
