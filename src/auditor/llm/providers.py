"""Concrete providers: Groq (cloud), Ollama (local), and a failover chain."""

from __future__ import annotations

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


class GroqProvider(JSONProvider):
    """Groq's free tier.

    Fast and far stronger than anything that fits on a 4 GB laptop GPU, but
    metered per minute, so it is a provider to fail OVER from rather than to
    depend on for a long batch run.
    """

    name = "groq"

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b",
                 reasoning_effort: str = "low") -> None:
        from groq import Groq

        if not api_key:
            raise ValueError("GroqProvider needs an API key")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client = Groq(api_key=api_key)

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0) -> JSONResult:
        started = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                # Must cover the model's internal reasoning AND its visible
                # output. gpt-oss reasons first; a tight budget is consumed
                # entirely by reasoning and JSON mode then fails with an empty
                # body, which reads like a prompt bug and is not.
                max_tokens=max_tokens,
                reasoning_effort=self.reasoning_effort,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - reclassified immediately
            raise _classify(exc) from exc

        usage = response.usage
        return JSONResult(
            data=parse_json_payload(response.choices[0].message.content or ""),
            provider=self.name,
            model=self.model,
            usage=Usage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
            latency_s=time.time() - started,
        )


class OllamaProvider(JSONProvider):
    """A local model over Ollama's OpenAI-compatible endpoint.

    Unmetered and offline, which is what makes it the right floor for a long
    batch: slower per call, but it cannot run out of quota halfway through.

    `think=False` is passed through for reasoning models (qwen3). Left on, they
    spend the token budget on a <think> block and can return nothing parseable;
    base.parse_json_payload strips the block defensively, but not emitting it
    is cheaper and more reliable.
    """

    name = "ollama"

    def __init__(self, model: str = "qwen3:4b",
                 base_url: str = "http://localhost:11434/v1",
                 timeout: int = 300, think: bool = False) -> None:
        from openai import OpenAI

        self.model = model
        self.think = think
        # Ollama ignores the key but the client requires a non-empty string.
        self._client = OpenAI(api_key="ollama", base_url=base_url, timeout=timeout)

    def complete_json(self, system, prompt, max_tokens=1024, temperature=0.0) -> JSONResult:
        started = time.time()
        kwargs = {}
        if not self.think:
            kwargs["extra_body"] = {"think": False}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            # Not every Ollama build accepts `think`; retry once without it
            # rather than failing over to a slower backend for a flag.
            if "think" in message.lower() and kwargs:
                try:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except Exception as inner:  # noqa: BLE001
                    raise _classify(inner) from inner
            else:
                raise _classify(exc) from exc

        usage = getattr(response, "usage", None)
        return JSONResult(
            data=parse_json_payload(response.choices[0].message.content or ""),
            provider=self.name,
            model=self.model,
            usage=Usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
                getattr(usage, "total_tokens", 0) or 0,
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
