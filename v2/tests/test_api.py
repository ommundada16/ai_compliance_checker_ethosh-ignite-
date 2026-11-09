"""API contract tests.

Endpoints that only read committed artefacts are tested directly. Anything
that needs Qdrant, an embedding model or an LLM is marked `integration` and
skipped when the service is not up, so the suite still runs on a laptop with
nothing else started -- a test suite that requires Docker to be running is a
test suite people stop running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402

client = TestClient(app)


def _qdrant_up() -> bool:
    return client.get("/health").json().get("qdrant", {}).get("ok", False)


# --- health ---------------------------------------------------------------

def test_health_reports_each_dependency_separately() -> None:
    """A single boolean sends whoever is on call to read logs."""
    body = client.get("/health").json()
    assert body["api"] == "ok"
    assert "qdrant" in body
    assert "provider" in body
    assert "corpus" in body


def test_health_answers_even_when_a_dependency_is_down() -> None:
    """Dependencies are built lazily precisely so this endpoint still responds
    and can say WHICH one is unavailable."""
    assert client.get("/health").status_code == 200


# --- data endpoints -------------------------------------------------------

def test_passage_list_omits_text() -> None:
    """The sidebar needs titles; shipping ~19k words to render a list is waste."""
    rows = client.get("/api/passages").json()
    assert rows
    assert "text" not in rows[0]
    assert {"passage_id", "section", "section_title", "page_start"} <= set(rows[0])


def test_single_passage_includes_text() -> None:
    body = client.get("/api/passages/CER.2.11").json()
    assert body["passage_id"] == "CER.2.11"
    assert "contraindications" in body["text"].lower()


def test_unknown_passage_is_404() -> None:
    assert client.get("/api/passages/CER.999").status_code == 404


def test_clause_lookup() -> None:
    body = client.get("/api/clauses/Art.61.1").json()
    assert body["article"] == 61
    assert body["path"] == "Chapter VI > Article 61 > (1)"


def test_unknown_clause_is_404() -> None:
    assert client.get("/api/clauses/Art.999.1").status_code == 404


# --- metrics --------------------------------------------------------------

def test_metrics_serves_committed_results() -> None:
    """The UI must show the numbers the repository records, not a separate
    live computation that could disagree with them."""
    body = client.get("/api/metrics").json()
    assert "v1_baseline" in body
    assert "summary" in body["v1_baseline"]


def test_metrics_strips_per_query_detail() -> None:
    """per_query is thousands of rows the dashboard never reads."""
    body = client.get("/api/metrics").json()
    assert "per_query" not in body["v1_baseline"]
    for result in body.get("v2_ablation", {}).get("results", {}).values():
        assert "per_query" not in result


# --- validation -----------------------------------------------------------

def test_search_rejects_an_empty_query() -> None:
    assert client.post("/api/search", json={"query": "", "k": 5}).status_code == 422


def test_search_rejects_an_out_of_range_k() -> None:
    assert client.post("/api/search", json={"query": "x", "k": 500}).status_code == 422


def test_audit_rejects_a_bad_confidence() -> None:
    body = {"passage_id": "CER.2.11", "min_confidence": 5.0}
    assert client.post("/api/audit/passage", json=body).status_code == 422


# --- integration ----------------------------------------------------------

@pytest.mark.integration
def test_search_returns_ranked_clauses() -> None:
    if not _qdrant_up():
        pytest.skip("Qdrant is not running")
    body = client.post(
        "/api/search",
        json={"query": "clinical evaluation benefit risk ratio", "k": 5,
              "rerank": False},
    ).json()
    assert len(body["results"]) == 5
    assert all(r["clause_id"] for r in body["results"])
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True), "results are not ranked"


@pytest.mark.integration
def test_search_for_an_exact_identifier_finds_that_instrument() -> None:
    """The sparse arm exists for exactly this: identifiers where the literal
    string IS the meaning."""
    if not _qdrant_up():
        pytest.skip("Qdrant is not running")
    body = client.post(
        "/api/search",
        json={"query": "Annex XIV post-market clinical follow-up PMCF plan",
              "k": 10, "rerank": False},
    ).json()
    assert any(r["clause_id"].startswith("Annex.XIV") for r in body["results"])
