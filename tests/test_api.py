"""Contract tests for the FastAPI production API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify /health returns status=ok, catalog_size=9742, and loaded artifacts."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["catalog_size"] == 9742
    assert data["artifacts_loaded"] is True


def test_recommend_endpoint_contract() -> None:
    """Verify POST /recommend returns valid recommendations, profile, and telemetry."""
    resp = client.post(
        "/recommend",
        json={
            "user_id": 10,
            "top_k": 6,
            "mmr_lambda": 0.75,
            "enable_agentic_rerank": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) == 6
    assert body["telemetry"]["total_ms"] > 0.0
    watched = set(body["user_profile"]["watched_movie_ids"])
    for item in body["recommendations"]:
        assert item["movie_id"] not in watched
        assert len(item["explanation"]) > 15


def test_chat_conversational_endpoint_contract() -> None:
    """Verify POST /chat parses natural language constraints and applies guardrails."""
    resp = client.post(
        "/chat",
        json={
            "user_id": 5,
            "query": "thriller psicológico estilo Fincher, < 135 min, depois de 1994, sem terror",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["parsed_intent"] is not None
    assert "Thriller" in body["parsed_intent"]["required_genres"]
    assert "Horror" in body["parsed_intent"]["excluded_genres"]
    assert len(body["recommendations"]) > 0
    for item in body["recommendations"]:
        assert "Horror" not in item["genres"]


def test_cold_start_endpoint_contract() -> None:
    """Verify POST /cold-start recommends items from 3 seed movie IDs."""
    resp = client.post(
        "/cold-start",
        json={
            "seed_movie_ids": [2959, 47, 2571],
            "natural_language_preferences": "neo-noir psychological thriller",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) == 5
    for item in body["recommendations"]:
        assert item["movie_id"] not in {2959, 47, 2571}


def test_import_letterboxd_multipart_endpoint() -> None:
    """Verify POST /import-letterboxd parses uploaded CSV and returns recommendations."""
    csv_bytes = (
        b"Date,Name,Year,Letterboxd URI,Rating\n"
        b"2025-10-12,Fight Club,1999,https://boxd.it/2a9q,5.0\n"
        b"2025-10-14,The Matrix,1999,https://boxd.it/2a1m,4.5\n"
        b"2025-10-19,Pulp Fiction,1994,https://boxd.it/29Pq,5.0\n"
    )

    resp = client.post(
        "/import-letterboxd",
        files={"file": ("ratings.csv", csv_bytes, "text/csv")},
        data={"top_k": "5", "mmr_lambda": "0.75"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["import_summary"]["matched_watched_count"] == 3
    recs = payload["recommendation_response"]["recommendations"]
    assert len(recs) == 5
    for r in recs:
        assert r["movie_id"] not in {2959, 2571, 296}


def test_catalog_search_and_telemetry_endpoints() -> None:
    """Verify GET /catalog/search and GET /telemetry/recent."""
    s_resp = client.get("/catalog/search", params={"q": "Matrix", "limit": 5})
    assert s_resp.status_code == 200
    items = s_resp.json()
    assert any(it["movie_id"] == 2571 for it in items)

    t_resp = client.get("/telemetry/recent", params={"limit": 5})
    assert t_resp.status_code == 200
    assert len(t_resp.json()) >= 1
