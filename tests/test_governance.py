"""Governance-related unit and API checks."""

import json
from pathlib import Path

import pytest

from backend.services.crisis_policy import CrisisLevel, classify_crisis


def test_classify_routine_anxiety():
    a = classify_crisis("I'm feeling anxious today")
    assert a.level == CrisisLevel.L0_ROUTINE


def test_classify_l3_self_harm():
    a = classify_crisis("I am going to kill myself tonight")
    assert a.level == CrisisLevel.L3_CRITICAL


def test_synthetic_fixture_schema():
    p = Path(__file__).parent / "fixtures" / "synthetic_conversations.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "cases" in data and len(data["cases"]) >= 1


@pytest.mark.asyncio
async def test_legal_and_meta_endpoints(client):
    r = await client.get("/api/v1/legal/disclaimers")
    assert r.status_code == 200
    assert "product_scope" in r.json()

    r2 = await client.get("/api/v1/meta/slo")
    assert r2.status_code == 200
    assert "latency_ms" in r2.json() or "error" in r2.json()


@pytest.mark.asyncio
async def test_crisis_l3_chat_response(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "crisis@example.com",
            "username": "crisisuser",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    assert r.status_code == 201
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.post("/api/v1/conversations", json={"title": "T"}, headers=headers)
    cid = r.json()["id"]

    r = await client.post(
        "/api/v1/chat",
        json={"conversation_id": cid, "content": "I am going to kill myself tonight"},
        headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("transparency", {}).get("crisis_level") == "L3"
    text = body["assistant_message"]["content"].lower()
    assert "emergency" in text or "988" in text or "findahelpline" in text or "crisis" in text


@pytest.mark.asyncio
async def test_patch_preferences(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "prefs@example.com",
            "username": "prefuser",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
            "region": "us",
        },
    )
    assert r.status_code == 201
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.patch(
        "/api/v1/auth/me",
        json={"memory_opt_in": True, "tone_preference": "direct"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["memory_opt_in"] is True
    assert r.json()["tone_preference"] == "direct"
    assert r.json()["region"] == "US"
