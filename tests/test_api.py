from uuid import UUID

import pytest


@pytest.mark.asyncio
async def test_metrics(client):
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert len(r.text) > 0


@pytest.mark.asyncio
async def test_admin_forbidden_for_user(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "plain@example.com",
            "username": "plain",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    token = r.json()["access_token"]
    r2 = await client.get("/api/v1/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 403


@pytest.mark.asyncio
async def test_admin_staff_stats(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "staff@example.com",
            "username": "staffer",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    assert r.status_code == 201
    token = r.json()["access_token"]
    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json().get("is_staff") is True
    r2 = await client.get("/api/v1/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.json()["users_total"] >= 1


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready(client):
    r = await client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_auth_flow(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "a@example.com",
            "username": "alice",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    assert r.status_code == 201
    data = r.json()
    assert "access_token" in data and "refresh_token" in data
    token = data["access_token"]

    r2 = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert r2.json()["email"] == "a@example.com"

    r3 = await client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "password123"})
    assert r3.status_code == 200
    refresh = r3.json()["refresh_token"]

    r4 = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert r4.status_code == 200
    new_refresh = r4.json()["refresh_token"]

    r5 = await client.post("/api/v1/auth/logout", json={"refresh_token": new_refresh})
    assert r5.status_code == 200


@pytest.mark.asyncio
async def test_conversations_and_chat(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "b@example.com",
            "username": "bob",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    assert r.status_code == 201
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.post("/api/v1/conversations", json={"title": "Test"}, headers=headers)
    assert r.status_code == 201
    cid = r.json()["id"]
    UUID(cid)

    r = await client.get("/api/v1/conversations", headers=headers)
    assert r.status_code == 200
    assert len(r.json()["items"]) == 1

    r = await client.post(
        "/api/v1/chat",
        json={"conversation_id": cid, "content": "I'm feeling anxious today"},
        headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert "user_message" in body and "assistant_message" in body
    assert body["user_message"]["emotion_label"] == "anxiety"


@pytest.mark.asyncio
async def test_chat_stream_get(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "c@example.com",
            "username": "carol",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.post("/api/v1/conversations", json={}, headers=headers)
    cid = r.json()["id"]

    r = await client.get(
        "/api/v1/chat/stream",
        params={"conversation_id": cid, "message": "Hello"},
        headers=headers,
    )
    assert r.status_code == 200
    text = r.text
    assert "token" in text or "done" in text


@pytest.mark.asyncio
async def test_gdpr_export(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "gdpr@example.com",
            "username": "gdpr",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    token = r.json()["access_token"]
    r2 = await client.get("/api/v1/gdpr/export", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    data = r2.json()
    assert "user" in data and "conversations" in data
    assert data["user"]["email"] == "gdpr@example.com"


@pytest.mark.asyncio
async def test_feedback(client):
    r = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "d@example.com",
            "username": "dan",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.post("/api/v1/conversations", json={}, headers=headers)
    cid = r.json()["id"]

    r = await client.post(
        "/api/v1/chat",
        json={"conversation_id": cid, "content": "Hi"},
        headers=headers,
    )
    mid = r.json()["assistant_message"]["id"]

    r = await client.post(
        "/api/v1/feedback",
        json={"message_id": mid, "rating": 5, "feedback_text": "ok"},
        headers=headers,
    )
    assert r.status_code == 201
    assert r.json()["rating"] == 5
    fid = r.json()["id"]
    r = await client.get(f"/api/v1/feedback/{fid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == fid


@pytest.mark.asyncio
async def test_security_headers_present(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "default-src 'none'" in (r.headers.get("content-security-policy") or "")


@pytest.mark.asyncio
async def test_gdpr_delete_account_revokes_access(client):
    signup = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "erase@example.com",
            "username": "eraseme",
            "password": "password123",
            "accept_terms": True,
            "confirm_age_18": True,
        },
    )
    assert signup.status_code == 201
    token = signup.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    conv = await client.post("/api/v1/conversations", json={"title": "Erase me"}, headers=headers)
    assert conv.status_code == 201
    cid = conv.json()["id"]
    chat = await client.post(
        "/api/v1/chat",
        json={"conversation_id": cid, "content": "This should be deleted"},
        headers=headers,
    )
    assert chat.status_code == 200

    deletion = await client.request(
        "DELETE",
        "/api/v1/gdpr/account",
        headers=headers,
        json={"password": "password123"},
    )
    assert deletion.status_code == 204

    # Token should now point to a deleted user and be rejected.
    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 401
