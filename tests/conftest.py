import os

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["JWT_SECRET"] = "pytest-jwt-secret-key-must-be-long-enough"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["ADMIN_BOOTSTRAP_EMAIL"] = "staff@example.com"

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import backend.models  # noqa: F401 — register models
from backend.db.base import Base
from backend.db import session as session_module


@pytest_asyncio.fixture
async def prepare_db():
    async with session_module.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with session_module.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(prepare_db, monkeypatch):
    import backend.core.redis_client as redis_client
    import backend.middleware.rate_limit as rl

    async def noop_ip(request, limit=None):
        return None

    async def noop_user(request, user_id):
        return None

    async def ping_ok():
        return True

    monkeypatch.setattr(rl, "enforce_rate_limit_ip", noop_ip)
    monkeypatch.setattr(rl, "enforce_rate_limit_user", noop_user)
    monkeypatch.setattr(redis_client, "ping_redis", ping_ok)

    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
