"""Long-term memory: embeddings + retrieval (JSON vectors; pgvector optional later)."""

import hashlib
import math
from datetime import datetime, timezone
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.models.memory_chunk import MemoryChunk
from backend.services.memory_pii import redact_for_embedding


def _deterministic_embedding(text: str, dim: int | None = None) -> list[float]:
    d = dim or settings.MEMORY_EMBEDDING_DIM
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    for i in range(d):
        b = raw[i % len(raw)]
        out.append((b - 128) / 128.0)
    return out


async def _hf_embedding(text: str) -> list[float] | None:
    if not settings.HF_EMBEDDING_URL or not settings.HF_API_TOKEN:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                settings.HF_EMBEDDING_URL,
                headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
                json={"inputs": text},
            )
            if not r.is_success:
                return None
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [float(x) for x in data[0]]
            if isinstance(data, list) and data and isinstance(data[0], dict):
                v = data[0].get("embedding") or data[0].get("values")
                if isinstance(v, list):
                    return [float(x) for x in v]
    except Exception:
        return None
    return None


async def embed_text(text: str) -> list[float]:
    v = await _hf_embedding(text)
    if v:
        return v
    return _deterministic_embedding(text)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _memory_text(text: str, max_len: int) -> str:
    t = text[:max_len]
    if settings.MEMORY_PI_REDACT:
        return redact_for_embedding(t)
    return t


async def retrieve_context(
    db: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    query_text: str,
    memory_allowed: bool = True,
) -> str:
    if not settings.MEMORY_ENABLED or not memory_allowed:
        return ""
    q_for_emb = _memory_text(query_text, 2000)
    q_emb = await embed_text(q_for_emb)
    r = await db.execute(
        select(MemoryChunk)
        .where(MemoryChunk.user_id == user_id)
        .where(MemoryChunk.conversation_id == conversation_id)
        .order_by(MemoryChunk.created_at.desc())
        .limit(50)
    )
    rows = r.scalars().all()
    if not rows:
        return ""
    now = datetime.now(timezone.utc)
    sim_w = max(0.0, settings.MEMORY_SIMILARITY_WEIGHT)
    rec_w = max(0.0, settings.MEMORY_RECENCY_WEIGHT)
    w_sum = sim_w + rec_w
    if w_sum == 0:
        sim_w, rec_w = 1.0, 0.0
    else:
        sim_w, rec_w = sim_w / w_sum, rec_w / w_sum

    scored: list[tuple[float, MemoryChunk]] = []
    for ch in rows:
        if not isinstance(ch.embedding, list):
            continue
        try:
            emb = [float(x) for x in ch.embedding]
            similarity = _cosine(q_emb, emb)
            created = ch.created_at if ch.created_at.tzinfo else ch.created_at.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            recency = 1.0 / (1.0 + age_days)
            s = (sim_w * similarity) + (rec_w * recency)
        except Exception:
            continue
        scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c.content for _, c in scored[:3]]
    return " ".join(top)[:1500]


async def index_user_turn(
    db: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    text: str,
    memory_allowed: bool = True,
) -> None:
    if not memory_allowed or not settings.MEMORY_ENABLED or len(text.strip()) < 3:
        return
    safe = _memory_text(text, 4000)
    emb = await embed_text(safe[:2000])
    ch = MemoryChunk(
        user_id=user_id,
        conversation_id=conversation_id,
        content=safe,
        embedding=emb,
    )
    db.add(ch)
    await db.flush()
