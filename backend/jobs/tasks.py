"""
RQ/Celery-compatible task stubs. Wire `rq worker` or Celery to Redis URL from settings.

Recommended queue names: `embed`, `safety`, `analytics`, `eval`.
"""

from __future__ import annotations

import logging
from uuid import UUID

logger = logging.getLogger(__name__)


def enqueue_memory_embed(user_id: str, conversation_id: str, text_snippet: str) -> None:
    """Push embedding job — implement with RQ: Queue(connection=redis).enqueue(embed_memory_job, ...)"""
    logger.info("enqueue_memory_embed stub user=%s conv=%s", user_id, conversation_id)


def embed_memory_job(user_id: str, conversation_id: str, text_snippet: str) -> None:
    """Worker: compute embedding and upsert MemoryChunk (keep chat path lean)."""
    logger.info("embed_memory_job stub")


def safety_rescan_job(message_id: str) -> None:
    logger.info("safety_rescan_job stub id=%s", message_id)


def eval_batch_job(dataset_version: str) -> None:
    logger.info("eval_batch_job stub version=%s", dataset_version)


def parse_uuid(s: str) -> UUID:
    return UUID(s)
