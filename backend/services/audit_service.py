"""Append-only audit trail for staff actions."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import AuditLog, User


async def log_action(
    db: AsyncSession,
    *,
    actor: User,
    action: str,
    resource: str,
    detail: dict[str, Any] | None = None,
) -> None:
    db.add(
        AuditLog(
            actor_id=actor.id,
            action=action,
            resource=resource,
            detail=detail or {},
        )
    )
    await db.flush()


async def log_system(
    db: AsyncSession,
    *,
    action: str,
    resource: str,
    user_id: UUID | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    db.add(
        AuditLog(
            actor_id=None,
            action=action,
            resource=resource,
            detail={**(detail or {}), "subject_user_id": str(user_id) if user_id else None},
        )
    )
    await db.flush()
