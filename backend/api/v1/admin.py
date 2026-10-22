from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Float, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_staff
from backend.db.session import get_db
from backend.models import Conversation, Feedback, Message, User
from backend.services import audit_service
from backend.models.enums import MessageRole
from backend.schemas.admin import AdminStatsOut, AdminUserRow, AdminUsersOut, SafetyAuditOut

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats", response_model=AdminStatsOut)
async def admin_stats(
    db: AsyncSession = Depends(get_db),
    staff: User = Depends(get_current_staff),
) -> AdminStatsOut:
    u = await db.scalar(select(func.count()).select_from(User))
    c = await db.scalar(select(func.count()).select_from(Conversation))
    m = await db.scalar(select(func.count()).select_from(Message))
    f = await db.scalar(select(func.count()).select_from(Feedback))
    out = AdminStatsOut(
        users_total=int(u or 0),
        conversations_total=int(c or 0),
        messages_total=int(m or 0),
        feedback_total=int(f or 0),
        generated_at=datetime.now(timezone.utc),
    )
    await audit_service.log_action(
        db,
        actor=staff,
        action="admin.stats.read",
        resource="admin",
        detail={"users_total": out.users_total},
    )
    await db.commit()
    return out


@router.get("/users", response_model=AdminUsersOut)
async def admin_users(
    db: AsyncSession = Depends(get_db),
    staff: User = Depends(get_current_staff),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> AdminUsersOut:
    r = await db.execute(select(User).order_by(User.created_at.desc()).offset(skip).limit(limit))
    rows = r.scalars().all()
    out = AdminUsersOut(
        items=[AdminUserRow.model_validate(u) for u in rows],
        skip=skip,
        limit=limit,
    )
    await audit_service.log_action(
        db,
        actor=staff,
        action="admin.users.list",
        resource="admin/users",
        detail={"skip": skip, "limit": limit},
    )
    await db.commit()
    return out


@router.post("/safety-audit", response_model=SafetyAuditOut)
async def safety_audit(
    db: AsyncSession = Depends(get_db),
    staff: User = Depends(get_current_staff),
    output_threshold: float = Query(0.85, ge=0.0, le=1.0),
) -> SafetyAuditOut:
    low = await db.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.role == MessageRole.ASSISTANT)
        .where(Message.safety_score.is_not(None))
        .where(cast(Message.safety_score, Float) < output_threshold)
    )
    r = await db.execute(
        select(Message.id).where(Message.moderation_flags.is_not(None)).limit(50)
    )
    blocked = 0
    sample: list[UUID] = []
    for (mid,) in r.all():
        sample.append(mid)
        msg = await db.get(Message, mid)
        if msg and isinstance(msg.moderation_flags, dict) and msg.moderation_flags.get("blocked"):
            blocked += 1
    result = SafetyAuditOut(
        messages_below_output_threshold=int(low or 0),
        messages_with_blocked_flag=blocked,
        sample_message_ids=sample[:20],
    )
    await audit_service.log_action(
        db,
        actor=staff,
        action="admin.safety_audit.run",
        resource="admin/safety-audit",
        detail={"output_threshold": output_threshold, "low_score_count": result.messages_below_output_threshold},
    )
    await db.commit()
    return result
