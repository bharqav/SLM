"""GDPR-style export and account deletion."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.deps import get_current_user
from backend.core.security import verify_password
from backend.db.session import get_db
from backend.models import Conversation, Feedback, MemoryChunk, User, UserSession
from backend.services import audit_service

router = APIRouter(prefix="/gdpr", tags=["gdpr"])


class DeleteAccountBody(BaseModel):
    password: str


@router.get("/export")
async def export_my_data(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Return a portable JSON snapshot of the authenticated user's data."""
    r = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    convs = r.scalars().unique().all()
    out_convs = []
    for c in convs:
        msgs = [
            {
                "id": str(m.id),
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in c.messages
        ]
        out_convs.append(
            {
                "id": str(c.id),
                "title": c.title,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "messages": msgs,
            }
        )
    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "username": user.username,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        },
        "conversations": out_convs,
    }


@router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_account(
    body: DeleteAccountBody,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    """Delete the authenticated user and all related rows (cascade)."""
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    uid = user.id
    await audit_service.log_system(
        db,
        action="gdpr.account.delete",
        resource="gdpr",
        user_id=uid,
        detail={"cascade": "user_sessions_conversations_messages_memory"},
    )
    # Explicit deletes keep behavior consistent on SQLite tests and production DBs.
    await db.execute(delete(UserSession).where(UserSession.user_id == user.id))
    await db.execute(delete(Feedback).where(Feedback.user_id == user.id))
    await db.execute(delete(MemoryChunk).where(MemoryChunk.user_id == user.id))
    await db.execute(delete(Conversation).where(Conversation.user_id == user.id))
    await db.delete(user)
    await db.commit()
