from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import desc, func, nulls_last, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user_rate_limited
from backend.db.session import get_db
from backend.models import Conversation, User
from backend.schemas.conversation import ConversationCreate, ConversationOut, ConversationPage, ConversationUpdate

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationPage)
async def list_conversations(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ConversationPage:
    base = select(Conversation).where(Conversation.user_id == user.id)
    total = (await db.execute(select(func.count()).select_from(Conversation).where(Conversation.user_id == user.id))).scalar_one()
    r = await db.execute(
        base.order_by(nulls_last(desc(Conversation.last_message_at)), desc(Conversation.updated_at)).limit(limit).offset(
            offset
        )
    )
    rows = r.scalars().all()
    return ConversationPage(
        items=[ConversationOut.model_validate(c) for c in rows],
        limit=limit,
        offset=offset,
        total=int(total),
    )


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: Request,
    body: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> ConversationOut:
    title = body.title or "New chat"
    conv = Conversation(user_id=user.id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return ConversationOut.model_validate(conv)


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    request: Request,
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> ConversationOut:
    conv = await db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return ConversationOut.model_validate(conv)


@router.put("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    request: Request,
    conversation_id: UUID,
    body: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> ConversationOut:
    conv = await db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if body.title is not None:
        conv.title = body.title
    if body.metadata is not None:
        conv.metadata_ = body.metadata
    if body.is_archived is not None:
        conv.is_archived = body.is_archived
    await db.commit()
    await db.refresh(conv)
    return ConversationOut.model_validate(conv)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    request: Request,
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> None:
    conv = await db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    await db.delete(conv)
    await db.commit()
