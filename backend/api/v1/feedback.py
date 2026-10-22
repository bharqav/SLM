from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user_rate_limited
from backend.db.session import get_db
from backend.models import Conversation, Feedback, Message, User
from backend.schemas.feedback import FeedbackCreate, FeedbackOut

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    request: Request,
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> FeedbackOut:
    msg = await db.get(Message, body.message_id)
    if msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
    c = await db.get(Conversation, msg.conversation_id)
    if c is None or c.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    fb = Feedback(
        message_id=body.message_id,
        user_id=user.id,
        rating=body.rating,
        feedback_text=body.feedback_text,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return FeedbackOut.model_validate(fb)


@router.get("/{feedback_id}", response_model=FeedbackOut)
async def get_feedback(
    feedback_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> FeedbackOut:
    fb = await db.get(Feedback, feedback_id)
    if fb is None or fb.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return FeedbackOut.model_validate(fb)
