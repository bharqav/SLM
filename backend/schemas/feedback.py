from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    message_id: UUID
    rating: int = Field(ge=1, le=5)
    feedback_text: str | None = Field(default=None, max_length=4000)


class FeedbackOut(BaseModel):
    id: UUID
    message_id: UUID
    user_id: UUID
    rating: int
    created_at: datetime

    model_config = {"from_attributes": True}
