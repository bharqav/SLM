from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class MessageOut(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    emotion_label: str | None
    emotion_confidence: Decimal | None
    input_safety_score: Decimal | None = None
    safety_score: Decimal | None
    moderation_flags: dict[str, Any] | None = None
    inference_provider: str | None = None
    inference_model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    conversation_id: UUID
    content: str = Field(min_length=1, max_length=32000)


class ChatResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
    transparency: dict[str, Any] | None = None
