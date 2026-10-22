from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str | None = Field(default="New chat", max_length=500)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] | None = None
    is_archived: bool | None = None


class ConversationOut(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None
    metadata: dict[str, Any] = Field(validation_alias="metadata_")

    model_config = {"from_attributes": True, "populate_by_name": True}


class ConversationPage(BaseModel):
    items: list[ConversationOut]
    limit: int
    offset: int
    total: int
