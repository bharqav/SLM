from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class AdminStatsOut(BaseModel):
    users_total: int
    conversations_total: int
    messages_total: int
    feedback_total: int
    generated_at: datetime


class AdminUserRow(BaseModel):
    id: UUID
    email: str
    username: str
    is_active: bool
    is_staff: bool
    created_at: datetime
    last_login_at: datetime | None

    model_config = {"from_attributes": True}


class AdminUsersOut(BaseModel):
    items: list[AdminUserRow]
    skip: int
    limit: int


class SafetyAuditOut(BaseModel):
    messages_below_output_threshold: int
    messages_with_blocked_flag: int
    sample_message_ids: list[UUID] = Field(default_factory=list)
