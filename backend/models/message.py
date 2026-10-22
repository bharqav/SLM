import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base
from backend.models.enums import MessageRole

if TYPE_CHECKING:
    from backend.models.conversation import Conversation
    from backend.models.feedback import Feedback


_message_role = Enum(
    MessageRole,
    name="message_role",
    native_enum=False,
    values_callable=lambda cls: [e.value for e in cls],
)


class Message(Base):
    """Single turn in a conversation (PRD: Messages + safety/emotion alignment)."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(_message_role, index=True)
    content: Mapped[str] = mapped_column(Text())
    emotion_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    emotion_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    input_safety_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    safety_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    moderation_flags: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    inference_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    inference_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    feedbacks: Mapped[list["Feedback"]] = relationship(back_populates="message")
