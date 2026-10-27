"""PRD-aligned fields: users, conversations, messages, feedback, sessions

Revision ID: 002
Revises: 001
Create Date: 2026-04-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")))
    op.add_column("users", sa.Column("email_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")))
    op.add_column("users", sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False))
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))

    op.add_column(
        "conversations",
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "conversations",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.add_column("conversations", sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f("ix_conversations_last_message_at"), "conversations", ["last_message_at"], unique=False)

    op.add_column("messages", sa.Column("input_safety_score", sa.Numeric(5, 4), nullable=True))
    op.add_column("messages", sa.Column("moderation_flags", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("inference_provider", sa.String(length=64), nullable=True))
    op.add_column("messages", sa.Column("inference_model", sa.String(length=256), nullable=True))
    op.add_column("messages", sa.Column("prompt_tokens", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("completion_tokens", sa.Integer(), nullable=True))

    op.add_column("feedback", sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False))
    op.create_unique_constraint("uq_feedback_user_message", "feedback", ["user_id", "message_id"])

    op.add_column("sessions", sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("sessions", sa.Column("user_agent", sa.String(length=512), nullable=True))
    op.add_column("sessions", sa.Column("ip_address", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("sessions", "ip_address")
    op.drop_column("sessions", "user_agent")
    op.drop_column("sessions", "revoked_at")

    op.drop_constraint("uq_feedback_user_message", "feedback", type_="unique")
    op.drop_column("feedback", "created_at")

    op.drop_column("messages", "completion_tokens")
    op.drop_column("messages", "prompt_tokens")
    op.drop_column("messages", "inference_model")
    op.drop_column("messages", "inference_provider")
    op.drop_column("messages", "moderation_flags")
    op.drop_column("messages", "input_safety_score")

    op.drop_index(op.f("ix_conversations_last_message_at"), table_name="conversations")
    op.drop_column("conversations", "last_message_at")
    op.drop_column("conversations", "updated_at")
    op.drop_column("conversations", "is_archived")

    op.drop_column("users", "last_login_at")
    op.drop_column("users", "updated_at")
    op.drop_column("users", "email_verified")
    op.drop_column("users", "is_active")
