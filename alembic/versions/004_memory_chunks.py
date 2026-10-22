"""memory_chunks for long-term memory

Revision ID: 004
Revises: 003
Create Date: 2026-04-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_memory_chunks_user_id"), "memory_chunks", ["user_id"], unique=False)
    op.create_index(op.f("ix_memory_chunks_conversation_id"), "memory_chunks", ["conversation_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_memory_chunks_conversation_id"), table_name="memory_chunks")
    op.drop_index(op.f("ix_memory_chunks_user_id"), table_name="memory_chunks")
    op.drop_table("memory_chunks")
