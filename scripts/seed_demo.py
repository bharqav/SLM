"""Seed a demo account, conversation, and starter messages for local demos."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select

from backend.core.security import hash_password
from backend.db.session import AsyncSessionLocal
from backend.models import Conversation, Message, MessageRole, User


DEMO_EMAIL = "demo@example.com"
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo-password-123"


async def main() -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == DEMO_EMAIL))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                email=DEMO_EMAIL,
                username=DEMO_USERNAME,
                password_hash=hash_password(DEMO_PASSWORD),
                is_staff=False,
                preferences={
                    "region": "US",
                    "memory_opt_in": True,
                    "tone_preference": "gentle",
                    "terms_version_accepted": "2026-04-12",
                },
            )
            db.add(user)
            await db.flush()

        result = await db.execute(select(Conversation).where(Conversation.user_id == user.id))
        conversation = result.scalar_one_or_none()
        if conversation is None:
            conversation = Conversation(user_id=user.id, title="Demo conversation")
            db.add(conversation)
            await db.flush()

            user_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content="I'm feeling overwhelmed and need a calm example conversation.",
                emotion_label="anxiety",
                emotion_confidence=Decimal("0.8700"),
                input_safety_score=Decimal("0.9900"),
            )
            assistant_message = Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content="That sounds difficult. We can take it one step at a time.",
                safety_score=Decimal("0.9800"),
                inference_provider="demo",
                inference_model="seed-script",
            )
            db.add_all([user_message, assistant_message])
            conversation.last_message_at = datetime.now(timezone.utc)

        await db.commit()

    print(f"Seeded demo user {DEMO_EMAIL} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    asyncio.run(main())