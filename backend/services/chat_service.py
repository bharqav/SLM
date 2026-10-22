from collections.abc import AsyncIterator
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core import metrics as prom_metrics
from backend.models import Message
from backend.services import chat_pipeline, inference_service, safety_service
from backend.services.chat_pipeline import StreamPrepareResult, TurnState


async def run_chat_turn(
    db: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    content: str,
) -> tuple[Message, Message, TurnState]:
    return await chat_pipeline.run_turn(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
        content=content,
    )


async def begin_stream_turn(
    db: AsyncSession,
    *,
    user_id: UUID,
    conversation_id: UUID,
    content: str,
) -> StreamPrepareResult:
    return await chat_pipeline.prepare_stream_turn(
        db,
        user_id=user_id,
        conversation_id=conversation_id,
        content=content,
    )


async def persist_assistant_message(
    db: AsyncSession,
    *,
    conversation_id: UUID,
    full_text: str,
    prompt: str,
    crisis_stream: bool = False,
) -> Message:
    return await chat_pipeline.finalize_stream_turn(
        db,
        conversation_id=conversation_id,
        full_text=full_text,
        prompt=prompt,
        crisis_stream=crisis_stream,
    )


async def stream_tokens(
    prompt: str,
    crisis_fixed: str | None = None,
    *,
    user_id: UUID | None = None,
) -> AsyncIterator[str]:
    """Yield response text in word chunks — but ONLY after output safety passes.

    The gateway buffers the full completion (batch provider contract), so we run
    the output safety gate on the complete text *before* yielding anything. A
    policy-violating completion is replaced with the canned refusal up front and
    the user never sees the violating bytes (previously they streamed live and
    only the persisted record was repaired).
    """
    if crisis_fixed:
        for word in crisis_fixed.split():
            yield word + " "
        return

    full = await inference_service.complete_full(prompt, user_id=user_id)
    score, flags = await safety_service.check_output(full)
    if safety_service.should_block_output(score) or safety_service.should_block_from_flags(flags):
        prom_metrics.safety_violation_flags_total.labels(stage="output").inc()
        full = (
            "I can't provide that response. If you're in crisis, please contact local emergency services "
            "or a crisis helpline."
        )
    for word in full.split():
        yield word + " "
