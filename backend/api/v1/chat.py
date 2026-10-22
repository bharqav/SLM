import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user_rate_limited
from backend.core import idempotency as idem
from backend.core.config import settings
from backend.core.security import get_sub_from_token
from backend.db.session import AsyncSessionLocal, get_db
from backend.models import User
from backend.schemas.message import ChatRequest, ChatResponse, MessageOut
from backend.services import chat_service
from backend.services.chat_pipeline import ChatBlockedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_REFUSAL_NOTE = "Output was constrained by safety policy."


def _idem_hash(body: ChatRequest) -> str:
    import hashlib

    payload = json.dumps(
        {"conversation_id": str(body.conversation_id), "content": body.content},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _transparency_event(transparency: dict) -> dict:
    return {"type": "transparency", "transparency": transparency}


@router.post("", response_model=ChatResponse)
async def chat_message(
    request: Request,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
) -> ChatResponse:
    idem_key = request.headers.get("Idempotency-Key") or request.headers.get("idempotency-key") or ""
    body_hash = _idem_hash(body)

    if idem_key.strip():
        begun = await idem.begin_chat_request(user.id, idem_key, body_hash)
        if idem.is_body_mismatch(begun):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Idempotency-Key already in use for a different request",
            )
        if idem.is_in_flight(begun):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Identical request with this Idempotency-Key is already in progress",
            )
        if begun is not None:
            return ChatResponse.model_validate(begun)

    try:
        u, a, st = await chat_service.run_chat_turn(
            db,
            user_id=user.id,
            conversation_id=body.conversation_id,
            content=body.content,
        )
    except ChatBlockedError as e:
        if idem_key.strip():
            await idem.complete_chat_request(user.id, idem_key, body_hash, None)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except PermissionError:
        if idem_key.strip():
            await idem.complete_chat_request(user.id, idem_key, body_hash, None)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    await db.commit()
    resp = ChatResponse(
        user_message=MessageOut.model_validate(u),
        assistant_message=MessageOut.model_validate(a),
        transparency=st.transparency,
    )

    if idem_key.strip():
        await idem.complete_chat_request(user.id, idem_key, body_hash, resp.model_dump(mode="json"))
    return resp


@router.websocket("/ws/{conversation_id}")
async def chat_ws(websocket: WebSocket, conversation_id: UUID):
    # Auth from query param: browser WS can't set headers. Tokens are short-lived
    # access JWTs; prefer a single-use WS ticket for stricter setups (documented).
    token = websocket.query_params.get("token")
    try:
        if not token:
            raise ValueError("missing token")
        sub = get_sub_from_token(token)
        uid = UUID(sub)
    except Exception:
        await websocket.accept()
        await websocket.close(code=4401)
        return

    async with AsyncSessionLocal() as db:
        user = await db.get(User, uid)
        if user is None or not user.is_active:
            await websocket.accept()
            await websocket.close(code=4401)
            return

        # Per-user rate limit on WS turns (non-blocking variant; WS can't 429).
        from backend.middleware.rate_limit import enforce_rate_limit_path

        await enforce_rate_limit_path(f"user:{user.id}", f"/api/v1/chat/ws/{conversation_id}")

        raw = await websocket.receive_text()
        try:
            payload = json.loads(raw)
            content = str(payload.get("content") or "")
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"type": "error", "detail": "invalid json"}))
            await websocket.close(code=4400)
            return

        if not content:
            await websocket.send_text(json.dumps({"type": "error", "detail": "content required"}))
            await websocket.close(code=4400)
            return
        if len(content) > settings.WS_MESSAGE_MAX_CHARS:
            await websocket.send_text(json.dumps({"type": "error", "detail": "message too long"}))
            await websocket.close(code=4400)
            return

        try:
            prep = await chat_service.begin_stream_turn(
                db,
                user_id=user.id,
                conversation_id=conversation_id,
                content=content,
            )
        except ChatBlockedError as e:
            await websocket.send_text(json.dumps({"type": "error", "detail": e.detail}))
            await websocket.close(code=4400)
            return
        except PermissionError:
            await websocket.send_text(json.dumps({"type": "error", "detail": "not found"}))
            await websocket.close(code=4404)
            return

        await db.commit()
        await websocket.send_text(
            json.dumps(
                {"type": "user_message", "message": MessageOut.model_validate(prep.user_message).model_dump(mode="json")}
            )
        )
        # Transparency before tokens: users on the streaming path get the same
        # safety annotation as REST consumers (constrained/crisis level).
        if prep.transparency:
            await websocket.send_text(json.dumps(_transparency_event(prep.transparency)))

        parts: list[str] = []
        async for chunk in chat_service.stream_tokens(
            prep.prompt, prep.crisis_fixed_reply, user_id=user.id
        ):
            parts.append(chunk)
            await websocket.send_text(json.dumps({"type": "token", "text": chunk}))

        full = "".join(parts).strip()
        crisis = prep.crisis_fixed_reply is not None
        async with AsyncSessionLocal() as session:
            assistant = await chat_service.persist_assistant_message(
                session,
                conversation_id=conversation_id,
                full_text=full,
                prompt=prep.prompt,
                crisis_stream=crisis,
            )
            await session.commit()
            await websocket.send_text(
                json.dumps({"type": "done", "message": MessageOut.model_validate(assistant).model_dump(mode="json")})
            )

    await websocket.close()


# Optional GET /stream for short messages — query-based SSE
@router.get("/stream")
async def chat_stream_get(
    request: Request,
    conversation_id: UUID = Query(...),
    message: str = Query(..., min_length=1, max_length=32000),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user_rate_limited),
):
    from starlette.responses import StreamingResponse

    body = ChatRequest(conversation_id=conversation_id, content=message)

    try:
        prep = await chat_service.begin_stream_turn(
            db,
            user_id=user.id,
            conversation_id=body.conversation_id,
            content=body.content,
        )
    except ChatBlockedError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except PermissionError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    await db.commit()

    async def event_gen():
        parts: list[str] = []
        try:
            yield f"data: {json.dumps({'type': 'user_message', 'message': MessageOut.model_validate(prep.user_message).model_dump(mode='json')})}\n\n"
            if prep.transparency:
                yield f"data: {json.dumps(_transparency_event(prep.transparency))}\n\n"
            async for chunk in chat_service.stream_tokens(
                prep.prompt, prep.crisis_fixed_reply, user_id=user.id
            ):
                parts.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"

            full = "".join(parts).strip()
            crisis = prep.crisis_fixed_reply is not None
            async with AsyncSessionLocal() as session:
                assistant = await chat_service.persist_assistant_message(
                    session,
                    conversation_id=body.conversation_id,
                    full_text=full,
                    prompt=prep.prompt,
                    crisis_stream=crisis,
                )
                await session.commit()
                yield f"data: {json.dumps({'type': 'done', 'message': MessageOut.model_validate(assistant).model_dump(mode='json')})}\n\n"
        except Exception:
            # Never leak internal exception text to SSE clients; log with context.
            logger.exception("SSE stream failed for conversation %s", body.conversation_id)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'stream failed'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
