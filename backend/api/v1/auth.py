from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.security import create_access_token, hash_password, verify_password
from backend.core.tokens import hash_refresh_token, new_refresh_token
from backend.api.deps import get_current_user
from backend.db.session import get_db
from backend.middleware.rate_limit import enforce_rate_limit_ip
from backend.models import User, UserSession
from backend.api.v1.legal import TERMS_VERSION
from backend.schemas.auth import (
    LogoutRequest,
    RefreshRequest,
    TokenResponse,
    UserLogin,
    UserPreferencesPatch,
    UserPublic,
    UserSignup,
)

import logging

logger = logging.getLogger("security.auth")

router = APIRouter(prefix="/auth", tags=["auth"])


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    request: Request,
    body: UserSignup,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    await enforce_rate_limit_ip(request)
    exists = await db.execute(select(User.id).where(User.email == body.email))
    if exists.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    exists_u = await db.execute(select(User.id).where(User.username == body.username))
    if exists_u.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    prefs: dict = {"terms_version_accepted": TERMS_VERSION}
    if body.region:
        prefs["region"] = body.region
    user = User(
        email=body.email,
        username=body.username,
        password_hash=hash_password(body.password),
        is_staff=bool(
            settings.ADMIN_BOOTSTRAP_EMAIL and body.email == settings.ADMIN_BOOTSTRAP_EMAIL,
        ),
        preferences=prefs,
    )
    db.add(user)
    await db.flush()

    refresh = new_refresh_token()
    exp = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    user.last_login_at = datetime.now(timezone.utc)
    sess = UserSession(
        user_id=user.id,
        token_hash=hash_refresh_token(refresh),
        expires_at=exp,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(sess)
    await db.commit()

    access = create_access_token(user.id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    body: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    await enforce_rate_limit_ip(request, limit=settings.LOGIN_RATE_LIMIT_PER_MINUTE)
    r = await db.execute(select(User).where(User.email == body.email))
    user = r.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    user.last_login_at = datetime.now(timezone.utc)
    refresh = new_refresh_token()
    exp = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    sess = UserSession(
        user_id=user.id,
        token_hash=hash_refresh_token(refresh),
        expires_at=exp,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(sess)
    await db.commit()

    access = create_access_token(user.id)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    request: Request,
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    await enforce_rate_limit_ip(request, limit=settings.LOGIN_RATE_LIMIT_PER_MINUTE)
    th = hash_refresh_token(body.refresh_token)
    r = await db.execute(select(UserSession).where(UserSession.token_hash == th))
    sess = r.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    now = datetime.now(timezone.utc)
    if _aware(sess.expires_at) < now:
        await db.delete(sess)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh expired")

    if sess.revoked_at is not None:
        # Refresh-token REUSE: an already-rotated/revoked token is being replayed.
        # Treat as theft signal — revoke the user's whole session family and 401.
        logger.warning(
            "refresh_token_replay_detected user_id=%s session=%s — revoking all sessions for user",
            sess.user_id,
            sess.id,
        )
        await db.execute(delete(UserSession).where(UserSession.user_id == sess.user_id))
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token reuse detected")

    # Rotation: mark old session revoked (keep it so replay detection stays possible)
    sess.revoked_at = now
    new_tok = new_refresh_token()
    exp = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    new_sess = UserSession(
        user_id=sess.user_id,
        token_hash=hash_refresh_token(new_tok),
        expires_at=exp,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(new_sess)
    await db.commit()

    access = create_access_token(sess.user_id)
    return TokenResponse(access_token=access, refresh_token=new_tok)


@router.post("/logout")
async def logout(
    body: LogoutRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    # Revoke (not delete) so a later replay of this token trips reuse detection.
    th = hash_refresh_token(body.refresh_token)
    now = datetime.now(timezone.utc)
    r = await db.execute(select(UserSession).where(UserSession.token_hash == th))
    sess = r.scalar_one_or_none()
    if sess is not None and sess.revoked_at is None:
        sess.revoked_at = now
        await db.commit()
    return {"status": "ok"}


@router.get("/me", response_model=UserPublic)
async def read_me(current: User = Depends(get_current_user)) -> UserPublic:
    return UserPublic.from_user(current)


@router.patch("/me", response_model=UserPublic)
async def patch_me(
    body: UserPreferencesPatch,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserPublic:
    p = dict(current.preferences or {})
    if body.memory_opt_in is not None:
        p["memory_opt_in"] = body.memory_opt_in
    if body.region is not None:
        p["region"] = body.region
    if body.tone_preference is not None:
        p["tone_preference"] = body.tone_preference
    current.preferences = p
    await db.commit()
    await db.refresh(current)
    return UserPublic.from_user(current)