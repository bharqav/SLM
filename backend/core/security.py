from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from passlib.context import CryptContext

from backend.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWTError = jwt.PyJWTError


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str | UUID, extra: dict[str, Any] | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode: dict[str, Any] = {"sub": str(subject), "exp": expire, "typ": "access", "iat": datetime.now(timezone.utc)}
    if extra:
        to_encode.update(extra)
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    # algorithms is an explicit allow-list: rejects alg-confusion and asymmetric-key
    # HMAC-forgery attempts (CVE-2024-33663 class).
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])


def get_sub_from_token(token: str) -> str:
    try:
        payload = decode_token(token)
        if payload.get("typ") != "access":
            raise JWTError("not an access token")
        sub = payload.get("sub")
        if not sub or not isinstance(sub, str):
            raise JWTError("missing sub")
        return sub
    except JWTError as e:
        raise ValueError("invalid token") from e
