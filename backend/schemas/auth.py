from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class UserSignup(BaseModel):
    email: EmailStr
    username: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=128)
    accept_terms: bool = False
    confirm_age_18: bool = False
    region: str | None = Field(default=None, max_length=2, description="ISO 3166-1 alpha-2, e.g. US")

    @model_validator(mode="after")
    def require_consent(self):
        if not self.accept_terms or not self.confirm_age_18:
            raise ValueError("You must accept the terms and confirm you meet the age requirement.")
        return self

    @field_validator("region", mode="before")
    @classmethod
    def upper_region(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().upper()[:2] or None
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class UserPublic(BaseModel):
    id: UUID
    email: EmailStr
    username: str
    is_active: bool = True
    email_verified: bool = False
    is_staff: bool = False
    created_at: datetime
    last_login_at: datetime | None = None
    region: str | None = None
    memory_opt_in: bool = False
    tone_preference: str = "gentle"
    terms_version_accepted: str | None = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_user(cls, u: Any):
        p = u.preferences or {}
        return cls(
            id=u.id,
            email=u.email,
            username=u.username,
            is_active=u.is_active,
            email_verified=u.email_verified,
            is_staff=u.is_staff,
            created_at=u.created_at,
            last_login_at=u.last_login_at,
            region=p.get("region"),
            memory_opt_in=bool(p.get("memory_opt_in", False)),
            tone_preference=str(p.get("tone_preference", "gentle")),
            terms_version_accepted=p.get("terms_version_accepted"),
        )


class UserPreferencesPatch(BaseModel):
    memory_opt_in: bool | None = None
    region: str | None = Field(default=None, max_length=2)
    tone_preference: str | None = None

    @field_validator("tone_preference")
    @classmethod
    def tone_ok(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in ("gentle", "direct"):
            raise ValueError("tone_preference must be 'gentle' or 'direct'")
        return v

    @field_validator("region", mode="before")
    @classmethod
    def upper_region(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().upper()[:2] or None
        return v
