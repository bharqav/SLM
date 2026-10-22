"""OAuth2 roadmap (PRD: JWT + OAuth2). Authorization-code flows are not implemented."""

from fastapi import APIRouter, HTTPException, status

from backend.core.config import settings

router = APIRouter(prefix="/auth/oauth", tags=["oauth"])


@router.get("/status")
def oauth_status() -> dict[str, str | bool]:
    return {
        "google_configured": bool(settings.GOOGLE_OAUTH_CLIENT_ID and settings.GOOGLE_OAUTH_CLIENT_SECRET),
        "note": "OAuth2 authorize/callback is not implemented in this build; use POST /api/v1/auth/login.",
    }


@router.get("/google/authorize")
def oauth_google_authorize() -> None:
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Google OAuth2 is not implemented. Use email/password at POST /api/v1/auth/login.",
    )


@router.post("/google/callback")
def oauth_google_callback() -> None:
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail="Google OAuth2 callback is not implemented.",
    )
