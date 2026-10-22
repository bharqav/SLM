"""Clinical / legal copy and product scope (governance §2)."""

from fastapi import APIRouter

router = APIRouter(prefix="/legal", tags=["legal"])

TERMS_VERSION = "2026-04-12"


@router.get("/disclaimers")
def get_disclaimers() -> dict[str, str | list[str]]:
    return {
        "terms_version": TERMS_VERSION,
        "product_scope": (
            "This service offers emotional support and general wellness conversation only. "
            "It does not provide medical advice, diagnosis, or treatment, and is not a substitute "
            "for licensed professionals or emergency services."
        ),
        "mandatory_disclaimers": [
            "If you are in immediate danger, contact local emergency services or a crisis line now.",
            "Responses may be inaccurate or incomplete; do not rely on this service for safety-critical decisions.",
        ],
        "age_policy": "Accounts are intended for users 18+ unless a supervised minor program exists in your region.",
        "prohibited_uses": [
            "Generating instructions for violence, self-harm, illegal acts, or sexual content involving minors.",
            "Harassment, impersonation, or attempting to bypass safety systems.",
        ],
        "clinician_review": (
            "Optional enterprise deployments may enable human review of flagged conversations under contract; "
            "this MVP does not guarantee clinician review of every message."
        ),
    }


@router.get("/consent-copy")
def consent_copy() -> dict[str, str]:
    return {
        "signup_checkbox_terms": "I accept the Terms of Service and Privacy Policy.",
        "signup_checkbox_age": "I confirm I am at least 18 years old (or have required parental consent).",
    }
