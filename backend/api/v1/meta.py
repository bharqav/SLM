"""SLO reference and governance metadata."""

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/meta", tags=["meta"])

_SLO_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "slo_targets.json"
_RELEASE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "release_registry.json"


@router.get("/slo")
def get_slo_targets() -> dict:
    try:
        with open(_SLO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"error": "slo_targets.json unavailable", "version": "fallback"}


@router.get("/governance")
def governance_index() -> dict[str, str]:
    return {
        "full_policy_doc": "docs/GOVERNANCE_AND_OPERATIONS.md",
        "prompt_version_env": "PROMPT_VERSION",
        "safety_ruleset_env": "SAFETY_RULESET_VERSION",
    }


@router.get("/releases")
def release_registry() -> dict:
    try:
        with open(_RELEASE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"error": "release_registry.json unavailable"}
