"""
Crisis escalation policy (governance §1): red-line intents, levels L0–L4, regional resources.
Not a substitute for clinical risk assessment — patterns are conservative heuristics.

Detection is heuristic and English-only: text is normalized (lowercase, leetspeak
de-obfuscation, homoglyph folding, zero-width stripping) before matching, but
non-English content will fall through to L0 and MUST be covered by a classifier
(see SAFETY_INPUT_HF_URL) or multilingual regex pack before broad deployment.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from backend.core.config import settings

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "crisis_resources.json"


class CrisisCategory(str, Enum):
    NONE = "none"
    SELF_HARM = "self_harm"
    HARM_OTHERS = "harm_others"
    ABUSE = "abuse"
    PSYCHOSIS = "psychosis"
    MEDICAL_EMERGENCY = "medical_emergency"


class CrisisLevel(str, Enum):
    """Assistant behavior tier (governance doc)."""

    L0_ROUTINE = "L0"
    L1_DISTRESS = "L1"
    L2_ELEVATED = "L2"
    L3_CRITICAL = "L3"
    L4_HUMAN_REVIEW = "L4"


@dataclass(frozen=True)
class CrisisAssessment:
    level: CrisisLevel
    category: CrisisCategory
    flags: dict[str, Any]


# --- Normalization -----------------------------------------------------------

_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w", "ѕ": "s",
}
_LEET = str.maketrans({"0": "o", "1": "i", "$": "s", "@": "a", "!": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if not unicodedata.category(ch) == "Cf")  # zero-width / format chars
    t = t.lower().translate(_LEET)
    t = "".join(_HOMOGLYPHS.get(ch, ch) for ch in t)
    t = re.sub(r"u\.?\s*\.?u", "u", t)  # h*u*rt-like splitters collapse
    return t


# Imminent / critical patterns (L3) — block unsafe generation; fixed crisis reply.
_RE_SELF_IMMINENT = re.compile(
    r"\b("
    r"kill(?:ing)?\s+myself|end\s+my\s+life|end\s+it\s+all\s+(?:now|today|tonight)|"
    r"suicide\s+plan|su1cide\s+plan|tak(?:e|ing)\s+(?:my\s+)?own\s+life|"
    r"going\s+to\s+(?:hang|shoot|overdose)|plan(?:ning)?\s+to\s+(?:kill|end)\s+(?:my|it)|"
    r"goodbye\s+(?:everyone|cruel\s+world)|"
    r"(?:will\s+not|won'?t)\s+wake\s+up(?:\s+(?:tomorrow|tonight|again))?|"
    r"don'?t\s+want\s+to\s+wake\s+up"
    r")\b",
    re.I,
)
_RE_HARM_OTHERS = re.compile(
    r"\b(kill\s+(?:him|her|them)|mass\s+shooting|school\s+shooting|"
    r"bomb\s+threat|hurt\s+people|hurt\s+everyone)\b",
    re.I,
)
_RE_MEDICAL = re.compile(
    r"\b("
    r"(?:can'?t|cannot)\s+breathe(?!\s*(?:when|whenever|while|like|because|too\s+hard|from\s+laugh))|"
    r"chest\s+pain|stroke|overdose|overdosing|"
    r"unconscious|unresponsive|bleeding\s+out"
    r")\b",
    re.I,
)
_RE_PSYCHOSIS = re.compile(
    r"\b(voices\s+(?:tell|told|are\s+telling)\s+me\s+to|being\s+controlled\s+by\s+(?:the\s+)?(?:cia|aliens))\b",
    re.I,
)
# Elevated (L2) — resources + de-escalation; still allow model with instructions.
_RE_SELF_ELEVATED = re.compile(
    r"\b("
    r"suicid|self[- ]harm|hurt\s+myself|want\s+to\s+die|"
    r"no\s+reason\s+to\s+live|no\s+point\s+in\s+living|"
    r"better\s+off\s+dead|\bkms\b|wanna\s+die|i\s+hate\s+being\s+alive"
    r")\b",
    re.I,
)
_RE_ABUSE = re.compile(
    r"\b(sexually\s+abused|being\s+beaten|being\s+hit|traffick|raped?\b|raping\s+me)\b",
    re.I,
)
# Distress (L1) — supportive steering; light-touch resources in prompt.
_RE_DISTRESS = re.compile(
    r"\b(hopeless|worthless|overwhelmed|can'?t\s+go\s+on|sobbing|breaking\s+down|"
    r"crying\s+(?:all|every)\s+day|hate\s+my\s+life|so\s+alone)\b",
    re.I,
)

# --- Multilingual patterns (regions served: US, GB, DE, FR, AU, CA) ------------
# Non-English detection is best-effort regex only — a multilingual classifier
# (SAFETY_INPUT_HF_URL) remains the roadmap item for reliable multilingual.

_RE_SELF_IMMINENT_ML = re.compile(
    r"(?:" +
    r"\b(?:ich\s+(?:will|möchte)\s+sterben|ich\s+will\s+mich\s+umbringen|"
    r"selbstmord\s+plan|suizid\s+plan|ich\s+will\s+(?:mein\s+)?leben\s+beenden|"
    r"ich\s+bring(?:e)?\s+mich\s+um|ich\s+will\s+nicht\s+mehr\s+aufwachen)\b" +
    r"|\b(?:je\s+(?:veux|dois)\s+mourir|je\s+veux\s+en\s+finir|"
    r"je\s+vais\s+me\s+(?:suicider|tuer)|plan\s+de\s+suicide|"
    r"je\s+ne\s+veux\s+plus\s+me\s+r[ée]veiller)\b" +
    r")",
    re.I,
)
_RE_SELF_ELEVATED_ML = re.compile(
    r"(?:" +
    r"\b(?:suizid|selbstmord|selbstverletzung|ich\s+verletz(?:e)?\s+mich|"
    r"ich\s+möchte\s+(?:gerne\s+)?tot\s+sein)\b" +
    r"|\b(?:suicide\b|j'?ai\s+pensé\s+au\s+suicide|je\s+me\s+scarifie|"
    r"je\s+me\s+bless(?:e|ai)\s+moi-même|je\s+préfère\s+être\s+morte?)\b" +
    r")",
    re.I,
)
_RE_MEDICAL_ML = re.compile(
    r"(?:" +
    r"\b(?:ich\s+kann\s+nicht\s+atmen|brustschmerzen|hirnschlag|"
    r"überdosis|bewusstlos|verbluten|ich\s+blute\s+stark)\b" +
    r"|\b(?:je\s+ne\s+peux\s+pas\s+respirer|douleur\s+(?:à\s+la\s+)?poitrine|"
    r"avc|surdose|inconscient|je\s+saigne\s+beaucoup)\b" +
    r")",
    re.I,
)


def _load_resources() -> dict[str, Any]:
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("crisis_resources.json failed to load — using bare fallback")
        return {"version": "fallback", "fallback": {}, "regions": {}}


def get_crisis_resources(region: str | None) -> dict[str, Any]:
    """Regional hotlines + global fallback when region unknown."""
    data = _load_resources()
    fb = data.get("fallback", {})
    regions: dict[str, Any] = data.get("regions", {})
    code = (region or "").strip().upper()[:2]
    if code and code in regions:
        return {
            "region": code,
            "hotlines": regions[code],
            "fallback": fb,
            "international_directory": fb.get("international_directory"),
        }
    return {
        "region": None,
        "hotlines": {},
        "fallback": fb,
        "international_directory": fb.get("international_directory"),
        "message": "Region unknown — use local emergency services and international directory.",
    }


def format_hotlines_for_prompt(res: dict[str, Any]) -> str:
    lines: list[str] = []
    for k, v in res.get("hotlines", {}).items():
        lines.append(f"- {v}")
    if not lines and res.get("fallback"):
        lines.append(str(res["fallback"].get("emergency_generic", "")))
    if url := res.get("international_directory"):
        lines.append(f"- Help finder: {url}")
    return "\n".join(x for x in lines if x)


def _base_assessment(text: str) -> CrisisAssessment:
    t = _normalize(text.strip())
    if not t:
        return CrisisAssessment(CrisisLevel.L0_ROUTINE, CrisisCategory.NONE, {})

    if _RE_SELF_IMMINENT.search(t) or _RE_SELF_IMMINENT_ML.search(t):
        return CrisisAssessment(
            CrisisLevel.L3_CRITICAL,
            CrisisCategory.SELF_HARM,
            {"crisis_pattern": "imminent_self_harm"},
        )
    if _RE_MEDICAL.search(t) or _RE_MEDICAL_ML.search(t):
        return CrisisAssessment(
            CrisisLevel.L3_CRITICAL,
            CrisisCategory.MEDICAL_EMERGENCY,
            {"crisis_pattern": "medical_emergency"},
        )
    if _RE_HARM_OTHERS.search(t):
        return CrisisAssessment(
            CrisisLevel.L3_CRITICAL,
            CrisisCategory.HARM_OTHERS,
            {"crisis_pattern": "violence_threat"},
        )
    if _RE_PSYCHOSIS.search(t):
        return CrisisAssessment(
            CrisisLevel.L2_ELEVATED,
            CrisisCategory.PSYCHOSIS,
            {"crisis_pattern": "psychosis_signal"},
        )
    if _RE_ABUSE.search(t):
        return CrisisAssessment(
            CrisisLevel.L2_ELEVATED,
            CrisisCategory.ABUSE,
            {"crisis_pattern": "abuse_disclosure"},
        )
    if _RE_SELF_ELEVATED.search(t) or _RE_SELF_ELEVATED_ML.search(t):
        return CrisisAssessment(
            CrisisLevel.L2_ELEVATED,
            CrisisCategory.SELF_HARM,
            {"crisis_pattern": "elevated_distress"},
        )
    if _RE_DISTRESS.search(t):
        return CrisisAssessment(
            CrisisLevel.L1_DISTRESS,
            CrisisCategory.NONE,
            {"crisis_pattern": "general_distress"},
        )
    return CrisisAssessment(CrisisLevel.L0_ROUTINE, CrisisCategory.NONE, {})


def classify_crisis(text: str) -> CrisisAssessment:
    """Classify with L3→L4 promotion to the human-review tier when enabled (governance §1)."""
    a = _base_assessment(text)
    if a.level == CrisisLevel.L3_CRITICAL and settings.CRISIS_HUMAN_REVIEW_ENABLED:
        return CrisisAssessment(
            CrisisLevel.L4_HUMAN_REVIEW,
            a.category,
            {**a.flags, "human_review": True},
        )
    return a


def build_crisis_reply(assessment: CrisisAssessment, region: str | None) -> str:
    """Fixed assistant text for L3/L4 (no LLM)."""
    res = get_crisis_resources(region)
    hotlines = format_hotlines_for_prompt(res)
    base = (
        "I'm really concerned about your safety. This service can't provide crisis counseling or emergency help.\n\n"
        "If you're in immediate danger, contact local emergency services now.\n\n"
    )
    if hotlines:
        base += "Crisis resources you can try:\n" + hotlines + "\n\n"
    base += (
        "You deserve support from trained responders. Please reach out to a crisis line or someone you trust."
    )
    return base.strip()


def maybe_log_human_review(user_id: str, assessment: CrisisAssessment, request_id: str) -> None:
    if not settings.CRISIS_HUMAN_REVIEW_ENABLED:
        return
    if assessment.level not in (CrisisLevel.L3_CRITICAL, CrisisLevel.L4_HUMAN_REVIEW):
        return
    logger.warning(
        "crisis_human_review_queue user_id=%s level=%s category=%s request_id=%s",
        user_id,
        assessment.level.value,
        assessment.category.value,
        request_id,
    )


def crisis_prompt_prefix(assessment: CrisisAssessment, region: str | None) -> str:
    """Instruction block for L1/L2 when still using the model."""
    if assessment.level not in (CrisisLevel.L1_DISTRESS, CrisisLevel.L2_ELEVATED):
        return ""
    res = get_crisis_resources(region)
    hotlines = format_hotlines_for_prompt(res)
    return (
        "[Crisis_instructions: Stay calm and short. Do not give medical or method advice. "
        "Validate feelings. Encourage professional and crisis support. "
        "If the user may be in immediate danger, urge emergency services and crisis lines.]\n"
        f"[Regional_resources]\n{hotlines}\n"
    )
