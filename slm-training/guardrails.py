"""
Guardrails for SLM inference: input filtering, output filtering, and
degeneracy checks.

Design
------
- No external dependencies beyond the stdlib so this can be imported from
  `inference.py`, `slm_api.py`, and benchmark scripts without adding a new
  runtime requirement.
- Patterns are intentionally conservative and user-facing; they block clear
  harmful/unsafe requests rather than attempting nuanced content moderation.
- All checks are local and deterministic; no network calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    safe: bool
    reason: str = ""


# Input filters: prompts that should be refused before generation.
_INPUT_PATTERNS = [
    # Self-harm / suicide
    re.compile(r"kill (myself|me)|suicide|self[- ]harm|end my life", re.I),
    # Violence / harm to others
    re.compile(
        r"how to (make|build|create|construct).*?(bomb|weapon|poison|explosive)|"
        r"(hurt|harm|attack|kill|murder|assault)\s+(someone|people|others|a person)",
        re.I,
    ),
    # Illegal activity
    re.compile(
        r"how to (steal|hack|fraud|scam|launder|counterfeit)|"
        r"(make|sell|buy|distribute)\s+(drugs|cocaine|heroin|meth|fentanyl)",
        re.I,
    ),
    # Hate / harassment
    re.compile(
        r"(racial|ethnic|religious|gender|sexual) slur|"
        r"(hate|harass|threaten)\s+(someone|people|others|a person)\s+because\s+of",
        re.I,
    ),
    # Sexual content
    re.compile(
        r"explicit\s+(sexual|pornographic)|"
        r"(rape|sexual\s+assault|child\s+sexual)",
        re.I,
    ),
]


# Output filters: generated text that should be blocked after generation.
_OUTPUT_PATTERNS = [
    re.compile(
        r"kill (myself|me|yourself|self)|suicide|self[- ]harm|end my life", re.I
    ),
    re.compile(
        r"(how to|steps to|instructions for).*?(make|build|create).*?(bomb|weapon|poison|explosive)",
        re.I,
    ),
    re.compile(
        r"(racial|ethnic|religious|gender|sexual) slur|"
        r"(hate|harass|threaten)\s+(someone|people|others|a person)\s+because\s+of",
        re.I,
    ),
]


def check_input(prompt: str) -> GuardrailResult:
    """Return whether a prompt is safe to send to the model."""
    for pat in _INPUT_PATTERNS:
        if pat.search(prompt):
            return GuardrailResult(
                safe=False,
                reason="Refused: this request may involve harmful or unsafe content.",
            )
    return GuardrailResult(safe=True, reason="")


def check_output(text: str) -> GuardrailResult:
    """Return whether generated text should be shown to the user."""
    for pat in _OUTPUT_PATTERNS:
        if pat.search(text):
            return GuardrailResult(
                safe=False,
                reason="Refused: model output may contain harmful or unsafe content.",
            )
    return GuardrailResult(safe=True, reason="")


def repetition_ratio(text: str) -> float:
    """Fraction of repeated 4-grams as a cheap degeneracy signal."""
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def is_degenerate(text: str, max_repetition: float = 0.5, min_length: int = 8) -> bool:
    """Flag degenerate outputs: excessive repetition or very short empty responses."""
    words = text.split()
    if len(words) < min_length:
        return True
    return repetition_ratio(text) > max_repetition
