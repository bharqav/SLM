"""PII-aware redaction before text is embedded or logged to long-term memory."""

from __future__ import annotations

import re

# Heuristic patterns — tune per locale; not a guarantee of removal.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def redact_for_embedding(text: str) -> str:
    """Strip likely PII before embedding / indexing."""
    t = text
    t = _EMAIL.sub("[email-redacted]", t)
    t = _PHONE.sub("[phone-redacted]", t)
    t = _SSN.sub("[id-redacted]", t)
    t = _CC.sub("[card-redacted]", t)
    return t
