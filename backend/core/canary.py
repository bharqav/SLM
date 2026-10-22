"""Canary traffic routing by stable user hash (release governance)."""

from __future__ import annotations

import hashlib
from uuid import UUID

from backend.core.config import settings


def user_in_canary_cohort(user_id: UUID) -> bool:
    pct = settings.CANARY_TRAFFIC_PERCENT
    if pct <= 0:
        return False
    if pct >= 100:
        return True
    h = hashlib.sha256(str(user_id).encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    return bucket < pct
