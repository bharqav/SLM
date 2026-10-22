"""Versioned prompt templates — keep model-facing copy out of scattered app logic."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from backend.core.config import settings

_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "prompt_templates"


@lru_cache(maxsize=16)
def load_template(version: str) -> dict:
    path = _ROOT / f"v{version}.json"
    if not path.is_file():
        path = _ROOT / "v1.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": version, "system_preamble": "", "style_directives": {}}


def preamble_for_turn(version: str | None = None) -> str:
    v = version or settings.PROMPT_VERSION
    data = load_template(v)
    return str(data.get("system_preamble", "")).strip()


def style_line(tone: str, version: str | None = None) -> str:
    v = version or settings.PROMPT_VERSION
    data = load_template(v)
    dirs_ = data.get("style_directives") or {}
    return str(dirs_.get(tone, dirs_.get("gentle", ""))).strip()
