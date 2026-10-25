#!/usr/bin/env python3
"""Verify that every relative link in the repository's Markdown files resolves.

Links to other ecosystems (http/https/mailto) and in-page anchors are skipped;
anything else must point at a file that exists. Exit code 0 means no broken
relative links, 1 means at least one target is missing.

Usage:
    python scripts/check_links.py
"""

from __future__ import annotations

import os
import re
import sys

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".next", "dist", "build"}
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def iter_markdown(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def check(root: str) -> list[str]:
    broken: list[str] = []
    for path in iter_markdown(root):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for match in LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            resolved = os.path.normpath(os.path.join(os.path.relpath(os.path.dirname(path), root), relative))
            if not os.path.exists(os.path.join(root, resolved)):
                broken.append(f"{os.path.relpath(path, root)} -> {target}")
    return broken


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    broken = check(root)
    if broken:
        print(f"BROKEN relative links ({len(broken)}):")
        for item in broken:
            print(f"  {item}")
        return 1
    print("All relative Markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
