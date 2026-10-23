"""Tiny smoke script for manual load sampling (not a full k6/Locust suite)."""
import asyncio
import os
import sys

import httpx

BASE = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
N = int(os.environ.get("REQUESTS", "20"))


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as client:
        r = await client.get("/health")
        r.raise_for_status()
        ok = 0
        for _ in range(N):
            rr = await client.get("/health")
            if rr.status_code == 200:
                ok += 1
        print(f"health {ok}/{N} ok")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
