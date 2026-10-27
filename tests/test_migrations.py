"""Alembic migration-chain tests.

The migration chain previously had no coverage and was broken in two ways that
no test caught: a ``down_revision`` pointing at a file name instead of a
revision id, and migrations using ``op`` without importing it. Both failures
happen before any database is touched, so they break every ``alembic upgrade
head`` run regardless of database.

These tests render the full upgrade path in Alembic's offline mode
(``--sql``) against the PostgreSQL dialect the project actually deploys on.
Offline mode needs no database server, so the suite stays deterministic in CI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Offline mode never connects; the URL only selects the SQL dialect.
POSTGRES_URL = "postgresql+asyncpg://slm:slm@localhost:5432/slm"

EXPECTED_TABLES = (
    "users",
    "conversations",
    "sessions",
    "messages",
    "feedback",
    "memory_chunks",
    "audit_logs",
)


def run_alembic(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "DATABASE_URL": POSTGRES_URL}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_upgrade_renders_complete_postgres_ddl() -> None:
    """`alembic upgrade head --sql` must render the whole chain without error."""
    proc = run_alembic("upgrade", "head", "--sql")

    assert proc.returncode == 0, (
        f"alembic upgrade failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    sql = proc.stdout
    for table in EXPECTED_TABLES:
        assert f"CREATE TABLE {table}" in sql, f"migration did not create table {table!r}"

    # The chain has to walk every revision, not stop at an earlier head.
    for revision in ("001", "002", "003", "004", "005"):
        assert revision in sql, f"revision {revision} was never applied"

    assert "audit_logs" in sql, "final revision 005 (audit_logs) missing"


def test_history_is_a_single_linear_chain() -> None:
    """Every migration must chain to exactly one parent, ending at 005."""
    proc = run_alembic("history")

    assert proc.returncode == 0, (
        f"alembic history failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "005 (head)" in proc.stdout, (
        f"expected 005 to be the single head:\n{proc.stdout}"
    )
    # A branched history would show more than one head entry.
    assert proc.stdout.count("(head)") == 1, (
        f"expected exactly one head revision:\n{proc.stdout}"
    )
