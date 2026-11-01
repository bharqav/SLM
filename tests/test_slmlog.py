"""Tests for slm-training/slmlog.py (shared pipeline logging)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

import slmlog  # noqa: E402
from slmlog import critical, error, fmt_dur, human, log, phase, warn  # noqa: E402


@pytest.fixture(autouse=True)
def _redirect_log_file(tmp_path, monkeypatch):
    """Keep test output out of the repository's train.log."""
    monkeypatch.setattr(slmlog, "LOG_PATH", str(tmp_path / "train.log"))


class TestLog:
    def test_log_prints_timestamped_message(self, capsys):
        log("hello world")
        out = capsys.readouterr().out
        assert out.startswith("[")
        assert "hello world" in out

    def test_log_appends_to_file(self, tmp_path):
        log("to the file")
        content = (tmp_path / "train.log").read_text(encoding="utf-8")
        assert "to the file" in content

    def test_severity_prefixes(self, capsys):
        warn("careful")
        error("broken")
        critical("fatal")
        out = capsys.readouterr().out
        assert "WARNING:" in out
        assert "ERROR:" in out
        assert "CRITICAL:" in out

    def test_phase_wraps_message_with_rules(self, capsys):
        phase("Stage 1")
        out = capsys.readouterr().out
        assert "Stage 1" in out
        assert "=" * 70 in out


class TestFmtDur:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0s"), (59, "59s"), (60, "1m00s"), (3600, "1h00m"), (-5, "0s")],
    )
    def test_boundaries(self, seconds, expected):
        assert fmt_dur(seconds) == expected


class TestHuman:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (999, "999"),
            (1_000, "1.0K"),
            (1_500_000, "1.5M"),
            (2_000_000_000, "2.0B"),
            (3_000_000_000_000, "3.0T"),
        ],
    )
    def test_units(self, value, expected):
        assert human(value) == expected
