"""Tests for download scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class TestDownloadScripts:
    def test_download_qwen_script_exists(self):
        script = SCRIPTS_DIR / "download_qwen_base.py"
        assert script.exists(), f"Missing script: {script}"

    def test_download_qwen_help(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "download_qwen_base.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "Qwen2.5-0.5B-Instruct" in result.stdout

    def test_download_empathetic_script_exists(self):
        script = SCRIPTS_DIR / "download_empathetic_dialogues.py"
        assert script.exists(), f"Missing script: {script}"

    def test_download_empathetic_help(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "download_empathetic_dialogues.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "EmpatheticDialogues" in result.stdout

    def test_experiment_tracker_help(self):
        script = SCRIPTS_DIR / "experiment_tracker.py"
        assert script.exists()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "Record an experiment run" in result.stdout

    def test_generate_provenance_help(self):
        script = SCRIPTS_DIR / "generate_provenance.py"
        assert script.exists()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "provenance" in result.stdout.lower()
