"""Tests for training harness and monitoring scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGetLastStep:
    """Tests for log parsing utilities shared by monitor/auto_train scripts."""

    def _write_log(self, tmp_path: Path, lines: list[str]) -> Path:
        log = tmp_path / "train.log"
        log.write_text("\n".join(lines), encoding="utf-8")
        return log

    def test_returns_last_step_number(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log = self._write_log(
            tmp_path,
            [
                "[2024-01-01 00:00:00] Step 0010 | Loss: 4.1234 | Throughput: 900 tokens/s",
                "[2024-01-01 00:00:10] Step 0025 | Loss: 3.9876 | Throughput: 950 tokens/s",
            ],
        )
        monkeypatch.setattr("monitor_training.TRAIN_LOG", log)
        # Import after monkeypatch
        sys.path.insert(0, str(PROJECT_ROOT))
        import monitor_training  # noqa: E402

        assert monitor_training.get_last_step() == 25

    def test_returns_none_for_empty_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log = self._write_log(tmp_path, [])
        monkeypatch.setattr("monitor_training.TRAIN_LOG", log)
        sys.path.insert(0, str(PROJECT_ROOT))
        import monitor_training  # noqa: E402

        assert monitor_training.get_last_step() is None

    def test_returns_none_for_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        missing = tmp_path / "missing.log"
        monkeypatch.setattr("monitor_training.TRAIN_LOG", missing)
        sys.path.insert(0, str(PROJECT_ROOT))
        import monitor_training  # noqa: E402

        assert monitor_training.get_last_step() is None

    def test_ignores_non_step_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log = self._write_log(
            tmp_path,
            [
                "[2024-01-01 00:00:00] Model created (757M params)",
                "[2024-01-01 00:00:01] Resumed from checkpoint_step_500.pt at step 500",
            ],
        )
        monkeypatch.setattr("monitor_training.TRAIN_LOG", log)
        sys.path.insert(0, str(PROJECT_ROOT))
        import monitor_training  # noqa: E402

        assert monitor_training.get_last_step() is None


class TestIsTrainingRunning:
    """Tests for training process detection."""

    def test_returns_false_when_pid_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("monitor_training.TRAIN_PID_FILE", tmp_path / "missing.pid")
        sys.path.insert(0, str(PROJECT_ROOT))
        import monitor_training  # noqa: E402

        assert monitor_training.is_training_running() is False

    def test_returns_false_for_non_existent_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pid_file = tmp_path / "train.pid"
        pid_file.write_text("99999", encoding="utf-8")
        monkeypatch.setattr("monitor_training.TRAIN_PID_FILE", pid_file)
        sys.path.insert(0, str(PROJECT_ROOT))
        import monitor_training  # noqa: E402

        assert monitor_training.is_training_running() is False

    def test_returns_true_for_running_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pid_file = tmp_path / "train.pid"
        pid_file.write_text("0", encoding="utf-8")
        monkeypatch.setattr("monitor_training.TRAIN_PID_FILE", pid_file)
        # Patch OpenProcess to simulate a running process
        with patch("ctypes.windll.kernel32.OpenProcess", return_value=1234):
            with patch("ctypes.windll.kernel32.CloseHandle"):
                sys.path.insert(0, str(PROJECT_ROOT))
                import monitor_training  # noqa: E402

                assert monitor_training.is_training_running() is True

    def test_returns_false_when_openprocess_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        pid_file = tmp_path / "train.pid"
        pid_file.write_text("0", encoding="utf-8")
        monkeypatch.setattr("monitor_training.TRAIN_PID_FILE", pid_file)
        with patch("ctypes.windll.kernel32.OpenProcess", return_value=0):
            sys.path.insert(0, str(PROJECT_ROOT))
            import monitor_training  # noqa: E402

            assert monitor_training.is_training_running() is False


class TestAutoTrainLoop:
    """Tests for auto_train_loop.py."""

    def test_start_training_builds_resume_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("auto_train_loop.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("auto_train_loop.TRAIN_LOG", tmp_path / "train.log")
        monkeypatch.setattr("auto_train_loop.TRAIN_PID_FILE", tmp_path / "train.pid")
        monkeypatch.setattr("auto_train_loop.TRAIN_SCRIPT", tmp_path / "train.py")
        monkeypatch.setattr("auto_train_loop.BEST_EMPATHY", tmp_path / "best.pt")
        (tmp_path / "slm-training").mkdir()
        (tmp_path / "train.py").write_text("", encoding="utf-8")
        sys.path.insert(0, str(PROJECT_ROOT))
        import auto_train_loop  # noqa: E402

        captured: dict = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            mock = MagicMock()
            mock.pid = 1234
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        with patch("subprocess.Popen", side_effect=fake_popen):
            auto_train_loop.start_training(max_steps=10, resume=True)

        assert "--resume" in captured["cmd"]
        assert "--max-steps" in captured["cmd"]
        assert "10" in captured["cmd"]

    def test_start_training_omits_resume_when_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("auto_train_loop.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("auto_train_loop.TRAIN_LOG", tmp_path / "train.log")
        monkeypatch.setattr("auto_train_loop.TRAIN_PID_FILE", tmp_path / "train.pid")
        monkeypatch.setattr("auto_train_loop.TRAIN_SCRIPT", tmp_path / "train.py")
        monkeypatch.setattr("auto_train_loop.BEST_EMPATHY", tmp_path / "best.pt")
        (tmp_path / "slm-training").mkdir()
        (tmp_path / "train.py").write_text("", encoding="utf-8")
        sys.path.insert(0, str(PROJECT_ROOT))
        import auto_train_loop  # noqa: E402

        captured: dict = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            mock = MagicMock()
            mock.pid = 1234
            mock.__enter__ = MagicMock(return_value=mock)
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        with patch("subprocess.Popen", side_effect=fake_popen):
            auto_train_loop.start_training(max_steps=10, resume=False)

        assert "--resume" not in captured["cmd"]

    def test_cleanup_pid_file_removes_stale_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("auto_train_loop.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("auto_train_loop.TRAIN_PID_FILE", tmp_path / "train.pid")
        monkeypatch.setattr("auto_train_loop.is_training_running", lambda: False)
        pid_file = tmp_path / "train.pid"
        pid_file.write_text("1234", encoding="utf-8")
        assert pid_file.exists()
        sys.path.insert(0, str(PROJECT_ROOT))
        import auto_train_loop  # noqa: E402

        auto_train_loop.cleanup_pid_file()
        assert not pid_file.exists()

    def test_cleanup_pid_file_keeps_file_when_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("auto_train_loop.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("auto_train_loop.TRAIN_PID_FILE", tmp_path / "train.pid")
        monkeypatch.setattr("auto_train_loop.is_training_running", lambda: True)
        pid_file = tmp_path / "train.pid"
        pid_file.write_text("1234", encoding="utf-8")
        sys.path.insert(0, str(PROJECT_ROOT))
        import auto_train_loop  # noqa: E402

        auto_train_loop.cleanup_pid_file()
        assert pid_file.exists()


class TestHarnessRunStep:
    """Tests for harness.py run_step utility."""

    def test_run_step_returns_zero_on_success(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT))
        import harness  # noqa: E402

        rc = harness.run_step(
            "echo hello", [sys.executable, "-c", "print('hello')"], cwd=tmp_path
        )
        assert rc == 0

    def test_run_step_returns_nonzero_on_failure(self, tmp_path: Path):
        sys.path.insert(0, str(PROJECT_ROOT))
        import harness  # noqa: E402

        rc = harness.run_step(
            "fail", [sys.executable, "-c", "raise SystemExit(1)"], cwd=tmp_path
        )
        assert rc == 1

    def test_run_step_prints_name(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        sys.path.insert(0, str(PROJECT_ROOT))
        import harness  # noqa: E402

        harness.run_step(
            "test-step", [sys.executable, "-c", "print('ok')"], cwd=tmp_path
        )
        captured = capsys.readouterr()
        assert "test-step" in captured.out


class TestRunAllGuard:
    """Tests for run_all.py's source-checkout guard.

    ``run_all:run`` backs the ``slm-bench`` console script, and the wheel ships
    the ``tests/`` package, so ``PROJECT_ROOT`` resolves to site-packages when
    the script is installed. The guard must report the missing checkout rather
    than fail inside pytest, and must stay silent in a real checkout.
    """

    def test_reports_missing_checkout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        sys.path.insert(0, str(PROJECT_ROOT))
        import run_all  # noqa: E402

        monkeypatch.setattr(run_all, "PROJECT_ROOT", tmp_path)

        assert run_all.run() == 2

        out = capsys.readouterr().out
        assert "source checkout" in out
        assert "slm-verify" in out

    def test_proceeds_when_checkout_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        sys.path.insert(0, str(PROJECT_ROOT))
        import run_all  # noqa: E402

        (tmp_path / "benchmarks").mkdir()
        monkeypatch.setattr(run_all, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            run_all,
            "subprocess",
            MagicMock(run=MagicMock(return_value=MagicMock(returncode=1))),
        )

        assert run_all.run() == 1

        assert "pytest failed" in capsys.readouterr().out
