from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_evaluate_standard():
    sys_path = str(PROJECT_ROOT / "benchmarks")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import evaluate_standard

    return evaluate_standard


class TestEvaluateStandard:
    def test_import_has_expected_callables(self):
        mod = _import_evaluate_standard()
        assert hasattr(mod, "evaluate")
        assert hasattr(mod, "main")
        assert hasattr(mod, "STANDARD_BENCHMARKS")

    def test_standard_benchmarks_list(self):
        mod = _import_evaluate_standard()
        expected = {
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "piqa",
            "commonsense_qa",
        }
        assert set(mod.STANDARD_BENCHMARKS) == expected

    @patch("evaluate_standard._load_slm")
    @patch("evaluate_standard._run_fallback_eval")
    def test_evaluate_slm_success(self, mock_eval, mock_load):
        mock_load.return_value = {
            "name": "slm-scratch",
            "load_success": True,
            "load_latency_ms": 100.0,
            "model_ref": MagicMock(),
            "enc_ref": MagicMock(),
            "device": "cpu",
            "checkpoint": "best_model.pt",
        }
        mock_eval.return_value = {
            "success": True,
            "results": {
                "hellaswag": {
                    "success": True,
                    "accuracy": 0.5,
                    "correct": 50,
                    "total": 100,
                    "errors": 0,
                    "time_s": 1.0,
                },
                "winogrande": {
                    "success": True,
                    "accuracy": 0.6,
                    "correct": 60,
                    "total": 100,
                    "errors": 0,
                    "time_s": 1.0,
                },
            },
        }

        mod = _import_evaluate_standard()
        report = mod.evaluate(
            model_name="slm-scratch",
            checkpoint="best_model.pt",
            tasks=["hellaswag", "winogrande"],
            num_fewshot=0,
            batch_size=8,
            device="cpu",
        )

        assert report["status"] == "ok"
        assert report["benchmarks"]["hellaswag"]["accuracy"] == 0.5
        assert report["benchmarks"]["winogrande"]["correct"] == 60

    @patch("evaluate_standard._load_slm")
    def test_evaluate_model_load_failure(self, mock_load):
        mock_load.return_value = {
            "name": "slm-scratch",
            "load_success": False,
            "error": "checkpoint missing",
        }

        mod = _import_evaluate_standard()
        report = mod.evaluate(
            model_name="slm-scratch",
            checkpoint="missing.pt",
            tasks=["hellaswag"],
            num_fewshot=0,
            batch_size=8,
            device="cpu",
        )

        assert report["status"] == "error"
        assert "checkpoint missing" in report["error"]

    @patch("evaluate_standard._load_slm")
    def test_evaluate_fallback_eval_failure(self, mock_load):
        mock_load.return_value = {
            "name": "slm-scratch",
            "load_success": True,
            "load_latency_ms": 100.0,
            "model_ref": MagicMock(),
            "enc_ref": MagicMock(),
            "device": "cpu",
        }

        with patch(
            "evaluate_standard._run_fallback_eval",
            return_value={
                "success": False,
                "error": "datasets not installed",
                "results": {},
            },
        ):
            mod = _import_evaluate_standard()
            report = mod.evaluate(
                model_name="slm-scratch",
                checkpoint="best_model.pt",
                tasks=["hellaswag"],
                num_fewshot=0,
                batch_size=8,
                device="cpu",
            )
        assert report["status"] == "error"
        assert "datasets not installed" in report["error"]

    def test_main_writes_json_output(self, tmp_path: Path):
        out_file = tmp_path / "eval.json"
        mod = _import_evaluate_standard()

        saved_argv = mod.sys.argv[:]
        try:
            mod.sys.argv = [
                "evaluate_standard.py",
                "--output",
                str(out_file),
            ]
            with patch.object(
                mod,
                "evaluate",
                return_value={
                    "generated_at": "2024-01-01T00:00:00+00:00",
                    "config": {
                        "model_name": "slm-scratch",
                        "checkpoint": "best_model.pt",
                        "tasks": ["hellaswag"],
                        "num_fewshot": 0,
                        "batch_size": 8,
                        "device": "cpu",
                        "limit": None,
                    },
                    "tiers": {
                        "slm-scratch": {
                            "load_success": True,
                            "load_latency_ms": 100.0,
                            "error": None,
                            "device": "cpu",
                        }
                    },
                    "benchmarks": {
                        "hellaswag": {
                            "success": True,
                            "accuracy": 0.5,
                            "correct": 50,
                            "total": 100,
                            "errors": 0,
                            "time_s": 1.0,
                        }
                    },
                    "status": "ok",
                },
            ):
                rc = mod.main()

            assert rc == 0
            assert out_file.exists()
            data = json.loads(out_file.read_text(encoding="utf-8"))
            assert data["status"] == "ok"
            assert data["benchmarks"]["hellaswag"]["accuracy"] == 0.5
        finally:
            mod.sys.argv = saved_argv

    def test_format_example_hellaswag(self):
        mod = _import_evaluate_standard()
        ex = {
            "ctx": "A man is sitting on a chair",
            "endings": [".", ",", "!", "?"],
            "label": "1",
        }
        ctx, choices, label_idx = mod._format_example("hellaswag", ex)
        assert ctx == "A man is sitting on a chair"
        assert choices == [".", ",", "!", "?"]
        assert label_idx == 1

    def test_format_example_winogrande(self):
        mod = _import_evaluate_standard()
        ex = {
            "sentence": "The _ is sitting on the chair",
            "option1": "man",
            "option2": "woman",
            "answer": "2",
        }
        ctx, choices, label_idx = mod._format_example("winogrande", ex)
        assert ctx == "The _ is sitting on the chair"
        assert choices == ["man", "woman"]
        assert label_idx == 1

    def test_format_example_arc_challenge(self):
        mod = _import_evaluate_standard()
        ex = {
            "question": "What is 2+2?",
            "choices": {
                "text": ["3", "4", "5", "6"],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "2",
        }
        ctx, choices, label_idx = mod._format_example("arc_challenge", ex)
        assert choices == ["3", "4", "5", "6"]
        assert label_idx == 1

    def test_format_example_piqa(self):
        mod = _import_evaluate_standard()
        ex = {
            "goal": "cook an egg",
            "sol1": "boil it",
            "sol2": "freeze it",
            "label": "0",
        }
        ctx, choices, label_idx = mod._format_example("piqa", ex)
        assert choices == ["boil it", "freeze it"]
        assert label_idx == 0

    def test_format_example_commonsense_qa(self):
        mod = _import_evaluate_standard()
        ex = {
            "question": "What is the capital of France?",
            "choices": {
                "text": ["London", "Paris", "Berlin", "Madrid", "Rome"],
                "label": ["A", "B", "C", "D", "E"],
            },
            "answerKey": "B",
        }
        ctx, choices, label_idx = mod._format_example("commonsense_qa", ex)
        assert choices == ["London", "Paris", "Berlin", "Madrid", "Rome"]
        assert label_idx == 1
