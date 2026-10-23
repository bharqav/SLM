"""Evaluation gate for the deterministic safety layer, run in CI.

Measures crisis-policy and injection-backstop recall on a probe corpus and
asserts the same thresholds used for model releases (scripts/training/
thresholds.json); high-severity false positives are measured on the benign set.

LLM-quality metrics (empathy, refusal quality, factuality) remain offline
training gates (docs/TRAINING_RUNBOOK.md §6); this file makes the policy layer
an enforceable release gate instead of documented-but-unenforced.
"""

import json
from pathlib import Path

from backend.services import safety_service
from backend.services.crisis_policy import classify_crisis

_ROOT = Path(__file__).resolve().parent.parent
_PROBES = json.loads((_ROOT / "tests" / "fixtures" / "safety_probes.json").read_text(encoding="utf-8"))
_THRESHOLDS = json.loads((_ROOT / "scripts" / "training" / "thresholds.json").read_text(encoding="utf-8"))

_CASES = _PROBES["cases"]


def _recall(hits: int, total: int) -> float:
    return hits / total if total else 1.0


def _injection_hit(text: str) -> bool:
    _, flags = safety_service._stub_input_check(text)
    return safety_service.should_block_from_flags(flags)


def test_self_harm_recall_gate():
    cases = [c for c in _CASES if c.get("expect_level") == "L3" and c.get("category") == "self_harm"]
    hits = sum(1 for c in cases if classify_crisis(c["text"]).level.value == "L3")
    recall = _recall(hits, len(cases))
    assert recall >= _THRESHOLDS["self_harm_recall_min"], f"self_harm_recall={recall:.3f}"


def test_medical_emergency_recall_gate():
    cases = [c for c in _CASES if c.get("category") == "medical_emergency"]
    hits = sum(1 for c in cases if classify_crisis(c["text"]).level.value == "L3")
    recall = _recall(hits, len(cases))
    assert recall >= _THRESHOLDS["medical_emergency_recall_min"], f"medical_recall={recall:.3f}"


def test_prompt_injection_recall_gate():
    cases = [c for c in _CASES if c.get("inject") is True]
    hits = sum(1 for c in cases if _injection_hit(c["text"]))
    recall = _recall(hits, len(cases))
    assert recall >= _THRESHOLDS["prompt_injection_recall_min"], f"injection_recall={recall:.3f}"


def test_high_severity_false_positive_gate():
    benign = [c for c in _CASES if c.get("posture") == "must_pass"]
    fps = [c["id"] for c in benign if classify_crisis(c["text"]).level.value == "L3"]
    assert not fps, f"Benign probes trip an L3 hard stop: {fps}"
