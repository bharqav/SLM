"""Minimal evaluation gate checker for model releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate release gates.")
    parser.add_argument("--predictions", required=True, help="JSON file with aggregate metrics.")
    parser.add_argument("--thresholds", required=True, help="JSON file with pass/fail thresholds.")
    args = parser.parse_args()

    metrics = _load_json(Path(args.predictions))
    thresholds = _load_json(Path(args.thresholds))

    checks: list[tuple[str, bool, float, float, str]] = [
        ("empathy_score", float(metrics.get("empathy_score", 0.0)) >= float(thresholds["empathy_score_min"]),
         float(metrics.get("empathy_score", 0.0)), float(thresholds["empathy_score_min"]), "min"),
        (
            "non_judgmental_violation_rate",
            float(metrics.get("non_judgmental_violation_rate", 1.0))
            <= float(thresholds["non_judgmental_violation_rate_max"]),
            float(metrics.get("non_judgmental_violation_rate", 1.0)),
            float(thresholds["non_judgmental_violation_rate_max"]),
            "max",
        ),
        ("self_harm_recall", float(metrics.get("self_harm_recall", 0.0)) >= float(thresholds["self_harm_recall_min"]),
         float(metrics.get("self_harm_recall", 0.0)), float(thresholds["self_harm_recall_min"]), "min"),
        (
            "medical_emergency_recall",
            float(metrics.get("medical_emergency_recall", 0.0)) >= float(thresholds["medical_emergency_recall_min"]),
            float(metrics.get("medical_emergency_recall", 0.0)),
            float(thresholds["medical_emergency_recall_min"]),
            "min",
        ),
        (
            "prompt_injection_recall",
            float(metrics.get("prompt_injection_recall", 0.0)) >= float(thresholds["prompt_injection_recall_min"]),
            float(metrics.get("prompt_injection_recall", 0.0)),
            float(thresholds["prompt_injection_recall_min"]),
            "min",
        ),
        (
            "high_severity_false_positive_rate",
            float(metrics.get("high_severity_false_positive_rate", 1.0))
            <= float(thresholds["high_severity_false_positive_rate_max"]),
            float(metrics.get("high_severity_false_positive_rate", 1.0)),
            float(thresholds["high_severity_false_positive_rate_max"]),
            "max",
        ),
        (
            "refusal_correctness",
            float(metrics.get("refusal_correctness", 0.0)) >= float(thresholds["refusal_correctness_min"]),
            float(metrics.get("refusal_correctness", 0.0)),
            float(thresholds["refusal_correctness_min"]),
            "min",
        ),
        (
            "unsupported_claim_rate",
            float(metrics.get("unsupported_claim_rate", 1.0)) <= float(thresholds["unsupported_claim_rate_max"]),
            float(metrics.get("unsupported_claim_rate", 1.0)),
            float(thresholds["unsupported_claim_rate_max"]),
            "max",
        ),
    ]

    failed = [c for c in checks if not c[1]]
    for name, ok, val, th, mode in checks:
        op = ">=" if mode == "min" else "<="
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val:.4f} {op} {th:.4f}")
    if failed:
        print(f"\nRelease gate failed ({len(failed)} checks).")
        return 1
    print("\nAll release gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
