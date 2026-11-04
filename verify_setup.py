#!/usr/bin/env python3
"""Verify project setup: check checkpoints, configs, and placeholder status.

Usage:
    python verify_setup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parent


def check_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def verify() -> int:
    print("[verify_setup] Checking project artifacts...\n")
    errors = []
    warnings = []

    # Check SLM checkpoint.
    #
    # This artifact is ~1 GB and deliberately gitignored: a fresh clone does not
    # have it until `make train` runs. Treat its absence as a warning with
    # guidance rather than a hard error, so the documented fast check
    # (`python harness.py --skip-model`) passes on a clean checkout.
    slm_ckpt = PROJECT_ROOT / "slm-training" / "best_model.pt"
    if not slm_ckpt.exists():
        warnings.append(
            f"missing SLM checkpoint: {slm_ckpt} "
            "(expected on a fresh clone; run `make train` to produce it)"
        )
    else:
        # Check if it's a placeholder JSON or real PyTorch checkpoint
        try:
            with open(slm_ckpt, "rb") as f:
                header = f.read(20)
            if header.startswith(b"{"):
                content = slm_ckpt.read_text(encoding="utf-8", errors="ignore")
                if (
                    "placeholder" in content.lower()
                    or "expected_model_class" in content
                ):
                    warnings.append(f"SLM checkpoint is a placeholder stub: {slm_ckpt}")
                else:
                    warnings.append(
                        f"SLM checkpoint appears to be JSON, not PyTorch: {slm_ckpt}"
                    )
            elif header.startswith(b"PK"):
                print(f"  OK: SLM checkpoint is real PyTorch zip bundle: {slm_ckpt}")
            else:
                warnings.append(f"SLM checkpoint has unexpected format: {slm_ckpt}")
        except Exception as exc:
            warnings.append(f"Could not inspect SLM checkpoint: {exc}")

    # Check empathetic checkpoint
    emp_ckpt = PROJECT_ROOT / "slm-training" / "empathetic_model.pt"
    if not emp_ckpt.exists():
        warnings.append(f"missing empathetic checkpoint: {emp_ckpt}")
    else:
        content = emp_ckpt.read_text(encoding="utf-8", errors="ignore")
        if "placeholder" in content.lower():
            warnings.append("empathetic_model.pt is a placeholder stub")
        else:
            print(f"  OK: empathetic checkpoint exists: {emp_ckpt}")

    # Check Qwen adapter configs
    stage1_cfg = check_json(PROJECT_ROOT / "slm-ft" / "adapter" / "adapter_config.json")
    if stage1_cfg is None:
        warnings.append("missing stage-1 adapter config")
    else:
        assert stage1_cfg.get("peft_type") == "LORA"
        print(
            f"  OK: stage-1 adapter config (r={stage1_cfg.get('r')}, targets={stage1_cfg.get('target_modules')})"
        )

    stage2_cfg = check_json(
        PROJECT_ROOT / "slm-ft" / "adapter_stage2" / "adapter_config.json"
    )
    if stage2_cfg is None:
        warnings.append("missing stage-2 adapter config")
    else:
        assert stage2_cfg.get("peft_type") == "LORA"
        print(
            f"  OK: stage-2 adapter config (r={stage2_cfg.get('r')}, targets={stage2_cfg.get('target_modules')})"
        )

    # Check merged config
    merged_cfg = check_json(PROJECT_ROOT / "slm-ft" / "merged_fp16" / "config.json")
    if merged_cfg is None:
        warnings.append("missing merged_fp16 config")
    else:
        print(
            f"  OK: merged_fp16 config (hidden_size={merged_cfg.get('hidden_size')}, layers={merged_cfg.get('num_hidden_layers')})"
        )

    # Check if merged weights are real or placeholder
    merged_weights = PROJECT_ROOT / "slm-ft" / "merged_fp16" / "model.safetensors"
    if not merged_weights.exists():
        warnings.append("missing merged_fp16 model weights")
    else:
        content = merged_weights.read_text(encoding="utf-8", errors="ignore")
        if content.strip().startswith("{"):
            warnings.append(
                "merged_fp16/model.safetensors is a placeholder JSON, not real weights"
            )
        else:
            print(
                f"  OK: merged_fp16 weights exist ({merged_weights.stat().st_size:,} bytes)"
            )

    # Check behavioral eval
    behavioral = check_json(
        PROJECT_ROOT / "slm-ft" / "adapter_stage2" / "behavioral_eval.json"
    )
    if behavioral is None:
        warnings.append("missing behavioral eval")
    else:
        passed = behavioral.get("passed", False)
        print(
            f"  OK: behavioral eval (passed={passed}, probes={behavioral.get('total_probes')})"
        )

    # Check README.
    #
    # A wheel install has no repository layout (the script lives in
    # site-packages), so a missing README is a warning rather than a hard error.
    # Inside a checkout the check still runs and still requires the placeholder
    # note, which is the property we actually care about.
    readme = PROJECT_ROOT / "README.md"
    if not readme.exists():
        warnings.append(
            "missing README.md (expected when running from an installed wheel; "
            "clone the repository to verify the docs)"
        )
    else:
        content = readme.read_text(encoding="utf-8")
        if "placeholder" not in content.lower():
            errors.append("README does not mention Qwen placeholder status")
        else:
            print(
                f"  OK: README documents placeholder status ({readme.stat().st_size:,} bytes)"
            )

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("All checks passed.")
        return 0
    elif errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    else:
        print(f"\n{len(warnings)} warning(s), all passable")
        return 0


if __name__ == "__main__":
    sys.exit(verify())
