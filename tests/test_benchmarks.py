"""Tests for benchmark and comparison artifacts.

Run:
    python -m pytest tests/test_benchmarks.py -v
"""

from __future__ import annotations

import json
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def comparison_json():
    path = PROJECT_ROOT / "slm-comparison" / "comparison_samples.json"
    assert path.exists(), f"missing artifact: {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture()
def benchmark_report():
    path = PROJECT_ROOT / "benchmarks" / "benchmark_report.md"
    assert path.exists(), f"missing artifact: {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def readme():
    path = PROJECT_ROOT / "README.md"
    assert path.exists(), f"missing artifact: {path}"
    return path.read_text(encoding="utf-8")


class TestComparisonArtifact:
    def test_top_level_keys(self, comparison_json):
        assert "generated_at" in comparison_json
        assert "config" in comparison_json
        assert "tiers" in comparison_json
        assert "samples" in comparison_json

    def test_expected_tiers(self, comparison_json):
        tiers = comparison_json["tiers"]
        assert "slm_code" in tiers
        assert "qwen_base" in tiers

    def test_slm_tier_loaded(self, comparison_json):
        tier = comparison_json["tiers"]["slm_code"]
        assert tier["loaded"] is True
        assert tier["params"] > 0
        assert "step" in tier
        assert "best_val_loss" in tier

    def test_qwen_tier_loaded(self, comparison_json):
        tier = comparison_json["tiers"]["qwen_base"]
        assert tier["loaded"] is True
        assert tier["params"] > 0

    def test_samples_count(self, comparison_json):
        samples = comparison_json["samples"]
        assert len(samples) == 2
        models = {s["model"] for s in samples}
        assert "slm-scratch" in models
        assert "qwen2.5-3b-base" in models

    def test_prompt_coverage(self, comparison_json):
        expected_prompts = {
            "I have been feeling really down lately",
            "The best way to handle stress is",
            "I do not think I can keep going",
            "I just need someone to listen to me",
            "I'm scared about my future",
            "I feel like nobody understands me",
            "What should I do when I feel overwhelmed?",
            "I've been having trouble sleeping at night",
            "I lost someone important to me",
            "Here is how the model works:",
            "def preprocess(text):",
            "return {",
            "async function fetchUserData(",
            "class ChatMessage:",
            "const response = await fetch(",
            "def train_model(dataset):",
            "import torch.nn as nn",
            "# TODO: implement this function",
            "if __name__ == '__main__':",
            "SELECT * FROM users WHERE",
        }
        seen = set()
        for block in comparison_json["samples"]:
            for gen in block["generations"]:
                seen.add(gen["prompt"])
        assert expected_prompts.issubset(seen)

    def test_generation_metrics_present(self, comparison_json):
        for block in comparison_json["samples"]:
            for gen in block["generations"]:
                assert "prompt_tokens" in gen
                assert "completion_tokens" in gen
                assert "latency_s" in gen
                assert "tokens_per_second" in gen
                assert gen["completion_tokens"] > 0
                assert gen["latency_s"] > 0
                assert gen["tokens_per_second"] > 0

    def test_seed_present(self, comparison_json):
        assert "seed" in comparison_json
        assert isinstance(comparison_json["seed"], int)


class TestBenchmarkReport:
    def test_forward_pass_table_present(self, benchmark_report):
        assert "Forward-Pass Throughput Benchmarks" in benchmark_report
        assert "SLM from-scratch" in benchmark_report and "10L" in benchmark_report
        assert "Qwen2.5-0.5B-Instruct" in benchmark_report

    def test_generation_table_present(self, benchmark_report):
        assert "Text Generation Benchmarks" in benchmark_report
        assert "Mean Throughput (tok/s)" in benchmark_report

    def test_qwen_proxy_disclosure(self, benchmark_report):
        assert "placeholder" in benchmark_report.lower()
        assert "proxy" in benchmark_report.lower()

    def test_reproducibility_command(self, benchmark_report):
        assert "generate_live_comparison.py" in benchmark_report


class TestReadme:
    def test_sections_present(self, readme):
        expected_sections = [
            "Project Structure",
            "From-Scratch SLM",
            "Qwen2.5-0.5B Comparison Baseline",
            "Comparison Methodology",
            "Data Pipeline",
            "Architecture Diagrams",
            "Training Details",
            "Inference & Generation",
            "Benchmarks",
            "What We Tried / Kept / Cut",
            "License",
            "Reproducing",
        ]
        for section in expected_sections:
            assert section in readme, f"missing README section: {section}"

    def test_mermaid_diagrams_present(self, readme):
        assert readme.count("```mermaid") >= 6

    def test_honest_caveats(self, readme):
        assert "placeholder" in readme.lower()
        assert "proxy" in readme.lower()

    def test_new_sections_present(self, readme):
        expected_sections = [
            "Testing & Quality",
            "Reproducibility",
            "Notebooks & Analysis",
            "Docker Deployment",
            "API Server",
        ]
        for section in expected_sections:
            assert section in readme, f"missing README section: {section}"

    def test_badges_present(self, readme):
        assert "![Python]" in readme
        assert "![License]" in readme
        assert "![CI]" in readme

    def test_new_docs_referenced(self, readme):
        assert "CONTRIBUTING.md" in readme
        assert "SECURITY.md" in readme
        assert "CHANGELOG.md" in readme
