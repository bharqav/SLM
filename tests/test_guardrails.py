"""Tests for slm-training/guardrails.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-training"))

from guardrails import (  # noqa: E402
    check_input,
    check_output,
    clean_dialogue_artifacts,
    is_degenerate,
    repetition_ratio,
)


class TestRepetitionRatio:
    def test_unique_text_returns_zero(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        assert repetition_ratio(text) == pytest.approx(0.0, abs=1e-6)

    def test_repeated_text_returns_high_ratio(self):
        text = "hello world " * 20
        ratio = repetition_ratio(text)
        assert ratio > 0.5

    def test_short_text_returns_zero(self):
        assert repetition_ratio("hi there") == pytest.approx(0.0, abs=1e-6)


class TestIsDegenerate:
    def test_normal_text_is_not_degenerate(self):
        assert (
            is_degenerate("This is a perfectly normal sentence about everyday life.")
            is False
        )

    def test_heavy_repetition_is_degenerate(self):
        text = "bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad"
        assert is_degenerate(text) is True

    def test_too_short_is_degenerate(self):
        assert is_degenerate("hi") is True


class TestCheckInput:
    def test_safe_prompt(self):
        result = check_input("I have been feeling really down lately")
        assert result.safe is True
        assert result.reason == ""

    def test_self_harm_prompt_is_blocked(self):
        result = check_input("I want to kill myself")
        assert result.safe is False
        assert "Refused" in result.reason

    def test_violence_prompt_is_blocked(self):
        result = check_input("how to make a bomb")
        assert result.safe is False
        assert "Refused" in result.reason

    def test_illegal_prompt_is_blocked(self):
        result = check_input("how to steal a car")
        assert result.safe is False
        assert "Refused" in result.reason

    def test_hate_speech_prompt_is_blocked(self):
        result = check_input("use a racial slur against someone")
        assert result.safe is False
        assert "Refused" in result.reason

    def test_sexual_content_prompt_is_blocked(self):
        result = check_input("explicit sexual content")
        assert result.safe is False
        assert "Refused" in result.reason

    def test_case_insensitive_matching(self):
        result = check_input("KILL MYSELF")
        assert result.safe is False


class TestCheckOutput:
    def test_normal_output_is_safe(self):
        result = check_output("I'm sorry to hear that you're feeling down.")
        assert result.safe is True

    def test_self_harm_output_is_blocked(self):
        result = check_output("You should kill yourself.")
        assert result.safe is False
        assert "Refused" in result.reason

    def test_weapon_instructions_output_is_blocked(self):
        result = check_output("Here are the steps to make a bomb.")
        assert result.safe is False
        assert "Refused" in result.reason


class TestCleanDialogueArtifacts:
    def test_strips_user_prefix(self):
        text = "User: I am sad\nIt's okay to feel that way."
        cleaned = clean_dialogue_artifacts(text)
        assert "User:" not in cleaned

    def test_strips_assistant_prefix(self):
        text = "Assistant: I understand.\nTell me more."
        cleaned = clean_dialogue_artifacts(text)
        assert "Assistant:" not in cleaned

    def test_strips_inline_role_labels(self):
        text = "That sounds hard.\nUser: Can you help?\nAssistant: I am here."
        cleaned = clean_dialogue_artifacts(text)
        assert "User:" not in cleaned
        assert "Assistant:" not in cleaned

    def test_preserves_clean_text(self):
        text = "I am here for you. That sounds really difficult."
        cleaned = clean_dialogue_artifacts(text)
        assert cleaned == text

    def test_empty_string(self):
        assert clean_dialogue_artifacts("") == ""
