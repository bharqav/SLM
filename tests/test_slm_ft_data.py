"""Regression tests for slm-ft data shaping helpers (no torch needed)."""

from pathlib import Path

import pytest

prepare = pytest.importorskip(
    "importlib", reason="slm-ft helpers import datasets indirectly"
).import_module

import importlib.util


def _load_prep():
    spec = importlib.util.spec_from_file_location(
        "slmft_prepare_data", Path(__file__).resolve().parents[1] / "slm-ft" / "prepare_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_undoes_comma_encoding():
    mod = _load_prep()
    assert mod.clean("a_comma_b") == "a,b"
    assert mod.clean("  x\t y  ").strip() == "x  y"  # tabs replaced with spaces, stripped


def test_ok_requires_user_and_assistant_turns():
    mod = _load_prep()
    assert mod._ok([{"role": "user", "content": "hi"}] * 5) is False
    assert mod._ok([{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}]) is False  # 1 exchange < 2
    good = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}] * 2
    assert mod._ok(good) is True


def test_trim_user_tail_ends_on_assistant():
    mod = _load_prep()
    turns = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out = mod.trim_user_tail(turns)
    assert out[-1]["role"] == "assistant"
    assert len(out) == 2


def test_hash_split_is_dialog_level_and_stable():
    mod = _load_prep()
    dialogs = [[
        {"role": "user", "content": f"msg {i}"},
        {"role": "assistant", "content": f"reply {i}"},
        {"role": "user", "content": f"msg {i}2"},
        {"role": "assistant", "content": f"reply {i}2"},
    ] for i in range(200)]
    train, val = mod.hash_split(dialogs, val_fraction=0.02)
    assert len(train) + len(val) == 200
    # re-running gives the same partition (no leakage)
    train2, val2 = mod.hash_split(dialogs, val_fraction=0.02)
    assert [d[0]["content"] for d in train] == [d[0]["content"] for d in train2]
    assert not ({d[0]["content"] for d in val} & {d[0]["content"] for d in train})
