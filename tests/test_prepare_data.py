"""Tests for slm-ft/prepare_data.py -- the empathetic-SFT corpus builder.

Covers the pure-stdlib paths (clean, _ok, trim_user_tail, hash_split, build_ed)
and the output contract of main(). build_esconv() needs the optional `datasets`
package plus network access, so it is covered by the recorded run in
slm-ft/README.md rather than here.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-ft"))

from prepare_data import (  # noqa: E402
    _ok,
    build_ed,
    clean,
    hash_split,
    main,
    trim_user_tail,
)

ED_HEADER = [
    "conv_id",
    "utterance_idx",
    "context",
    "prompt",
    "speaker_idx",
    "utterance",
    "selfeval",
    "tags",
]

# conv_id -> [(utterance_idx, speaker_idx, utterance, selfeval), ...]
_Convs = dict[str, list[tuple[int, int, str, str]]]


def _turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _dialog(marker: str) -> list[dict]:
    """A valid 4-turn dialog tagged with `marker` for leak detection."""
    return [
        _turn("user", f"{marker} u1"),
        _turn("assistant", f"{marker} a1"),
        _turn("user", f"{marker} u2"),
        _turn("assistant", f"{marker} a2"),
    ]


def _write_ed_csv(path: Path, convs: _Convs) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(ED_HEADER)
        for conv_id, rows in convs.items():
            for idx, spk, text, selfeval in rows:
                writer.writerow(
                    [
                        conv_id,
                        str(idx),
                        "sad",
                        "the prompt",
                        str(spk),
                        text,
                        selfeval,
                        "",
                    ]
                )


def _write_corpus(csv_dir: Path, convs: _Convs) -> None:
    """build_ed() needs train/valid/test.csv; mirror `convs` into all three."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        _write_ed_csv(csv_dir / f"{name}.csv", convs)


def test_clean_restores_comma_placeholder() -> None:
    # `clean` substitutes the token for a comma but does not reflow the space
    # around it, so "a _comma_ b" becomes "a , b".
    assert clean("sad _comma_ tired") == "sad , tired"


def test_clean_collapses_whitespace_and_strips() -> None:
    assert clean("\tHello\tworld  ") == "Hello world"


def test_ok_requires_two_turns_of_each_role() -> None:
    assert _ok([_turn("user", "a"), _turn("assistant", "b")]) is False
    assert (
        _ok(
            [
                _turn("user", "a"),
                _turn("assistant", "b"),
                _turn("user", "c"),
                _turn("assistant", "d"),
            ]
        )
        is True
    )


def test_trim_user_tail_drops_trailing_user_turns() -> None:
    dialog = [
        _turn("user", "a"),
        _turn("assistant", "b"),
        _turn("user", "c"),
        _turn("user", "d"),
    ]
    assert trim_user_tail(dialog) == [_turn("user", "a"), _turn("assistant", "b")]


def test_trim_user_tail_keeps_assistant_final() -> None:
    dialog = [_turn("user", "a"), _turn("assistant", "b")]
    assert trim_user_tail(dialog) == dialog


def test_trim_user_tail_of_empty_dialog() -> None:
    assert trim_user_tail([]) == []


def test_hash_split_is_deterministic_and_exhaustive() -> None:
    dialogs = [_dialog(f"d{i}") for i in range(50)]
    train_a, val_a = hash_split(dialogs)
    train_b, val_b = hash_split(dialogs)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) + len(val_a) == len(dialogs)


def test_hash_split_never_leaks_a_dialog_into_both_splits() -> None:
    dialogs = [_dialog(f"d{i}") for i in range(200)]
    train, val = hash_split(dialogs)
    train_keys = {json.dumps(d, sort_keys=True) for d in train}
    val_keys = {json.dumps(d, sort_keys=True) for d in val}
    assert train_keys.isdisjoint(val_keys)


def test_hash_split_honours_val_fraction() -> None:
    dialogs = [_dialog(f"d{i}") for i in range(1000)]
    _, val = hash_split(dialogs, val_fraction=0.10)
    assert len(val) / len(dialogs) == pytest.approx(0.10, abs=0.02)


def test_build_ed_sorts_turns_and_assigns_roles(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        {
            "c1": [
                (2, 0, "second user line", ""),
                (0, 0, "first user line", ""),
                (3, 1, "second support line", ""),
                (1, 1, "first support line", ""),
            ],
        },
    )
    expected = [
        _turn("user", "first user line"),
        _turn("assistant", "first support line"),
        _turn("user", "second user line"),
        _turn("assistant", "second support line"),
    ]
    dialogs = build_ed(tmp_path)
    assert len(dialogs) == 3
    assert all(d == expected for d in dialogs)


def test_build_ed_merges_consecutive_assistant_turns(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        {
            "c1": [
                (0, 0, "first user line", ""),
                (1, 1, "first support line", ""),
                (2, 0, "second user line", ""),
                (3, 1, "second support line", ""),
                (4, 1, "third support line", ""),
            ],
        },
    )
    expected = [
        _turn("user", "first user line"),
        _turn("assistant", "first support line"),
        _turn("user", "second user line"),
        _turn("assistant", "second support line third support line"),
    ]
    dialogs = build_ed(tmp_path)
    assert all(d == expected for d in dialogs)


def test_build_ed_drops_flagged_single_speaker_and_short(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        {
            "keep": [
                (0, 0, "keep marker", ""),
                (1, 1, "keep reply", ""),
                (2, 0, "keep again", ""),
                (3, 1, "keep answer", ""),
            ],
            "flagged": [
                (0, 0, "flagged marker", ""),
                (1, 1, "flagged reply", "not_proficient"),
                (2, 0, "flagged again", ""),
                (3, 1, "flagged answer", ""),
            ],
            "single": [
                (0, 0, "single marker", ""),
                (1, 0, "single two", ""),
                (2, 0, "single three", ""),
                (3, 0, "single four", ""),
            ],
            "short": [
                (0, 0, "short marker", ""),
                (1, 1, "short reply", ""),
            ],
        },
    )
    dialogs = build_ed(tmp_path)
    assert {d[0]["content"] for d in dialogs} == {"keep marker"}


def test_build_ed_restores_comma_placeholders(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        {
            "c1": [
                (0, 0, "I am sad _comma_ really", ""),
                (1, 1, "I hear you", ""),
                (2, 0, "that is hard", ""),
                (3, 1, "it is", ""),
            ],
        },
    )
    dialogs = build_ed(tmp_path)
    assert all(d[0] == _turn("user", "I am sad , really") for d in dialogs)


def test_build_ed_exits_when_a_csv_is_missing(tmp_path: Path) -> None:
    _write_ed_csv(tmp_path / "train.csv", {})
    with pytest.raises(SystemExit):
        build_ed(tmp_path)


def test_build_ed_exits_when_a_required_column_is_missing(tmp_path: Path) -> None:
    for name in ("train", "valid", "test"):
        with open(tmp_path / f"{name}.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["conv_id", "utterance_idx", "utterance"])
            writer.writerow(["1", "0", "hello"])
    with pytest.raises(SystemExit):
        build_ed(tmp_path)


def test_main_writes_jsonl_and_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_dir = tmp_path / "ed"
    _write_corpus(
        csv_dir,
        {
            "c1": [
                (0, 0, "first user line", ""),
                (1, 1, "first support line", ""),
                (2, 0, "second user line", ""),
                (3, 1, "second support line", ""),
            ],
        },
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data.py", "--ed-csv-dir", str(csv_dir), "--out-dir", str(out_dir)],
    )
    monkeypatch.setattr("prepare_data.build_esconv", lambda: [])

    main()

    expected = json.dumps(
        {
            "messages": [
                _turn("user", "first user line"),
                _turn("assistant", "first support line"),
                _turn("user", "second user line"),
                _turn("assistant", "second support line"),
            ]
        },
        sort_keys=True,
    )
    records = []
    for name in ("sft_train.jsonl", "sft_val.jsonl"):
        records += [
            json.loads(line)
            for line in (out_dir / name).read_text(encoding="utf-8").splitlines()
        ]
    assert len(records) == 3  # one dialog mirrored across train/valid/test
    assert {json.dumps(r, sort_keys=True) for r in records} == {expected}

    stats = json.loads((out_dir / "corpus_stats.json").read_text(encoding="utf-8"))
    assert stats["esconv"] == 0
    assert stats["empathetic_dialogues"] == len(records)
    assert stats["train"] + stats["val"] == len(records)
