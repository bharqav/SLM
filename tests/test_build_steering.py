"""Tests for the policy-steering SFT set builder.

Pure logic: builds the steering dialogs, checks the crisis/safety policy
shape (assistant-final, system-first) and the stage-2 mix arithmetic
(steering xWEIGHT + replay slice).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "slm-ft" / "steering"))

import build_steering  # noqa: E402

EXPECTED_STEERING = (
    len(build_steering.L2_EN_USERS) * 3
    + len(build_steering.L2_ML)
    + len(build_steering.L1_PAIRS)
    + len(build_steering.MED_PAIRS)
    + len(build_steering.MULTI_TURN)
    + len(build_steering.INJ_PAIRS)
    + len(build_steering.ID_PAIRS)
    + len(build_steering.PII_PAIRS)
)


class TestDialog:
    def test_single_turn_shape(self):
        d = build_steering.d("u", "a")
        msgs = d["messages"]
        assert msgs[0]["role"] == "system"
        assert [m["role"] for m in msgs[1:]] == ["user", "assistant"]
        assert msgs[-1]["role"] == "assistant"

    def test_multi_turn_shape(self):
        d = build_steering.d("u1", "a1", extra_user="u2", extra_assistant="a2")
        assert [m["role"] for m in d["messages"][1:]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_extra_turn_needs_both_halves(self):
        d = build_steering.d("u1", "a1", extra_user="u2", extra_assistant=None)
        assert len(d["messages"]) == 3

    def test_no_empty_content(self):
        d = build_steering.d("u", "a")
        assert all(m["content"].strip() for m in d["messages"])


class TestBuildAll:
    def test_total_count(self):
        dialogs = build_steering.build_all()
        assert len(dialogs) == EXPECTED_STEERING == 30

    def test_every_dialog_is_assistant_final(self):
        for d in build_steering.build_all():
            msgs = d["messages"]
            assert msgs[0]["role"] == "system"
            assert msgs[-1]["role"] == "assistant"
            assert sum(1 for m in msgs if m["role"] == "user") >= 1

    def test_each_l2_user_appears_three_times(self):
        users = [d["messages"][1]["content"] for d in build_steering.build_all()]
        for u in build_steering.L2_EN_USERS:
            assert users.count(u) == 3

    def test_replies_never_run_out(self):
        # 6 users x 3 shuffles needs 18 replies from a 6-entry pool
        users = [d["messages"][1]["content"] for d in build_steering.build_all()]
        assert len(users) == EXPECTED_STEERING


class TestMain:
    def test_writes_all_outputs(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / "data"
        replay = tmp_path / "sft_train.jsonl"
        with open(replay, "w", encoding="utf-8") as fh:
            for i in range(build_steering.REPLAY_N + 10):
                fh.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": f"u{i}"},
                                {"role": "assistant", "content": "a"},
                            ]
                        }
                    )
                    + "\n"
                )
        monkeypatch.setattr(build_steering, "OUT", out)
        monkeypatch.setattr(build_steering, "REPLAY_FROM", replay)

        build_steering.main()

        steering = [
            json.loads(line)
            for line in (out / "steering_sft.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        stage2 = [
            json.loads(line)
            for line in (out / "stage2_sft.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(steering) == EXPECTED_STEERING
        assert len(stage2) == EXPECTED_STEERING * build_steering.WEIGHT + build_steering.REPLAY_N

        stats = json.loads((out / "steering_stats.json").read_text(encoding="utf-8"))
        assert stats == {
            "steering_dialogs": EXPECTED_STEERING,
            "stage2_steering_weighted": EXPECTED_STEERING * build_steering.WEIGHT,
            "stage2_replay": build_steering.REPLAY_N,
        }
        assert "steering_dialogs" in capsys.readouterr().out

    def test_replay_slice_capped(self, tmp_path, monkeypatch):
        out = tmp_path / "data"
        replay = tmp_path / "sft_train.jsonl"
        replay.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(build_steering, "OUT", out)
        monkeypatch.setattr(build_steering, "REPLAY_FROM", replay)

        build_steering.main()

        stage2 = (out / "stage2_sft.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(stage2) == EXPECTED_STEERING * build_steering.WEIGHT + 1
        stats = json.loads((out / "steering_stats.json").read_text(encoding="utf-8"))
        assert stats["stage2_replay"] == 1
