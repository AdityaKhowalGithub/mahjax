import json

from examples.generate_hk_rl_win_supplement import (
    SUPPLEMENT_VERSION,
    TRAJECTORY_POLICY,
    generate_supplement,
)
from mahjax.hong_kong_mahjong.action import Action


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_small_win_supplement_is_deterministic_and_terminal_dominant(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    kwargs = dict(seed_start=910_000_000, target=2, max_games=100, max_steps=256)
    generate_supplement(first, **kwargs)
    generate_supplement(second, **kwargs)
    for filename in ("win_train.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    rows = _rows(first / "win_train.jsonl")
    assert len(rows) == 2
    identities = {(row["metadata"]["seed"], row["metadata"]["player"], row["metadata"]["step"]) for row in rows}
    assert len(identities) == 2
    for row in rows:
        legal_ids = {action["id"] for action in row["legal_actions"]}
        win_ids = legal_ids & {Action.RON, Action.TSUMO}
        assert win_ids
        assert row["metadata"]["split"] == "train"
        assert row["metadata"]["trajectory_policy"] == TRAJECTORY_POLICY
        assert row["metadata"]["supplement_version"] == SUPPLEMENT_VERSION
        assert row["metadata"]["trajectory_action_id"] in win_ids
        assert row["oracle_diagnostics"]["oracle_action_id"] in win_ids
        assert set(row["best_action_names"]) <= {"RON", "TSUMO"}
        assert row["action_rewards"][row["metadata"]["trajectory_action_name"]] == 1.0
        prompt = row["messages"][1]["content"]
        assert "HAND=" in prompt and "LEGAL_ACTIONS=" in prompt
        assert all(secret not in prompt for secret in ("DECK=", "OPPONENT_HAND=", "action_rewards"))
