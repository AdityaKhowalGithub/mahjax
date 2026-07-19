import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.generate_hk_rl_oracle_data import (
    ORACLE_SCORER,
    ORACLE_SCORER_VERSION,
    SCHEMA_VERSION,
    generate_dataset,
)


def _load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("hk-oracle-data")
    manifest = generate_dataset(output, seed=6100, train_games=2, eval_games=1, max_steps=256)
    return output, manifest


def test_generation_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, seed=6200, train_games=1, eval_games=1, max_steps=256)
    generate_dataset(second, seed=6200, train_games=1, eval_games=1, max_steps=256)

    for filename in ("train.jsonl", "eval.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_prompts_are_two_message_and_leakage_safe(generated):
    output, _ = generated
    for row in _load(output / "train.jsonl") + _load(output / "eval.jsonl"):
        assert [message["role"] for message in row["messages"]] == ["system", "user"]
        prompt = row["messages"][1]["content"]
        assert prompt.count("HAND=") == 1
        assert "LAST_DRAW=" in prompt
        assert "LEGAL_ACTIONS=" in prompt
        assert "action_rewards" not in prompt
        assert "raw_action_scores" not in prompt
        assert "oracle_diagnostics" not in prompt
        for forbidden in ("DECK=", "WALL_ORDER=", "OPPONENT_HAND=", "P0_HAND=", "P1_HAND=", "P2_HAND=", "P3_HAND="):
            assert forbidden not in prompt


def test_rewards_align_exactly_with_legal_actions(generated):
    output, _ = generated
    for row in _load(output / "train.jsonl") + _load(output / "eval.jsonl"):
        metadata = row["metadata"]
        legal_ids = [action["id"] for action in row["legal_actions"]]
        legal_names = [action["name"] for action in row["legal_actions"]]
        assert row["schema"] == SCHEMA_VERSION
        assert metadata["data_quality"] == "oracle_privileged_reward"
        assert isinstance(metadata["seed"], int)
        assert isinstance(metadata["player"], int)
        assert isinstance(metadata["step"], int)
        assert isinstance(metadata["game_id"], str)
        assert set(row["action_rewards"]) == set(legal_names)
        assert set(row["best_action_names"]) <= set(legal_names)
        assert row["oracle_diagnostics"]["oracle_action_name"] in row["best_action_names"]
        assert row["oracle_diagnostics"]["oracle_action_id"] in legal_ids
        assert row["oracle_diagnostics"]["scorer"] == ORACLE_SCORER
        assert row["oracle_diagnostics"]["scorer_version"] == ORACLE_SCORER_VERSION
        assert metadata["trajectory_action_id"] in legal_ids
        assert metadata["trajectory_action_name"] in legal_names
        assert metadata["trajectory_action_name"] == next(
            action["name"] for action in row["legal_actions"] if action["id"] == metadata["trajectory_action_id"]
        )
        if metadata["trajectory_policy"] == "oracle":
            assert metadata["trajectory_action_id"] == row["oracle_diagnostics"]["oracle_action_id"]
            assert metadata["trajectory_action_name"] in row["best_action_names"]
        assert all(-1.0 <= reward <= 1.0 for reward in row["action_rewards"].values())
        best = max(row["action_rewards"].values())
        assert set(row["best_action_names"]) == {
            name for name, reward in row["action_rewards"].items() if reward == best
        }
        rewards = set(row["action_rewards"].values())
        if len(rewards) > 1:
            assert min(rewards) == -1.0
            assert max(rewards) == 1.0

        prompt_legal = row["messages"][1]["content"].split("LEGAL_ACTIONS=[", 1)[1].removesuffix("]")
        assert prompt_legal == ",".join(f"{action['id']}:{action['name']}" for action in row["legal_actions"])


def test_whole_games_are_isolated_across_splits(generated):
    output, manifest = generated
    train = _load(output / "train.jsonl")
    eval_rows = _load(output / "eval.jsonl")
    train_seeds = {row["metadata"]["seed"] for row in train}
    eval_seeds = {row["metadata"]["seed"] for row in eval_rows}

    assert train_seeds == set(manifest["splits"]["train"]["game_seeds"])
    assert eval_seeds == set(manifest["splits"]["eval"]["game_seeds"])
    assert train_seeds.isdisjoint(eval_seeds)
    assert {row["metadata"]["split"] for row in train} == {"train"}
    assert {row["metadata"]["split"] for row in eval_rows} == {"eval"}
    assert {row["metadata"]["trajectory_policy"] for row in train + eval_rows} == {"oracle", "heuristic"}
    contract = manifest["oracle_reward_contract"]
    assert contract["scorer"] == ORACLE_SCORER
    assert contract["scorer_version"] == ORACLE_SCORER_VERSION
    assert contract["feature_weights"]["live_ukeire"] > 0
    assert contract["terminal_priority"] > 0


def test_existing_outputs_require_explicit_overwrite(generated):
    output, _ = generated
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_dataset(output, seed=6100, train_games=2, eval_games=1, max_steps=256)


def test_exact_split_record_targets(tmp_path):
    output = tmp_path / "exact"
    manifest = generate_dataset(
        output,
        seed=6300,
        train_games=2,
        eval_games=1,
        max_steps=256,
        records_per_game=8,
        train_record_target=10,
        eval_record_target=5,
    )
    assert len(_load(output / "train.jsonl")) == 10
    assert len(_load(output / "eval.jsonl")) == 5
    assert manifest["splits"]["train"]["records"] == 10
    assert manifest["splits"]["eval"]["records"] == 5


def test_stratified_sampling_covers_late_steps_and_includes_final_decision():
    from examples.generate_hk_rl_oracle_data import generate_game

    rows = generate_game(6400, "train", max_steps=256, records_per_game=8)
    steps = [row["metadata"]["step"] for row in rows]
    trajectory_steps = rows[0]["metadata"]["trajectory_steps"]
    assert len(rows) == 8
    assert steps[0] == 0
    assert steps[-1] == trajectory_steps - 1
    assert rows[-1]["metadata"]["is_final_decision"] is True
    assert all(not row["metadata"]["is_final_decision"] for row in rows[:-1])
    assert steps == sorted(set(steps))
    assert steps[-1] > 8  # the simulator did not stop when the sample filled
    assert [row["metadata"]["trajectory_sample_index"] for row in rows] == list(range(8))
    assert {row["metadata"]["trajectory_sampling"] for row in rows} == {
        "evenly_spaced_full_trajectory_including_endpoints"
    }


def test_direct_script_entrypoint_resolves_sibling_import():
    script = Path(__file__).parents[2] / "examples" / "generate_hk_rl_oracle_data.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--records-per-game" in result.stdout
