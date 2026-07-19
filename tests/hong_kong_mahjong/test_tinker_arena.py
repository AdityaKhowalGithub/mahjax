import asyncio
import json
from pathlib import Path

import pytest

from examples.eval_hk_tinker_arena import (
    play_hand,
    run_paired_arena,
    seed_cluster_bootstrap,
    summarize_arena,
    validate_arena_inputs,
    write_results,
)
from tests.hong_kong_mahjong.test_tinker_rl_adapter import _record, _write


async def _first_legal(messages, seed):
    del seed
    return messages[1]["content"].split("LEGAL_ACTIONS=[", 1)[1].split(":", 1)[1].split(",", 1)[0].removesuffix("]")


def test_invalid_outputs_fall_back_and_hand_finishes() -> None:
    async def invalid(messages, seed):
        return "not an action"

    hand = asyncio.run(play_hand(
        invalid,
        model_label="local",
        game_seed=970001,
        learner_player=0,
        opponent_policy="rule_based",
        sampling_seed=3,
    ))
    assert hand["terminal_steps"] > 0
    assert hand["decisions"]
    assert all(row["invalid_output"] and row["fallback_used"] for row in hand["decisions"])
    assert all(row["executed_action_id"] in row["legal_action_ids"] for row in hand["decisions"])


def test_paired_arena_uses_all_four_rotations_and_matched_initial_games() -> None:
    hands, paired = asyncio.run(run_paired_arena(
        [970002],
        _first_legal,
        _first_legal,
        opponent_policy="rule_based",
        sampling_seed=9,
    ))
    assert len(hands) == 8
    assert len(paired) == 4
    assert {row["learner_player"] for row in paired} == {0, 1, 2, 3}
    assert all(row["reward_delta"] == 0 for row in paired)
    assert all(row["base_reward"] == row["checkpoint_reward"] for row in paired)


def test_seed_exclusion_uses_both_frozen_splits(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    valid = validate_arena_inputs(train, evaluation, [30, 31])
    assert valid["seed_exclusion_passed"] is True
    assert valid["frozen_train_seed_count"] == 1
    with pytest.raises(ValueError, match="overlap"):
        validate_arena_inputs(train, evaluation, [20, 30])


def test_cluster_bootstrap_and_gameplay_summary_are_separate_from_proxy() -> None:
    paired = [
        {"game_seed": seed, "learner_player": player, "reward_delta": 2.0}
        for seed in (1, 2)
        for player in range(4)
    ]
    ci = seed_cluster_bootstrap(paired, samples=100, seed=4)
    assert ci["clusters"] == 2
    assert ci["point_estimate"] == 2.0
    hands = []
    for model, reward in (("base", 0.0), ("checkpoint", 2.0)):
        hands.append({
            "model": model, "terminal_reward": reward, "won": reward > 0,
            "ron": False, "tsumo": reward > 0, "deal_in": False, "draw": reward == 0,
            "decisions": [{
                "canonical_format": True, "legal_action": True, "invalid_output": False,
                "latency_seconds": 0.01,
            }],
        })
    summary = summarize_arena(hands, paired, provenance={"proxy": "not_used"}, bootstrap_samples=100)
    assert summary["evaluation_type"] == "paired_gameplay_arena"
    assert "oracle" not in summary["paired_reward_delta"]
    assert summary["checkpoint"]["mean_terminal_reward"] == 2.0


def test_result_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "arena"
    write_results(output, [{"model": "base"}], {"evaluation_type": "paired_gameplay_arena"})
    assert json.loads((output / "hands.jsonl").read_text())["model"] == "base"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_results(output, [], {})


def test_paid_mode_requires_key_before_tinker_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from examples import eval_hk_tinker_arena

    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setattr(eval_hk_tinker_arena, "_load_key_for_explicit_run", lambda path: None)
    with pytest.raises(SystemExit, match="set TINKER_API_KEY"):
        eval_hk_tinker_arena.main([
            "--train-jsonl", str(train), "--eval-jsonl", str(evaluation),
            "--game-seeds", "30", "--output-dir", str(tmp_path / "out"),
            "--checkpoint-path", "tinker://checkpoint", "--run",
        ])
