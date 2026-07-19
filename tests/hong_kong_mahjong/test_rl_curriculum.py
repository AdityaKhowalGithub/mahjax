import asyncio
import json

import pytest

from examples.build_hk_rl_curriculum import build_curriculum
from examples.eval_hk_tinker_rl import (
    dataset_fingerprint,
    evaluate_base,
    file_sha256,
    summarize_base,
    write_base_results,
)


def _record(split, seed, step, rewards, best):
    actions = [(0, "DISCARD_1M"), (77, "PASS")]
    return {
        "schema": "mahjax.hk_rl_oracle.v1",
        "messages": [
            {"role": "system", "content": "Choose one action."},
            {"role": "user", "content": f"STATE={seed}:{step}\nLEGAL_ACTIONS=[0:DISCARD_1M,77:PASS]"},
        ],
        "metadata": {
            "ruleset": "hk_old_style_v1", "data_quality": "oracle_privileged_reward",
            "split": split, "seed": seed, "game_id": f"game-{seed}", "player": 0,
            "step": step, "trajectory_policy": "natural" if step % 2 == 0 else "supplement",
        },
        "legal_actions": [{"id": action_id, "name": name} for action_id, name in actions],
        "action_rewards": rewards,
        "best_action_names": best,
    }


def _tsumogiri_record(split, seed, step, rewards, best, policy):
    record = _record(split, seed, step, rewards, best)
    actions = [(0, "DISCARD_1M"), (68, "TSUMOGIRI"), (77, "PASS")]
    record["messages"][1]["content"] = (
        f"STATE={seed}:{step}\nLEGAL_ACTIONS=[0:DISCARD_1M,68:TSUMOGIRI,77:PASS]"
    )
    record["metadata"]["trajectory_policy"] = policy
    record["legal_actions"] = [{"id": action_id, "name": name} for action_id, name in actions]
    return record


def _write(path, records):
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _base_artifact(output, records, outputs, train_path):
    async def sample(_messages, seed):
        return outputs[seed - 100]

    rows = asyncio.run(evaluate_base(records, sample, seed=100, concurrency=2))
    summary = summarize_base(
        rows,
        base_model="Qwen/Qwen3-8B",
        dataset={
            "path": str(train_path), "file_sha256": file_sha256(train_path),
            "selected_records_fingerprint": dataset_fingerprint(records),
            "selected_records": len(records), "split": "train",
        },
        config={
            "renderer": "qwen3_disable_thinking", "temperature": 0.0,
            "max_output_tokens": 12, "seed": 100, "malformed_penalty": -1.0,
        },
    )
    write_base_results(output, rows, summary)
    return output / "base_predictions.jsonl"


def test_curriculum_is_deterministic_strict_and_eval_is_untouched(tmp_path):
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    records = [
        _record("train", 10, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"]),
        _record("train", 11, 1, {"DISCARD_1M": 1.0, "PASS": -1.0}, ["DISCARD_1M"]),
        _record("train", 12, 2, {"DISCARD_1M": 0.0, "PASS": 0.0}, ["DISCARD_1M", "PASS"]),
        _record("train", 13, 3, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"]),
    ]
    eval_records = [_record("eval", 20, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"])]
    _write(train, records)
    _write(evaluation, eval_records)
    base = _base_artifact(tmp_path / "base", records, ["DISCARD_1M", "DISCARD_1M", "PASS", "PASS"], train)
    kwargs = dict(
        train_jsonl=train, eval_jsonl=evaluation, base_predictions=base,
        base_model="Qwen/Qwen3-8B", renderer="qwen3_disable_thinking",
        sampling_seed=100, rehearsal_rate=1.0, selection_seed=7,
    )
    dry = build_curriculum(output_dir=tmp_path / "unused", dry_run=True, **kwargs)
    assert dry["selection"]["core_records"] == 1
    assert dry["selection"]["constant_reward_excluded"] == 1
    assert dry["selection"]["rehearsal_records"] == 2

    first, second = tmp_path / "first", tmp_path / "second"
    build_curriculum(output_dir=first, **kwargs)
    build_curriculum(output_dir=second, **kwargs)
    for filename in ("train.jsonl", "eval.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert (first / "eval.jsonl").read_bytes() == evaluation.read_bytes()
    selected = [json.loads(line) for line in (first / "train.jsonl").read_text().splitlines()]
    assert len(selected) == 3
    assert all(record in records for record in selected)
    with pytest.raises(FileExistsError, match="non-empty|overwrite"):
        build_curriculum(output_dir=first, **kwargs)


def test_curriculum_rejects_base_hash_or_model_mismatch(tmp_path):
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    records = [_record("train", 10, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"])]
    _write(train, records)
    _write(evaluation, [_record("eval", 20, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"])])
    base = _base_artifact(tmp_path / "base", records, ["DISCARD_1M"], train)
    common = dict(train_jsonl=train, eval_jsonl=evaluation, base_predictions=base, output_dir=tmp_path / "out", sampling_seed=100)
    with pytest.raises(ValueError, match="model"):
        build_curriculum(base_model="wrong", **common)
    train.write_text(train.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank lines|hash"):
        build_curriculum(**common)


def test_anti_tsumogiri_mode_keeps_all_bias_failures_and_caps_other_mistakes(tmp_path):
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    records = [
        _tsumogiri_record("train", 10, 0, {"DISCARD_1M": 1.0, "TSUMOGIRI": -1.0, "PASS": 0.0}, ["DISCARD_1M"], "oracle"),
        _tsumogiri_record("train", 11, 1, {"DISCARD_1M": 0.0, "TSUMOGIRI": -1.0, "PASS": 1.0}, ["PASS"], "heuristic"),
        _tsumogiri_record("train", 12, 2, {"DISCARD_1M": -1.0, "TSUMOGIRI": 0.0, "PASS": 1.0}, ["PASS"], "oracle"),
        _tsumogiri_record("train", 13, 3, {"DISCARD_1M": 1.0, "TSUMOGIRI": 0.0, "PASS": -1.0}, ["DISCARD_1M"], "heuristic"),
        _tsumogiri_record("train", 14, 4, {"DISCARD_1M": 1.0, "TSUMOGIRI": 0.0, "PASS": -1.0}, ["DISCARD_1M"], "supplement"),
        _tsumogiri_record("train", 15, 5, {"DISCARD_1M": 1.0, "TSUMOGIRI": 0.0, "PASS": -1.0}, ["DISCARD_1M"], "oracle"),
    ]
    _write(train, records)
    _write(evaluation, [_record("eval", 20, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"])])
    base = _base_artifact(
        tmp_path / "base", records,
        ["TSUMOGIRI", "TSUMOGIRI", "DISCARD_1M", "PASS", "PASS", "DISCARD_1M"],
        train,
    )
    kwargs = dict(
        train_jsonl=train, eval_jsonl=evaluation, base_predictions=base,
        output_dir=tmp_path / "out", sampling_seed=100, selection_seed=9,
        selection_mode="anti_tsumogiri_balanced", rehearsal_rate=0.05,
    )
    manifest = build_curriculum(**kwargs)
    assert manifest["selection"] | {
        "anti_tsumogiri_records": 2,
        "other_mistake_candidate_records": 3,
        "other_mistake_target_records": 2,
        "other_mistake_selected_records": 2,
        "rehearsal_records": 1,
        "selected_records": 5,
    } == manifest["selection"]
    selected = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert {record["metadata"]["seed"] for record in selected} >= {10, 11}
    assert len({json.dumps(record["messages"], sort_keys=True) for record in selected}) == 5


def test_curriculum_rejects_duplicate_selected_prompts(tmp_path):
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    records = [
        _record("train", 10, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"]),
        _record("train", 11, 1, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"]),
    ]
    records[1]["messages"] = records[0]["messages"]
    _write(train, records)
    _write(evaluation, [_record("eval", 20, 0, {"DISCARD_1M": -1.0, "PASS": 1.0}, ["PASS"])])
    base = _base_artifact(tmp_path / "base", records, ["DISCARD_1M", "DISCARD_1M"], train)
    with pytest.raises(ValueError, match="duplicate prompt"):
        build_curriculum(
            train_jsonl=train, eval_jsonl=evaluation, base_predictions=base,
            output_dir=tmp_path / "out", sampling_seed=100,
        )
