import asyncio
import json
from pathlib import Path

import pytest

from examples.eval_hk_tinker_rl import (
    dataset_fingerprint,
    evaluate_base,
    evaluate_checkpoint_against_frozen,
    evaluate_paired,
    grade_output,
    load_frozen_base_predictions,
    seed_cluster_bootstrap,
    summarize_base,
    summarize_paired,
    validate_provenance_manifest,
    write_base_results,
)
from tests.hong_kong_mahjong.test_tinker_rl_adapter import _record, _write


def test_grade_output_reports_oracle_regret_and_tie_accuracy() -> None:
    record = _record("eval", 20)
    best = grade_output(record, "PASS")
    assert best["oracle_reward"] == 0.25
    assert best["normalized_regret"] == 0.0
    assert best["best_action_with_ties"] is True
    invalid = grade_output(record, "PASS now")
    assert invalid["legal_action"] is False
    assert invalid["normalized_regret"] == 1.25


def test_paired_eval_uses_identical_records_and_sampling_seeds() -> None:
    records = [_record("eval", 20), _record("eval", 21)]
    calls: list[tuple[str, int, str]] = []

    async def base(messages, seed):
        calls.append(("base", seed, messages[1]["content"]))
        return "DISCARD_1M"

    async def checkpoint(messages, seed):
        calls.append(("checkpoint", seed, messages[1]["content"]))
        return "PASS"

    rows = asyncio.run(evaluate_paired(records, base, checkpoint, seed=100, concurrency=2))
    assert [row["sample_seed"] for row in rows] == [100, 101]
    assert {(kind, seed) for kind, seed, _ in calls} == {
        ("base", 100), ("checkpoint", 100), ("base", 101), ("checkpoint", 101)
    }
    assert all(row["paired_delta"]["oracle_reward"] == 1.25 for row in rows)


def test_base_only_eval_is_deterministic_and_preserves_exact_pairing_record() -> None:
    records = [_record("eval", 20), _record("eval", 21)]
    calls: list[tuple[int, str]] = []

    async def base(messages, seed):
        calls.append((seed, messages[1]["content"]))
        return "PASS"

    rows = asyncio.run(evaluate_base(records, base, seed=400, concurrency=2))
    assert [row["sample_seed"] for row in rows] == [400, 401]
    assert [seed for seed, _ in calls] == [400, 401]
    assert rows[0]["record"] == records[0]
    assert rows[0]["record_fingerprint"] == dataset_fingerprint([records[0]])
    assert rows[0]["base"]["best_action_with_ties"] is True


def test_base_summary_and_artifact_include_reproducibility_contract(tmp_path: Path) -> None:
    records = [_record("eval", 20), _record("eval", 21)]

    async def base(messages, seed):
        return "PASS"

    rows = asyncio.run(evaluate_base(records, base, seed=4, concurrency=1))
    fingerprint = dataset_fingerprint(records)
    summary = summarize_base(
        rows,
        base_model="Qwen/Qwen3-8B",
        dataset={"path": "/frozen/eval.jsonl", "selected_records_fingerprint": fingerprint},
        config={"seed": 4, "temperature": 0.0, "max_output_tokens": 12},
    )
    assert summary["evaluation_mode"] == "base_only"
    assert summary["model"] == {"kind": "base_model", "path": "Qwen/Qwen3-8B"}
    assert summary["dataset"]["selected_records_fingerprint"] == fingerprint
    assert summary["base"]["mean_oracle_reward"] == 0.25
    assert summary["by_game_seed"]["20"]["best_action_accuracy"] == 1.0
    assert summary["by_decision"]["uniform_legal"]["legal_rate"] == 1.0

    output = tmp_path / "baseline"
    write_base_results(output, rows, summary)
    stored_rows = [json.loads(line) for line in (output / "base_predictions.jsonl").read_text().splitlines()]
    stored_summary = json.loads((output / "summary.json").read_text())
    assert stored_rows[0]["record"] == records[0]
    assert stored_summary["config"]["seed"] == 4
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_base_results(output, rows, summary)


def _frozen_artifact(tmp_path: Path, records: list[dict]) -> Path:
    async def base(messages, seed):
        return "PASS"

    rows = asyncio.run(evaluate_base(records, base, seed=100, concurrency=1))
    output = tmp_path / "frozen"
    summary = summarize_base(
        rows,
        base_model="Qwen/Qwen3-8B",
        dataset={
            "path": str(tmp_path / "eval.jsonl"),
            "file_sha256": "sha256:eval",
            "selected_records": len(records),
            "selected_records_fingerprint": dataset_fingerprint(records),
            "split": "eval",
        },
        config={
            "renderer": "qwen3_disable_thinking",
            "temperature": 0.0,
            "max_output_tokens": 12,
            "seed": 100,
            "concurrency": 1,
            "malformed_penalty": -1.0,
            "limit": None,
        },
    )
    write_base_results(output, rows, summary)
    return output / "base_predictions.jsonl"


def test_frozen_base_loader_validates_full_artifact_then_selects_exact_prefix(tmp_path: Path) -> None:
    records = [_record("eval", 20), _record("eval", 21)]
    predictions = _frozen_artifact(tmp_path, records)
    rows, source = load_frozen_base_predictions(
        predictions,
        full_eval_records=records,
        selected_records=records[:1],
        base_model="Qwen/Qwen3-8B",
        renderer="qwen3_disable_thinking",
        temperature=0.0,
        max_output_tokens=12,
        seed=100,
        malformed_penalty=-1.0,
        eval_file_sha256="sha256:eval",
    )
    assert len(rows) == 1
    assert rows[0]["record"] == records[0]
    assert source["source_count"] == 2
    assert source["comparison_selected_records_fingerprint"] == dataset_fingerprint(records[:1])


@pytest.mark.parametrize("mutation", ["reorder", "duplicate_index", "truncate", "embedded_record"])
def test_frozen_base_loader_rejects_corruption(tmp_path: Path, mutation: str) -> None:
    records = [_record("eval", 20), _record("eval", 21)]
    predictions = _frozen_artifact(tmp_path, records)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    if mutation == "reorder":
        rows.reverse()
    elif mutation == "duplicate_index":
        rows[1]["index"] = 0
    elif mutation == "truncate":
        rows.pop()
    else:
        rows[0]["record"]["metadata"]["step"] += 1
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="indices|row count|embedded record"):
        load_frozen_base_predictions(
            predictions,
            full_eval_records=records,
            selected_records=records,
            base_model="Qwen/Qwen3-8B",
            renderer="qwen3_disable_thinking",
            temperature=0.0,
            max_output_tokens=12,
            seed=100,
            malformed_penalty=-1.0,
            eval_file_sha256="sha256:eval",
        )


def test_frozen_comparison_is_not_changed_by_live_base_nondeterminism() -> None:
    record = _record("eval", 20)

    async def frozen_base(messages, seed):
        return "PASS"

    async def nondeterministic_live_base(messages, seed):
        return "DISCARD_1M"

    async def checkpoint(messages, seed):
        return "PASS"

    frozen = asyncio.run(evaluate_base([record], frozen_base, seed=5, concurrency=1))
    frozen_pair = asyncio.run(evaluate_checkpoint_against_frozen(
        [record], frozen, checkpoint, seed=5, concurrency=1
    ))
    fresh_pair = asyncio.run(evaluate_paired(
        [record], nondeterministic_live_base, checkpoint, seed=5, concurrency=1
    ))
    assert frozen_pair[0]["paired_delta"]["oracle_reward"] == 0.0
    assert fresh_pair[0]["paired_delta"]["oracle_reward"] == 1.25


def test_seed_cluster_bootstrap_and_independent_summary() -> None:
    rows = []
    for game_seed in (1, 1, 2, 2):
        rows.append({
            "game_seed": game_seed,
            "trajectory_policy": "uniform_legal",
            "base": {
                "oracle_reward": -1.0, "normalized_regret": 1.25,
                "best_action_with_ties": False, "canonical_format": True,
                "parseable_action": True, "legal_action": True,
            },
            "checkpoint": {
                "oracle_reward": 0.25, "normalized_regret": 0.0,
                "best_action_with_ties": True, "canonical_format": True,
                "parseable_action": True, "legal_action": True,
            },
            "paired_delta": {
                "oracle_reward": 1.25, "normalized_regret": -1.25,
                "best_action_accuracy": 1.0,
            },
        })
    ci = seed_cluster_bootstrap(rows, "oracle_reward", samples=100, seed=7)
    assert ci["clusters"] == 2
    assert ci["ci95_lower"] == pytest.approx(1.25)
    summary = summarize_paired(
        rows, base_model="base", checkpoint_path="checkpoint", settings={"temperature": 0.0},
        dataset={"file_sha256": "sha256:eval", "selected_records_fingerprint": "sha256:selected"},
        bootstrap_samples=100,
    )
    assert summary["checkpoint"]["mean_oracle_reward"] == 0.25
    assert summary["by_game_seed"]["1"]["base"]["count"] == 2
    assert summary["by_decision"]["uniform_legal"]["checkpoint"]["best_action_accuracy"] == 1.0
    assert summary["success_target"]["met"] is False  # fewer than 500 unseen states
    assert summary["dataset"]["file_sha256"] == "sha256:eval"


def test_cluster_bootstrap_point_estimate_weights_seeds_equally() -> None:
    rows = [
        {"game_seed": 1, "paired_delta": {"oracle_reward": 1.0}},
        {"game_seed": 2, "paired_delta": {"oracle_reward": -1.0}},
        {"game_seed": 2, "paired_delta": {"oracle_reward": -1.0}},
        {"game_seed": 2, "paired_delta": {"oracle_reward": -1.0}},
    ]
    result = seed_cluster_bootstrap(rows, "oracle_reward", samples=100, seed=3)
    assert result["point_estimate"] == 0.0


def test_empty_and_bad_concurrency_are_rejected() -> None:
    async def sample(messages, seed):
        return "PASS"

    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(evaluate_paired([_record("eval", 20)], sample, sample, seed=0, concurrency=0))
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(evaluate_base([_record("eval", 20)], sample, seed=0, concurrency=0))
    with pytest.raises(ValueError, match="empty"):
        summarize_paired([], base_model="b", checkpoint_path="c", dataset={}, settings={})
    with pytest.raises(ValueError, match="empty"):
        summarize_base([], base_model="b", dataset={}, config={})


def test_base_only_paid_mode_does_not_require_checkpoint_but_requires_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples import eval_hk_tinker_rl

    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setattr(eval_hk_tinker_rl, "_load_key_for_explicit_run", lambda path: None)
    with pytest.raises(SystemExit, match="set TINKER_API_KEY"):
        eval_hk_tinker_rl.main([
            "--train-jsonl", str(train),
            "--eval-jsonl", str(evaluation),
            "--output-dir", str(tmp_path / "output"),
            "--run",
            "--base-only",
            "--env-file", str(tmp_path / "missing.env"),
        ])


def test_base_only_train_split_writes_audited_train_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples import eval_hk_tinker_rl

    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    train_records = [_record("train", 10), _record("train", 11)]
    train_records[1]["messages"][1]["content"] += "\nTURN=11"
    _write(train, train_records)
    _write(evaluation, [_record("eval", 20)])

    async def fake_run_base_remote(args, records):
        async def sample(messages, seed):
            return "PASS"

        return await evaluate_base(records, sample, seed=args.seed, concurrency=args.concurrency)

    monkeypatch.setenv("TINKER_API_KEY", "test-only")
    monkeypatch.setattr(eval_hk_tinker_rl, "_load_key_for_explicit_run", lambda path: None)
    monkeypatch.setattr(eval_hk_tinker_rl, "run_base_remote", fake_run_base_remote)
    output = tmp_path / "output"
    assert eval_hk_tinker_rl.main([
        "--train-jsonl", str(train),
        "--eval-jsonl", str(evaluation),
        "--output-dir", str(output),
        "--run",
        "--base-only",
        "--base-only-split", "train",
        "--seed", "100",
    ]) == 0
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "base_predictions.jsonl").read_text().splitlines()]
    assert summary["dataset"]["split"] == "train"
    assert summary["dataset"]["path"] == str(train.resolve())
    assert summary["dataset"]["selected_records"] == 2
    assert summary["dataset"]["selected_records_fingerprint"] == dataset_fingerprint(train_records)
    assert summary["config"]["base_only_split"] == "train"
    assert [row["record"]["metadata"]["split"] for row in rows] == ["train", "train"]


@pytest.mark.parametrize("mode_args", [[], ["--frozen-base-predictions", "frozen.jsonl"]])
def test_train_split_is_rejected_for_paired_and_checkpoint_modes(
    tmp_path: Path, mode_args: list[str]
) -> None:
    from examples import eval_hk_tinker_rl

    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    with pytest.raises(SystemExit, match="paired/checkpoint modes are eval-only"):
        eval_hk_tinker_rl.main([
            "--train-jsonl", str(train),
            "--eval-jsonl", str(evaluation),
            "--output-dir", str(tmp_path / "output"),
            "--validate-only",
            "--base-only-split", "train",
            *mode_args,
        ])


def test_paid_paired_mode_requires_provenance_by_default(tmp_path: Path) -> None:
    from examples import eval_hk_tinker_rl

    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    with pytest.raises(SystemExit, match="provenance-manifest is required by default"):
        eval_hk_tinker_rl.main([
            "--train-jsonl", str(train),
            "--eval-jsonl", str(evaluation),
            "--output-dir", str(tmp_path / "output"),
            "--run",
            "--checkpoint-path", "tinker://checkpoint/123",
            "--env-file", str(tmp_path / "missing.env"),
        ])


def test_provenance_manifest_binds_checkpoint_model_renderer_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    manifest = {
        "schema": "mahjax.hk_tinker_provenance.v1",
        "checkpoint_path": "tinker://checkpoint/123",
        "sampler_checkpoint_path": "tinker://checkpoint/123",
        "state_checkpoint_path": "tinker://state/123",
        "base_model": "Qwen/Qwen3-8B",
        "renderer": "qwen3_disable_thinking",
        "train_sha256": "sha256:train",
        "eval_sha256": "sha256:eval",
        "datasets": {
            "train": {"file_sha256": "sha256:train"},
            "eval": {"file_sha256": "sha256:eval"},
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_provenance_manifest(
        path,
        checkpoint_path="tinker://checkpoint/123",
        base_model="Qwen/Qwen3-8B",
        renderer="qwen3_disable_thinking",
        train_sha256="sha256:train",
        eval_sha256="sha256:eval",
    ) == manifest
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_provenance_manifest(
            path,
            checkpoint_path="tinker://checkpoint/different",
            base_model="Qwen/Qwen3-8B",
            renderer="qwen3_disable_thinking",
            train_sha256="sha256:train",
            eval_sha256="sha256:eval",
        )
