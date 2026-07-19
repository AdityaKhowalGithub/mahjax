#!/usr/bin/env python3
"""Base-only or paired evaluation for Hong Kong Mahjong oracle rewards."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.eval_hk_tinker_sft import parsed_response_text
from examples.run_hk_tinker_rl import (
    DEFAULT_MODEL,
    DEFAULT_RENDERER,
    _load_key_for_explicit_run,
    read_oracle_jsonl,
    score_action,
    validate_datasets,
)

SampleCompletion = Callable[[Sequence[Mapping[str, str]], int], Awaitable[str]]
PROVENANCE_SCHEMA = "mahjax.hk_tinker_provenance.v1"


def dataset_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact ordered record selection independent of JSONL formatting."""
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_frozen_base_predictions(  # noqa: C901 - one fail-closed artifact audit
    predictions_path: Path,
    *,
    full_eval_records: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
    base_model: str,
    renderer: str,
    temperature: float,
    max_output_tokens: int,
    seed: int,
    malformed_penalty: float,
    eval_file_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and fully audit a frozen base artifact before selecting its prefix."""
    predictions_path = Path(predictions_path).resolve()
    summary_path = predictions_path.with_name("summary.json")
    try:
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"frozen base artifact is incomplete: {error.filename}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid frozen base artifact JSON: {error.msg}") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("frozen base predictions must be a non-empty JSONL without blank lines")
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid frozen base predictions JSONL: {error.msg}") from error
    if not isinstance(summary, dict) or summary.get("evaluation_mode") != "base_only":
        raise ValueError("frozen base sibling summary must describe a base_only evaluation")
    expected_count = summary.get("evaluated_count")
    dataset = summary.get("dataset")
    config = summary.get("config")
    model = summary.get("model")
    if not isinstance(dataset, dict) or not isinstance(config, dict) or not isinstance(model, dict):
        raise ValueError("frozen base summary is missing model, dataset, or config")
    if expected_count != len(rows) or dataset.get("selected_records") != len(rows):
        raise ValueError("frozen base row count does not match sibling summary")
    if len(rows) > len(full_eval_records) or len(rows) < len(selected_records):
        raise ValueError("frozen base artifact is truncated or incompatible with current selection")
    if model != {"kind": "base_model", "path": base_model}:
        raise ValueError("frozen base model does not match requested base model")
    expected_config = {
        "renderer": renderer,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        "malformed_penalty": malformed_penalty,
    }
    config_mismatches = [
        key for key, value in expected_config.items() if config.get(key) != value
    ]
    if config_mismatches:
        raise ValueError(
            "frozen base config mismatch: " + ", ".join(sorted(config_mismatches))
        )
    source_records = list(full_eval_records[: len(rows)])
    if dataset.get("file_sha256") != eval_file_sha256:
        raise ValueError("frozen base eval file hash does not match current eval JSONL")
    source_fingerprint = dataset_fingerprint(source_records)
    if dataset.get("selected_records_fingerprint") != source_fingerprint:
        raise ValueError("frozen base selected-record fingerprint mismatch")

    seen_indices: set[int] = set()
    graded_fields = (
        "raw_output",
        "prediction",
        "prediction_action",
        "oracle_reward",
        "normalized_regret",
        "best_action_with_ties",
        "canonical_format",
        "parseable_action",
        "legal_action",
    )
    for index, (row, record) in enumerate(zip(rows, source_records, strict=True)):
        if not isinstance(row, dict) or row.get("index") != index or index in seen_indices:
            raise ValueError("frozen base indices must be unique, contiguous, and ordered")
        seen_indices.add(index)
        metadata = record["metadata"]
        identity = (metadata["seed"], metadata["game_id"], metadata["player"], metadata["step"])
        stored_identity = (row.get("game_seed"), row.get("game_id"), row.get("player"), row.get("step"))
        if stored_identity != identity:
            raise ValueError(f"frozen base game identity mismatch at index {index}")
        if row.get("sample_seed") != seed + index:
            raise ValueError(f"frozen base sample seed mismatch at index {index}")
        record_fingerprint = dataset_fingerprint([record])
        if row.get("record_fingerprint") != record_fingerprint or row.get("record") != record:
            raise ValueError(f"frozen base embedded record mismatch at index {index}")
        stored_grade = row.get("base")
        if not isinstance(stored_grade, dict):
            raise ValueError(f"frozen base grade missing at index {index}")
        recomputed = grade_output(record, str(stored_grade.get("raw_output", "")), malformed_penalty)
        if any(stored_grade.get(field) != recomputed[field] for field in graded_fields):
            raise ValueError(f"frozen base grade mismatch at index {index}")

    selected_fingerprint = dataset_fingerprint(selected_records)
    if dataset_fingerprint(source_records[: len(selected_records)]) != selected_fingerprint:
        raise ValueError("frozen base selection is reordered or does not match current eval selection")
    source = {
        "kind": "frozen_base_predictions",
        "predictions_path": str(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "source_count": len(rows),
        "source_selected_records_fingerprint": source_fingerprint,
        "comparison_count": len(selected_records),
        "comparison_selected_records_fingerprint": selected_fingerprint,
    }
    return rows[: len(selected_records)], source


def validate_provenance_manifest(
    path: Path,
    *,
    checkpoint_path: str,
    base_model: str,
    renderer: str,
    train_sha256: str,
    eval_sha256: str,
) -> dict[str, Any]:
    """Verify that a paid paired eval is bound to the audited training inputs."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"provenance manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid provenance manifest JSON: {error.msg}") from error
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "checkpoint_path": checkpoint_path,
        "base_model": base_model,
        "renderer": renderer,
        "train_sha256": train_sha256,
        "eval_sha256": eval_sha256,
    }
    if not isinstance(manifest, dict):
        raise ValueError("provenance manifest must be a JSON object")
    mismatches = [
        f"{field}: expected {value!r}, got {manifest.get(field)!r}"
        for field, value in expected.items()
        if manifest.get(field) != value
    ]
    if mismatches:
        raise ValueError("provenance mismatch: " + "; ".join(mismatches))
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not all(
        isinstance(datasets.get(split), dict) for split in ("train", "eval")
    ):
        raise ValueError("provenance mismatch: generated manifest must contain train/eval datasets")
    structural_mismatches = []
    if manifest.get("sampler_checkpoint_path") != checkpoint_path:
        structural_mismatches.append("sampler_checkpoint_path")
    if datasets["train"].get("file_sha256") != train_sha256:
        structural_mismatches.append("datasets.train.file_sha256")
    if datasets["eval"].get("file_sha256") != eval_sha256:
        structural_mismatches.append("datasets.eval.file_sha256")
    if structural_mismatches:
        raise ValueError(
            "provenance mismatch in generated structure: " + ", ".join(structural_mismatches)
        )
    return manifest


def grade_output(record: Mapping[str, Any], raw_output: str, malformed_penalty: float = -1.0) -> dict[str, Any]:
    scored = score_action(record, raw_output, malformed_penalty)
    best_reward = max(float(value) for value in record["action_rewards"].values())
    # Invalid actions use the lowest normalized reward for regret, keeping regret in [0, 2].
    chosen_for_regret = scored["oracle_reward"] if scored["legal_action"] else -1.0
    return {
        "raw_output": raw_output,
        "prediction": scored["prediction"],
        "prediction_action": scored["prediction_action"],
        "oracle_reward": scored["oracle_reward"],
        "normalized_regret": best_reward - chosen_for_regret,
        "best_action_with_ties": scored["best_action"],
        "canonical_format": scored["canonical_format"],
        "parseable_action": scored["parseable_action"],
        "legal_action": scored["legal_action"],
    }


async def evaluate_paired(
    records: Sequence[Mapping[str, Any]],
    sample_base: SampleCompletion,
    sample_checkpoint: SampleCompletion,
    *,
    seed: int,
    concurrency: int,
    malformed_penalty: float = -1.0,
) -> list[dict[str, Any]]:
    """Sample both models on identical records and per-record sampling seeds."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
        sample_seed = seed + index
        async with semaphore:
            base_started = time.perf_counter()
            base_raw = await sample_base(record["messages"], sample_seed)
            base_latency = time.perf_counter() - base_started
            checkpoint_started = time.perf_counter()
            checkpoint_raw = await sample_checkpoint(record["messages"], sample_seed)
            checkpoint_latency = time.perf_counter() - checkpoint_started
        base = grade_output(record, base_raw, malformed_penalty)
        checkpoint = grade_output(record, checkpoint_raw, malformed_penalty)
        return {
            "index": index,
            "sample_seed": sample_seed,
            "game_seed": record["metadata"]["seed"],
            "game_id": record["metadata"]["game_id"],
            "player": record["metadata"]["player"],
            "step": record["metadata"]["step"],
            "trajectory_policy": record["metadata"]["trajectory_policy"],
            "legal_action_names": [action["name"] for action in record["legal_actions"]],
            "best_action_names": list(record["best_action_names"]),
            "base": {**base, "latency_seconds": round(base_latency, 6)},
            "checkpoint": {**checkpoint, "latency_seconds": round(checkpoint_latency, 6)},
            "paired_delta": {
                "oracle_reward": checkpoint["oracle_reward"] - base["oracle_reward"],
                "normalized_regret": checkpoint["normalized_regret"] - base["normalized_regret"],
                "best_action_accuracy": float(checkpoint["best_action_with_ties"])
                - float(base["best_action_with_ties"]),
            },
        }

    return list(await asyncio.gather(*(one(index, record) for index, record in enumerate(records))))


async def evaluate_checkpoint_against_frozen(
    records: Sequence[Mapping[str, Any]],
    frozen_base_rows: Sequence[Mapping[str, Any]],
    sample_checkpoint: SampleCompletion,
    *,
    seed: int,
    concurrency: int,
    malformed_penalty: float = -1.0,
) -> list[dict[str, Any]]:
    """Sample only the checkpoint and pair it with an audited frozen baseline."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if len(records) != len(frozen_base_rows):
        raise ValueError("frozen base rows and selected records must have equal length")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, record: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
        sample_seed = seed + index
        if frozen["index"] != index or frozen["sample_seed"] != sample_seed:
            raise ValueError(f"frozen base pairing mismatch at index {index}")
        async with semaphore:
            started = time.perf_counter()
            raw_output = await sample_checkpoint(record["messages"], sample_seed)
            latency = time.perf_counter() - started
        base = dict(frozen["base"])
        checkpoint = grade_output(record, raw_output, malformed_penalty)
        checkpoint["latency_seconds"] = round(latency, 6)
        return {
            "index": index,
            "sample_seed": sample_seed,
            "game_seed": record["metadata"]["seed"],
            "game_id": record["metadata"]["game_id"],
            "player": record["metadata"]["player"],
            "step": record["metadata"]["step"],
            "trajectory_policy": record["metadata"]["trajectory_policy"],
            "legal_action_names": [action["name"] for action in record["legal_actions"]],
            "best_action_names": list(record["best_action_names"]),
            "base": base,
            "checkpoint": checkpoint,
            "paired_delta": {
                "oracle_reward": checkpoint["oracle_reward"] - base["oracle_reward"],
                "normalized_regret": checkpoint["normalized_regret"] - base["normalized_regret"],
                "best_action_accuracy": float(checkpoint["best_action_with_ties"])
                - float(base["best_action_with_ties"]),
            },
        }

    return list(
        await asyncio.gather(
            *(one(index, record, frozen) for index, (record, frozen) in enumerate(
                zip(records, frozen_base_rows, strict=True)
            ))
        )
    )


async def evaluate_base(
    records: Sequence[Mapping[str, Any]],
    sample_base: SampleCompletion,
    *,
    seed: int,
    concurrency: int,
    malformed_penalty: float = -1.0,
) -> list[dict[str, Any]]:
    """Evaluate an untouched base model with deterministic per-record seeds."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
        sample_seed = seed + index
        async with semaphore:
            started = time.perf_counter()
            raw_output = await sample_base(record["messages"], sample_seed)
            latency = time.perf_counter() - started
        return {
            "index": index,
            "sample_seed": sample_seed,
            "record_fingerprint": dataset_fingerprint([record]),
            "game_seed": record["metadata"]["seed"],
            "game_id": record["metadata"]["game_id"],
            "player": record["metadata"]["player"],
            "step": record["metadata"]["step"],
            "trajectory_policy": record["metadata"]["trajectory_policy"],
            # Preserve the complete frozen oracle row so a later checkpoint run can
            # reconstruct the exact prompt, reward map, ties, and pairing identity.
            "record": dict(record),
            "base": {
                **grade_output(record, raw_output, malformed_penalty),
                "latency_seconds": round(latency, 6),
            },
        }

    return list(await asyncio.gather(*(one(index, record) for index, record in enumerate(records))))


def _model_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int]:
    selected = [row[key] for row in rows]
    count = len(selected)
    return {
        "count": count,
        "mean_oracle_reward": statistics.fmean(float(row["oracle_reward"]) for row in selected),
        "mean_normalized_regret": statistics.fmean(float(row["normalized_regret"]) for row in selected),
        "best_action_accuracy": statistics.fmean(float(row["best_action_with_ties"]) for row in selected),
        "canonical_rate": statistics.fmean(float(row["canonical_format"]) for row in selected),
        "parseable_rate": statistics.fmean(float(row["parseable_action"]) for row in selected),
        "legal_rate": statistics.fmean(float(row["legal_action"]) for row in selected),
    }


def _grouped_model_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    decision_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_rows[int(row["game_seed"])].append(row)
        decision_rows[str(row["trajectory_policy"])].append(row)
    by_seed = {
        str(game_seed): _model_metrics(grouped, key)
        for game_seed, grouped in sorted(seed_rows.items())
    }
    by_decision = {
        decision: _model_metrics(grouped, key)
        for decision, grouped in sorted(decision_rows.items())
    }
    return by_seed, by_decision


def summarize_base(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_model: str,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize base-only results independently of all training metrics."""
    if not rows:
        raise ValueError("cannot summarize empty results")
    by_seed, by_decision = _grouped_model_metrics(rows, "base")
    return {
        "evaluation_mode": "base_only",
        "model": {"kind": "base_model", "path": base_model},
        "dataset": dict(dataset),
        "config": dict(config),
        "evaluated_count": len(rows),
        "game_seed_count": len(by_seed),
        "base": _model_metrics(rows, "base"),
        "by_game_seed": by_seed,
        "by_decision": by_decision,
    }


def seed_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Percentile CI for a paired delta, resampling whole game-seed clusters."""
    if samples < 1:
        raise ValueError("bootstrap samples must be at least 1")
    clusters: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        clusters[int(row["game_seed"])].append(float(row["paired_delta"][field]))
    cluster_means = [statistics.fmean(values) for _, values in sorted(clusters.items())]
    if not cluster_means:
        raise ValueError("cannot bootstrap empty results")
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choice(cluster_means) for _ in cluster_means) for _ in range(samples)
    )
    lower = estimates[max(0, int(0.025 * samples) - 1)]
    upper = estimates[min(samples - 1, int(0.975 * samples))]
    return {
        "clusters": len(cluster_means),
        "samples": samples,
        "point_estimate": statistics.fmean(cluster_means),
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def summarize_paired(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_model: str,
    checkpoint_path: str,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
    base_source: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty results")
    reward_gain = seed_cluster_bootstrap(rows, "oracle_reward", samples=bootstrap_samples, seed=bootstrap_seed)
    accuracy_gain = seed_cluster_bootstrap(
        rows, "best_action_accuracy", samples=bootstrap_samples, seed=bootstrap_seed + 1
    )
    by_seed: dict[str, Any] = {}
    seed_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    decision_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_rows[int(row["game_seed"])].append(row)
        decision_rows[str(row["trajectory_policy"])].append(row)
    for game_seed, grouped in sorted(seed_rows.items()):
        by_seed[str(game_seed)] = {
            "base": _model_metrics(grouped, "base"),
            "checkpoint": _model_metrics(grouped, "checkpoint"),
        }
    by_decision = {
        decision: {"base": _model_metrics(grouped, "base"), "checkpoint": _model_metrics(grouped, "checkpoint")}
        for decision, grouped in sorted(decision_rows.items())
    }
    enough_unseen_states = len(rows) >= 500
    success = enough_unseen_states and (
        (reward_gain["point_estimate"] >= 0.15 and reward_gain["ci95_lower"] > 0)
        or (accuracy_gain["point_estimate"] >= 0.10 and accuracy_gain["ci95_lower"] > 0)
    )
    summary = {
        "base_model": base_model,
        "checkpoint_path": checkpoint_path,
        "dataset": dict(dataset),
        "evaluated_count": len(rows),
        "game_seed_count": len(seed_rows),
        "base": _model_metrics(rows, "base"),
        "checkpoint": _model_metrics(rows, "checkpoint"),
        "paired_gain": {
            "oracle_reward": reward_gain,
            "best_action_accuracy": accuracy_gain,
            "mean_normalized_regret_delta": statistics.fmean(
                float(row["paired_delta"]["normalized_regret"]) for row in rows
            ),
        },
        "success_target": {
            "met": success,
            "requires_unseen_states": 500,
            "reward_gain_threshold": 0.15,
            "best_action_accuracy_gain_threshold": 0.10,
            "requires_seed_bootstrap_ci95_lower_above_zero": True,
        },
        "by_game_seed": by_seed,
        "by_decision": by_decision,
        "settings": dict(settings),
    }
    if base_source is not None:
        summary["base_source"] = dict(base_source)
    return summary


def write_results(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path, summary_path = output_dir / "paired_predictions.jsonl", output_dir / "summary.json"
    if paired_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation artifacts in {output_dir}")
    with paired_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_base_results(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, summary_path = output_dir / "base_predictions.jsonl", output_dir / "summary.json"
    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation artifacts in {output_dir}")
    with predictions_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


async def run_remote(args: argparse.Namespace, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import tinker
    from tinker_cookbook import renderers

    service = tinker.ServiceClient()
    base_client = await service.create_sampling_client_async(base_model=args.base_model)
    checkpoint_client = await service.create_sampling_client_async(model_path=args.checkpoint_path)
    tokenizer = base_client.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.base_model)
    stop = renderer.get_stop_sequences()

    def sampler(client: Any) -> SampleCompletion:
        async def sample(messages: Sequence[Mapping[str, str]], seed: int) -> str:
            response = await client.sample_async(
                prompt=renderer.build_generation_prompt(list(messages)),
                sampling_params=tinker.SamplingParams(
                    max_tokens=args.max_output_tokens, temperature=args.temperature, seed=seed, stop=stop
                ),
                num_samples=1,
            )
            return parsed_response_text(renderer, response.sequences[0].tokens)

        return sample

    return await evaluate_paired(
        records, sampler(base_client), sampler(checkpoint_client), seed=args.seed,
        concurrency=args.concurrency, malformed_penalty=args.malformed_penalty,
    )


async def run_base_remote(args: argparse.Namespace, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import tinker
    from tinker_cookbook import renderers

    service = tinker.ServiceClient()
    base_client = await service.create_sampling_client_async(base_model=args.base_model)
    tokenizer = base_client.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.base_model)
    stop = renderer.get_stop_sequences()

    async def sample(messages: Sequence[Mapping[str, str]], seed: int) -> str:
        response = await base_client.sample_async(
            prompt=renderer.build_generation_prompt(list(messages)),
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
                seed=seed,
                stop=stop,
            ),
            num_samples=1,
        )
        return parsed_response_text(renderer, response.sequences[0].tokens)

    return await evaluate_base(
        records,
        sample,
        seed=args.seed,
        concurrency=args.concurrency,
        malformed_penalty=args.malformed_penalty,
    )


async def run_checkpoint_remote(
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    frozen_base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import tinker
    from tinker_cookbook import renderers

    service = tinker.ServiceClient()
    checkpoint_client = await service.create_sampling_client_async(model_path=args.checkpoint_path)
    tokenizer = checkpoint_client.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.base_model)
    stop = renderer.get_stop_sequences()

    async def sample(messages: Sequence[Mapping[str, str]], seed: int) -> str:
        response = await checkpoint_client.sample_async(
            prompt=renderer.build_generation_prompt(list(messages)),
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_output_tokens,
                temperature=args.temperature,
                seed=seed,
                stop=stop,
            ),
            num_samples=1,
        )
        return parsed_response_text(renderer, response.sequences[0].tokens)

    return await evaluate_checkpoint_against_frozen(
        records,
        frozen_base_rows,
        sample,
        seed=args.seed,
        concurrency=args.concurrency,
        malformed_penalty=args.malformed_penalty,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true", help="Explicitly launch paid paired sampling")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate only the untouched base model; does not require --checkpoint-path",
    )
    parser.add_argument(
        "--base-only-split",
        choices=("train", "eval"),
        default="eval",
        help="Dataset split for --base-only mining; paired/checkpoint evaluation is eval-only",
    )
    parser.add_argument(
        "--frozen-base-predictions",
        type=Path,
        help="Sample only the checkpoint and compare against an audited base_predictions.jsonl",
    )
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        help="Audited manifest binding checkpoint, model, renderer, and dataset hashes",
    )
    parser.add_argument(
        "--allow-unverified-provenance",
        action="store_true",
        help="Audited override: permit paired paid eval without a verified provenance manifest",
    )
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--malformed-penalty", type=float, default=-1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.base_only and args.frozen_base_predictions is not None:
        raise SystemExit("--base-only and --frozen-base-predictions are mutually exclusive")
    if args.base_only_split != "eval" and not args.base_only:
        raise SystemExit("--base-only-split train requires --base-only; paired/checkpoint modes are eval-only")
    validation = validate_datasets(args.train_jsonl, args.eval_jsonl)
    target_split = args.base_only_split if args.base_only else "eval"
    target_jsonl = args.train_jsonl if target_split == "train" else args.eval_jsonl
    full_records, _ = read_oracle_jsonl(target_jsonl, target_split)
    records = full_records
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        records = records[: args.limit]
    config = {
        "renderer": args.renderer,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "malformed_penalty": args.malformed_penalty,
        "limit": args.limit,
        "base_only_split": target_split,
    }
    dataset = {
        "path": str(target_jsonl.resolve()),
        "file_sha256": file_sha256(target_jsonl),
        "train_file_sha256": file_sha256(args.train_jsonl),
        "eval_file_sha256": file_sha256(args.eval_jsonl),
        "selected_records_fingerprint": dataset_fingerprint(records),
        "selected_records": len(records),
        "split": target_split,
    }
    frozen_rows: list[dict[str, Any]] | None = None
    base_source: dict[str, Any] | None = None
    if args.frozen_base_predictions is not None:
        try:
            frozen_rows, base_source = load_frozen_base_predictions(
                args.frozen_base_predictions,
                full_eval_records=full_records,
                selected_records=records,
                base_model=args.base_model,
                renderer=args.renderer,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                seed=args.seed,
                malformed_penalty=args.malformed_penalty,
                eval_file_sha256=dataset["file_sha256"],
            )
        except ValueError as error:
            raise SystemExit(f"Refusing frozen-base evaluation: {error}") from error
    local_summary = {
        "dataset": validation,
        "evaluation_records": len(records),
        "target": {
            "split": target_split,
            "path": str(target_jsonl.resolve()),
            "file_sha256": dataset["file_sha256"],
            "selected_records_fingerprint": dataset["selected_records_fingerprint"],
        },
    }
    if base_source is not None:
        local_summary["base_source"] = base_source
    print(json.dumps(local_summary, indent=2, sort_keys=True))
    if args.validate_only:
        print("Local validation passed; Tinker was not imported or contacted.")
        return 0
    if not args.base_only and not args.checkpoint_path:
        raise SystemExit("--checkpoint-path is required with --run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output directory: {args.output_dir}")
    provenance: dict[str, Any] | None = None
    if not args.base_only:
        if args.provenance_manifest is not None:
            try:
                provenance = validate_provenance_manifest(
                    args.provenance_manifest,
                    checkpoint_path=args.checkpoint_path,
                    base_model=args.base_model,
                    renderer=args.renderer,
                    train_sha256=dataset["train_file_sha256"],
                    eval_sha256=dataset["file_sha256"],
                )
            except ValueError as error:
                raise SystemExit(f"Refusing paired evaluation: {error}") from error
        elif not args.allow_unverified_provenance:
            raise SystemExit(
                "Refusing paired evaluation: --provenance-manifest is required by default; "
                "use --allow-unverified-provenance only after an explicit audit."
            )
    _load_key_for_explicit_run(args.env_file.resolve())
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Refusing remote evaluation: set TINKER_API_KEY to a current credential first.")
    started = time.perf_counter()
    if args.base_only:
        rows = asyncio.run(run_base_remote(args, records))
        summary = summarize_base(
            rows,
            base_model=args.base_model,
            dataset=dataset,
            config=config,
        )
        summary["runtime"] = {"wall_seconds": round(time.perf_counter() - started, 3)}
        write_base_results(args.output_dir, rows, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if frozen_rows is not None:
        rows = asyncio.run(run_checkpoint_remote(args, records, frozen_rows))
    else:
        rows = asyncio.run(run_remote(args, records))
    settings = {
        **config,
        "eval_jsonl": str(args.eval_jsonl.resolve()),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "provenance": provenance,
        "unverified_provenance_override": args.allow_unverified_provenance,
    }
    summary = summarize_paired(
        rows, base_model=args.base_model, checkpoint_path=args.checkpoint_path,
        dataset=dataset, settings=settings, base_source=base_source,
        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.seed,
    )
    write_results(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
