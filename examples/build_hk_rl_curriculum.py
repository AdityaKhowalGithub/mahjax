#!/usr/bin/env python3
"""Build a deterministic train-only curriculum from audited base predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.eval_hk_tinker_rl import (
    dataset_fingerprint,
    file_sha256,
    load_frozen_base_predictions,
)
from examples.run_hk_tinker_rl import DEFAULT_MODEL, DEFAULT_RENDERER, read_oracle_jsonl, validate_datasets

CURRICULUM_VERSION = "mahjax.hk_rl_curriculum.v2"
SELECTION_MODES = ("all_mistakes", "anti_tsumogiri_balanced")


def action_family(action: str | None) -> str:
    if action in {"RON", "TSUMO"}:
        return "win"
    if action == "OPEN_KONG" or (action or "").startswith("SELF_KONG_"):
        return "kong"
    if action in {"PON", "CHOW_LEFT", "CHOW_MIDDLE", "CHOW_RIGHT"}:
        return "call"
    if action == "PASS":
        return "pass"
    if action == "TSUMOGIRI" or (action or "").startswith("DISCARD_"):
        return "discard"
    return "invalid"


def _identity(record: Mapping[str, Any]) -> tuple[int, str, int, int]:
    metadata = record["metadata"]
    return metadata["seed"], metadata["game_id"], metadata["player"], metadata["step"]


def _priority(seed: int, namespace: str, record: Mapping[str, Any]) -> str:
    value = json.dumps([seed, namespace, *_identity(record)], separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def oracle_best_action_family(record: Mapping[str, Any]) -> str:
    """Return a stable family label, preserving the rare cross-family tie."""
    return "+".join(sorted({action_family(action) for action in record["best_action_names"]}))


def _stratified_sample(
    candidates: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    count: int,
    seed: int,
    namespace: str,
) -> list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    """Round-robin strata so small policy/target-family cells remain represented."""
    groups: dict[
        tuple[str, str], list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for item in candidates:
        _, record, _ = item
        key = (record["metadata"]["trajectory_policy"], oracle_best_action_family(record))
        groups[key].append(item)
    for key, values in groups.items():
        values.sort(key=lambda item: _priority(seed, f"{namespace}:{key}", item[1]))
    selected: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    keys = sorted(groups)
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def _stratified_rehearsal(
    candidates: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]], count: int, seed: int
) -> list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for item in candidates:
        _, record, base_row = item
        key = (record["metadata"]["trajectory_policy"], action_family(base_row["base"]["prediction_action"]))
        groups[key].append(item)
    for key, values in groups.items():
        values.sort(key=lambda item: _priority(seed, f"rehearsal:{key}", item[1]))
    selected: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    family_priority = {"win": 0, "call": 1, "kong": 2, "pass": 3, "discard": 4, "invalid": 5}
    keys = sorted(groups, key=lambda key: (family_priority[key[1]], key[0], key[1]))
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def _distribution(
    items: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    policies = Counter(item[1]["metadata"]["trajectory_policy"] for item in items)
    base_actions = Counter(str(item[2]["base"]["prediction_action"]) for item in items)
    base_families = Counter(action_family(item[2]["base"]["prediction_action"]) for item in items)
    best_actions = Counter(action for _, record, _ in items for action in record["best_action_names"])
    best_families = Counter(oracle_best_action_family(record) for _, record, _ in items)
    rewards = [float(item[2]["base"]["oracle_reward"]) for item in items]
    return {
        "count": len(items),
        "trajectory_policies": dict(sorted(policies.items())),
        "base_action_families": dict(sorted(base_families.items())),
        "base_actions": dict(sorted(base_actions.items())),
        "oracle_best_actions": dict(sorted(best_actions.items())),
        "oracle_best_action_families": dict(sorted(best_families.items())),
        "base_reward": {
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "mean": sum(rewards) / len(rewards) if rewards else None,
        },
    }


def build_curriculum(  # noqa: C901 - fail-closed artifact checks belong together
    *,
    train_jsonl: Path,
    eval_jsonl: Path,
    base_predictions: Path,
    output_dir: Path,
    base_model: str = DEFAULT_MODEL,
    renderer: str = DEFAULT_RENDERER,
    temperature: float = 0.0,
    max_output_tokens: int = 12,
    sampling_seed: int = 0,
    malformed_penalty: float = -1.0,
    rehearsal_rate: float = 0.05,
    selection_seed: int = 20260717,
    selection_mode: str = "all_mistakes",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= rehearsal_rate <= 1.0:
        raise ValueError("rehearsal_rate must be in [0, 1]")
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")
    train_jsonl, eval_jsonl = Path(train_jsonl).resolve(), Path(eval_jsonl).resolve()
    base_predictions, output_dir = Path(base_predictions).resolve(), Path(output_dir).resolve()
    validation = validate_datasets(train_jsonl, eval_jsonl)
    train_records, train_seeds = read_oracle_jsonl(train_jsonl, "train")
    _, eval_seeds = read_oracle_jsonl(eval_jsonl, "eval")
    if train_seeds & eval_seeds:
        raise ValueError("train/eval seed overlap")

    summary_path = base_predictions.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset", {}).get("split") != "train":
        raise ValueError("base artifact must be a base-only split=train evaluation")
    base_rows, base_source = load_frozen_base_predictions(
        base_predictions,
        full_eval_records=train_records,
        selected_records=train_records,
        base_model=base_model,
        renderer=renderer,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=sampling_seed,
        malformed_penalty=malformed_penalty,
        eval_file_sha256=file_sha256(train_jsonl),
    )
    if len(base_rows) != len(train_records):
        raise ValueError("base artifact must cover every original training record")

    source_lines = train_jsonl.read_bytes().splitlines(keepends=True)
    if len(source_lines) != len(train_records) or any(not line.endswith(b"\n") for line in source_lines):
        raise ValueError("training JSONL must have exactly one newline-terminated source line per record")
    items = [(index, record, base_rows[index]) for index, record in enumerate(train_records)]
    informative = [item for item in items if len(set(item[1]["action_rewards"].values())) > 1]
    all_mistakes = [item for item in informative if not bool(item[2]["base"]["best_action_with_ties"])]
    anti_tsumogiri = [
        item for item in all_mistakes
        if item[2]["base"]["prediction_action"] == "TSUMOGIRI"
        and "TSUMOGIRI" not in item[1]["best_action_names"]
    ]
    other_mistakes = [item for item in all_mistakes if item not in anti_tsumogiri]
    if selection_mode == "anti_tsumogiri_balanced":
        if not anti_tsumogiri:
            raise ValueError("anti_tsumogiri_balanced requires at least one anti-TSUMOGIRI mistake")
        other_target = min(len(anti_tsumogiri), len(other_mistakes))
        balanced_others = _stratified_sample(
            other_mistakes, other_target, selection_seed, "anti_tsumogiri_other"
        )
        core = [*anti_tsumogiri, *balanced_others]
    else:
        other_target = len(other_mistakes)
        balanced_others = other_mistakes
        core = all_mistakes
    rehearsal_candidates = [item for item in informative if bool(item[2]["base"]["best_action_with_ties"])]
    rehearsal_count = min(
        len(rehearsal_candidates), math.ceil(len(rehearsal_candidates) * rehearsal_rate)
    )
    rehearsal = _stratified_rehearsal(rehearsal_candidates, rehearsal_count, selection_seed)
    selected = [*core, *rehearsal]
    if not selected:
        raise ValueError("curriculum selection is empty after informative/core/rehearsal filtering")
    selected.sort(key=lambda item: _priority(selection_seed, "final_order", item[1]))
    selected_indices = [item[0] for item in selected]
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("curriculum selection contains duplicate records")
    prompt_fingerprints = [
        json.dumps(item[1]["messages"], sort_keys=True, separators=(",", ":")) for item in selected
    ]
    if len(prompt_fingerprints) != len(set(prompt_fingerprints)):
        raise ValueError("curriculum selection contains duplicate prompts")

    manifest: dict[str, Any] = {
        "schema": CURRICULUM_VERSION,
        "selection": {
            "mode": selection_mode,
            "rule": (
                "core=all nonconstant-reward train records where frozen base is not oracle-best"
                if selection_mode == "all_mistakes" else
                "core=every base=TSUMOGIRI mistake where TSUMOGIRI is not oracle-best, plus an "
                "equal-sized set of other mistakes when available (otherwise all available), "
                "deterministically stratified by trajectory_policy and oracle-best action family"
            ) + (
                "; rehearsal=deterministic prioritized sample of nonconstant base-best records by "
                "trajectory_policy and base action family"
            ),
            "selection_seed": selection_seed,
            "rehearsal_rate_of_informative_base_best_candidates": rehearsal_rate,
            "core_records": len(core),
            "all_mistake_records": len(all_mistakes),
            "anti_tsumogiri_records": len(anti_tsumogiri),
            "other_mistake_candidate_records": len(other_mistakes),
            "other_mistake_target_records": other_target,
            "other_mistake_selected_records": len(balanced_others),
            "other_mistake_balance_cap_reason": (
                "all other mistakes selected because fewer exist than anti-TSUMOGIRI mistakes"
                if selection_mode == "anti_tsumogiri_balanced"
                and len(other_mistakes) < len(anti_tsumogiri)
                else None
            ),
            "rehearsal_candidate_records": len(rehearsal_candidates),
            "rehearsal_records": len(rehearsal),
            "selected_records": len(selected),
            "constant_reward_excluded": len(items) - len(informative),
            "easy_best_excluded_from_core": len(rehearsal_candidates),
        },
        "artifacts": {
            "original_train": {
                "path": str(train_jsonl), "sha256": file_sha256(train_jsonl),
                "fingerprint": dataset_fingerprint(train_records), "records": len(train_records),
            },
            "original_eval": {
                "path": str(eval_jsonl), "sha256": file_sha256(eval_jsonl),
                "records": validation["eval"]["records"],
            },
            "base_predictions": base_source,
        },
        "base_contract": {
            "model": base_model, "renderer": renderer, "temperature": temperature,
            "max_output_tokens": max_output_tokens, "sampling_seed": sampling_seed,
            "malformed_penalty": malformed_penalty, "split": "train",
        },
        "distributions": {
            "anti_tsumogiri": _distribution(anti_tsumogiri),
            "other_mistakes_selected": _distribution(balanced_others),
            "core": _distribution(core),
            "rehearsal": _distribution(rehearsal),
            "selected": _distribution(selected),
        },
        "eval_contract": "copied byte-identically; eval outcomes are never loaded for selection",
        "selected_identity_fingerprint": dataset_fingerprint([item[1] for item in selected]),
    }
    if dry_run:
        manifest["dry_run"] = True
        return manifest

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (output_dir / "train.jsonl", output_dir / "eval.jsonl", output_dir / "manifest.json")
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"refusing to overwrite curriculum artifacts in {output_dir}")
    temporary = output_paths[0].with_suffix(".jsonl.tmp")
    with temporary.open("wb") as output:
        for index in selected_indices:
            output.write(source_lines[index])
    temporary.replace(output_paths[0])
    shutil.copyfile(eval_jsonl, output_paths[1])
    manifest["outputs"] = {
        "train.jsonl": {"records": len(selected), "sha256": file_sha256(output_paths[0])},
        "eval.jsonl": {"records": validation["eval"]["records"], "sha256": file_sha256(output_paths[1])},
    }
    output_paths[2].write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=12)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--malformed-penalty", type=float, default=-1.0)
    parser.add_argument("--rehearsal-rate", type=float, default=0.05)
    parser.add_argument("--selection-seed", type=int, default=20260717)
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="all_mistakes")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_curriculum(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
