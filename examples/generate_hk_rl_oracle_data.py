#!/usr/bin/env python3
"""Generate deterministic contextual-bandit data from a hidden-information oracle.

The oracle may inspect the complete simulator state when assigning rewards, but
the model prompt is produced by the same leakage-safe serializer used by the HK
SFT smoke dataset.  Each row is one contextual-bandit decision: Tinker samples
an action from ``messages`` and the trainer looks up its scalar reward in the
top-level ``action_rewards`` mapping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp

import mahjax
from mahjax.hong_kong_mahjong.oracle import (
    EXACT_FEATURE_WEIGHTS,
    EXACT_TERMINAL_PRIORITY,
    score_legal_actions_exact,
)
from mahjax.hong_kong_mahjong.players import rule_based_player

try:
    from examples.generate_hk_sft_smoke import RULESET, action_name, public_state_prompt
except ModuleNotFoundError:  # Direct ``python examples/generate_*.py`` execution.
    from generate_hk_sft_smoke import RULESET, action_name, public_state_prompt

SCHEMA_VERSION = "mahjax.hk_rl_oracle.v1"
ORACLE_SCORER = "mahjax.hong_kong_mahjong.oracle.score_legal_actions_exact"
ORACLE_SCORER_VERSION = "hk_exact_v1"
ORACLE_FEATURE_NAMES = ("terminal_value", "shanten_progress", "live_ukeire", "hand_value", "safety")
SYSTEM_PROMPT = (
    "You play Hong Kong Old Style Mahjong (HKOS v1). "
    "Reply with exactly one canonical action token from LEGAL_ACTIONS and no other text."
)

_ENV = mahjax.make("hong_kong_mahjong", round_mode="single")
_INIT_GAME = jax.jit(_ENV.init)
_STEP_GAME = jax.jit(_ENV.step)
_HEURISTIC_POLICY = jax.jit(rule_based_player)
_SCORE_ACTIONS = jax.jit(score_legal_actions_exact)


def _jsonable(value: Any) -> Any:
    """Convert oracle diagnostics containing JAX/NumPy values to JSON data."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _score_value(value: Any) -> float:
    """Extract a scalar reward from the oracle's public score representation."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        for key in ("reward", "score", "utility", "value", "total"):
            if key in value:
                return _score_value(value[key])
    for attribute in ("reward", "score", "utility", "value", "total"):
        if hasattr(value, attribute):
            return _score_value(getattr(value, attribute))
    raise TypeError(f"oracle score is not scalar and has no scalar score field: {value!r}")


def _unpack_scores(result: Any) -> tuple[Mapping[Any, Any], Mapping[str, Any]]:
    """Accept the oracle's mapping or ``(mapping, diagnostics)`` result."""
    diagnostics: Mapping[str, Any] = {}
    scores = result
    if isinstance(result, tuple) and len(result) == 2:
        scores, diagnostics = result
    elif hasattr(result, "scores"):
        scores = result.scores
        diagnostics = getattr(result, "diagnostics", {})
    if not isinstance(scores, Mapping):
        raise TypeError("score_legal_actions must return a mapping or (mapping, diagnostics)")
    return scores, diagnostics


def _canonical_raw_scores(state: Any, legal_ids: Sequence[int]) -> tuple[dict[int, float], dict[str, Any]]:
    result = _SCORE_ACTIONS(state)
    if isinstance(result, Mapping) or isinstance(result, tuple) or hasattr(result, "scores"):
        scores, diagnostics = _unpack_scores(result)
    else:
        values = jnp.asarray(result)
        if values.ndim != 1:
            raise TypeError("score_legal_actions must return a one-dimensional array or action mapping")
        scores = {action_id: values[action_id] for action_id in legal_ids}
        diagnostics = {}
    canonical: dict[int, float] = {}
    for action_id in legal_ids:
        name = action_name(action_id)
        if action_id in scores:
            value = scores[action_id]
        elif name in scores:
            value = scores[name]
        elif str(action_id) in scores:
            value = scores[str(action_id)]
        else:
            raise ValueError(f"oracle omitted legal action {action_id}:{name}")
        canonical[action_id] = _score_value(value)
    extra_keys = []
    for key in scores:
        if key not in legal_ids and str(key) not in {str(item) for item in legal_ids}:
            if not (isinstance(key, str) and key in {action_name(item) for item in legal_ids}):
                extra_keys.append(str(key))
    if extra_keys:
        raise ValueError(f"oracle scored non-legal actions: {sorted(extra_keys)}")
    return canonical, dict(diagnostics)


def normalize_action_rewards(raw_scores: Mapping[int, float]) -> dict[int, float]:
    """Min-max normalize legal-action oracle scores into the stable [-1, 1] range."""
    if not raw_scores:
        raise ValueError("cannot normalize an empty action-score mapping")
    low, high = min(raw_scores.values()), max(raw_scores.values())
    if high == low:
        return {action_id: 0.0 for action_id in raw_scores}
    scale = high - low
    return {action_id: 2.0 * (score - low) / scale - 1.0 for action_id, score in raw_scores.items()}


def _oracle_action(state: Any, raw_scores: Mapping[int, float], legal_ids: Sequence[int]) -> int:
    """Return the exact-label argmax, breaking ties by lowest action id."""
    del state
    action = min(legal_ids, key=lambda item: (-raw_scores[item], item))
    if action not in legal_ids:
        raise RuntimeError(f"oracle_policy selected illegal action {action}")
    return action


def _record(
    state: Any,
    *,
    seed: int,
    split: str,
    game_step: int,
    trajectory_policy: str,
) -> tuple[dict[str, Any], int]:
    legal_ids = [int(item) for item in jnp.flatnonzero(state.legal_action_mask)]
    raw_scores, oracle_diagnostics = _canonical_raw_scores(state, legal_ids)
    normalized = normalize_action_rewards(raw_scores)
    best_value = max(raw_scores.values())
    best_ids = [action_id for action_id in legal_ids if raw_scores[action_id] == best_value]
    oracle_action = _oracle_action(state, raw_scores, legal_ids)

    player = int(state.current_player)
    reward_by_name = {action_name(action_id): normalized[action_id] for action_id in legal_ids}
    raw_by_name = {action_name(action_id): raw_scores[action_id] for action_id in legal_ids}
    record = {
        "schema": SCHEMA_VERSION,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": public_state_prompt(state, legal_ids)},
        ],
        "legal_actions": [{"id": action_id, "name": action_name(action_id)} for action_id in legal_ids],
        "action_rewards": reward_by_name,
        "best_action_names": [action_name(action_id) for action_id in best_ids],
        "raw_action_scores": raw_by_name,
        "oracle_diagnostics": {
            "oracle_action_id": oracle_action,
            "oracle_action_name": action_name(oracle_action),
            "score_min": min(raw_scores.values()),
            "score_max": max(raw_scores.values()),
            "scorer": ORACLE_SCORER,
            "scorer_version": ORACLE_SCORER_VERSION,
            "feature_weights": {
                name: float(value) for name, value in zip(ORACLE_FEATURE_NAMES, EXACT_FEATURE_WEIGHTS.tolist())
            },
            "terminal_priority": float(EXACT_TERMINAL_PRIORITY),
            "details": _jsonable(oracle_diagnostics),
        },
        "metadata": {
            "ruleset": RULESET,
            "data_quality": "oracle_privileged_reward",
            "split": split,
            "seed": seed,
            "game_id": f"hk-oracle-{seed}",
            "player": player,
            "step": game_step,
            "trajectory_policy": trajectory_policy,
        },
    }
    return record, oracle_action


def generate_game(seed: int, split: str, max_steps: int, records_per_game: int | None = None) -> list[dict[str, Any]]:
    """Generate a full hand, then sample its timeline at even deterministic intervals."""
    if records_per_game is not None and records_per_game < 1:
        raise ValueError("records_per_game must be at least 1 when set")
    state = _INIT_GAME(jax.random.PRNGKey(seed))
    records: list[dict[str, Any]] = []
    for game_step in range(max_steps):
        if bool(state.terminated) or bool(state.truncated):
            trajectory_steps = len(records)
            if records_per_game is not None and trajectory_steps < records_per_game:
                raise RuntimeError(
                    f"seed {seed} terminated after {trajectory_steps} decisions, "
                    f"below records_per_game={records_per_game}"
                )
            count = trajectory_steps if records_per_game is None else records_per_game
            if count == 1:
                selected_indices = [trajectory_steps - 1]
            else:
                denominator = count - 1
                selected_indices = [
                    (sample_index * (trajectory_steps - 1) + denominator // 2) // denominator
                    for sample_index in range(count)
                ]
            selected = [records[index] for index in selected_indices]
            for sample_index, record in enumerate(selected):
                record["metadata"].update({
                    "trajectory_steps": trajectory_steps,
                    "trajectory_sample_index": sample_index,
                    "trajectory_sampling": "evenly_spaced_full_trajectory_including_endpoints",
                    "is_final_decision": record["metadata"]["step"] == trajectory_steps - 1,
                })
            return selected
        use_oracle = (seed + game_step) % 2 == 0
        policy_name = "oracle" if use_oracle else "heuristic"
        record, oracle_action = _record(
            state,
            seed=seed,
            split=split,
            game_step=game_step,
            trajectory_policy=policy_name,
        )
        if use_oracle:
            action = oracle_action
        else:
            action_key = jax.random.fold_in(jax.random.PRNGKey(seed), game_step)
            action = int(_HEURISTIC_POLICY(state, action_key))
        legal_ids = [item["id"] for item in record["legal_actions"]]
        if action not in legal_ids:
            raise RuntimeError(f"{policy_name} selected illegal action {action}")
        record["metadata"]["trajectory_action_id"] = action
        record["metadata"]["trajectory_action_name"] = action_name(action)
        records.append(record)
        state = _STEP_GAME(state, jnp.int32(action))
    raise RuntimeError(f"seed {seed} did not terminate within --max-steps={max_steps}")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _split_manifest(filename: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    policies = Counter(record["metadata"]["trajectory_policy"] for record in records)
    represented_seeds = sorted({int(record["metadata"]["seed"]) for record in records})
    return {
        "file": filename,
        "game_seeds": represented_seeds,
        "games": len(represented_seeds),
        "records": len(records),
        "trajectory_policy_distribution": dict(sorted(policies.items())),
    }


def generate_dataset(
    output_dir: Path,
    *,
    seed: int,
    train_games: int,
    eval_games: int,
    max_steps: int,
    records_per_game: int | None = None,
    train_record_target: int | None = None,
    eval_record_target: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write disjoint train/eval contextual-bandit JSONL and a manifest."""
    if train_games < 1 or eval_games < 1:
        raise ValueError("train_games and eval_games must both be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if train_record_target is not None and train_record_target < 1:
        raise ValueError("train_record_target must be at least 1 when set")
    if eval_record_target is not None and eval_record_target < 1:
        raise ValueError("eval_record_target must be at least 1 when set")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in ("train.jsonl", "eval.jsonl", "manifest.json")}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing files: {', '.join(existing)}")

    train_seeds = list(range(seed, seed + train_games))
    eval_seeds = list(range(seed + train_games, seed + train_games + eval_games))
    if set(train_seeds) & set(eval_seeds):
        raise AssertionError("train and eval game seeds must be disjoint")
    train_records = [
        row
        for game_seed in train_seeds
        for row in generate_game(game_seed, "train", max_steps, records_per_game)
    ]
    eval_records = [
        row
        for game_seed in eval_seeds
        for row in generate_game(game_seed, "eval", max_steps, records_per_game)
    ]
    if train_record_target is not None:
        if len(train_records) < train_record_target:
            raise ValueError(
                f"generated only {len(train_records)} train records, below requested target {train_record_target}"
            )
        train_records = train_records[:train_record_target]
    if eval_record_target is not None:
        if len(eval_records) < eval_record_target:
            raise ValueError(
                f"generated only {len(eval_records)} eval records, below requested target {eval_record_target}"
            )
        eval_records = eval_records[:eval_record_target]

    _write_jsonl(paths["train.jsonl"], train_records)
    _write_jsonl(paths["eval.jsonl"], eval_records)
    manifest = {
        "schema": SCHEMA_VERSION,
        "ruleset": RULESET,
        "format": "contextual_bandit",
        "reward_normalization": "per_state_min_max_[-1,1]; ties_preserved; constant_scores_map_to_0",
        "oracle_reward_contract": {
            "scorer": ORACLE_SCORER,
            "scorer_version": ORACLE_SCORER_VERSION,
            "feature_weights": {
                name: float(value) for name, value in zip(ORACLE_FEATURE_NAMES, EXACT_FEATURE_WEIGHTS.tolist())
            },
            "terminal_priority": float(EXACT_TERMINAL_PRIORITY),
            "trajectory_oracle_policy": "deterministic exact-score argmax; lowest action id breaks ties",
        },
        "leakage_contract": "prompt contains actor private hand plus public state only; oracle hidden-state data is metadata",
        "settings": {
            "base_seed": seed,
            "train_games": train_games,
            "eval_games": eval_games,
            "max_steps": max_steps,
            "records_per_game": records_per_game,
            "train_record_target": train_record_target,
            "eval_record_target": eval_record_target,
            "round_mode": "single",
            "trajectory_mixture": "deterministic alternating oracle/heuristic by (seed+step)%2",
            "trajectory_sampling": (
                "simulate each hand to termination, then select approximately evenly spaced "
                "decision indices including the first and final decision"
            ),
        },
        "splits": {
            "train": _split_manifest("train.jsonl", train_records),
            "eval": _split_manifest("eval.jsonl", eval_records),
        },
        "totals": {
            "games": len({row["metadata"]["seed"] for row in train_records})
            + len({row["metadata"]["seed"] for row in eval_records}),
            "records": len(train_records) + len(eval_records),
        },
    }
    temporary_manifest = paths["manifest.json"].with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary_manifest.replace(paths["manifest.json"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-games", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--records-per-game",
        type=int,
        default=64,
        help="Cap emitted decisions per game (default: 64; 8 eval games yield up to 512 unseen states)",
    )
    parser.add_argument("--train-records", type=int, help="Require and emit exactly this many training records")
    parser.add_argument("--eval-records", type=int, help="Require and emit exactly this many evaluation records")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_dataset(
        args.output_dir,
        seed=args.seed,
        train_games=args.train_games,
        eval_games=args.eval_games,
        max_steps=args.max_steps,
        records_per_game=args.records_per_game,
        train_record_target=args.train_records,
        eval_record_target=args.eval_records,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
