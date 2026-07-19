#!/usr/bin/env python3
"""Generate natural HKOS win-declaration states as a training-only supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp

import mahjax

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.generate_hk_rl_oracle_data import (
    ORACLE_SCORER,
    ORACLE_SCORER_VERSION,
    SCHEMA_VERSION,
    _record,
)
from examples.generate_hk_sft_smoke import action_name
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.oracle import oracle_policy

DEFAULT_SEED_START = 900_000_000
SUPPLEMENT_VERSION = "hk_win_declaration_v1"
TRAJECTORY_POLICY = "fast_oracle_win_supplement"

_ENV = mahjax.make("hong_kong_mahjong", round_mode="single")
_INIT = jax.jit(_ENV.init)
_STEP = jax.jit(_ENV.step)
_FAST_ORACLE = jax.jit(oracle_policy)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _dataset_seeds(path: Path) -> set[int]:
    return {
        int(json.loads(line)["metadata"]["seed"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def scan_win_states(
    *,
    seed_start: int,
    target: int,
    max_games: int,
    max_steps: int,
    forbidden_seeds: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan natural all-fast-oracle hands and return exact-labeled win states."""

    if target < 1 or max_games < 1 or max_steps < 1:
        raise ValueError("target, max_games, and max_steps must be positive")
    if target > max_games:
        raise ValueError("target cannot exceed max_games because a hand has at most one declaration state")
    forbidden = forbidden_seeds or set()
    requested = set(range(seed_start, seed_start + max_games))
    overlap = requested & forbidden
    if overlap:
        raise ValueError(f"supplement seed range overlaps frozen natural data: {sorted(overlap)[:10]}")

    records: list[dict[str, Any]] = []
    scanned_seeds: list[int] = []
    total_steps_scanned = 0
    for game_offset in range(max_games):
        seed = seed_start + game_offset
        scanned_seeds.append(seed)
        state = _INIT(jax.random.PRNGKey(seed))
        for game_step in range(max_steps):
            if bool(state.terminated | state.truncated):
                break
            total_steps_scanned += 1
            has_win = bool(state.legal_action_mask[Action.RON] | state.legal_action_mask[Action.TSUMO])
            action = int(_FAST_ORACLE(state))
            if not bool(state.legal_action_mask[action]):
                raise RuntimeError(f"fast oracle selected illegal action {action} at seed {seed}, step {game_step}")
            if has_win:
                record, exact_action = _record(
                    state,
                    seed=seed,
                    split="train",
                    game_step=game_step,
                    trajectory_policy=TRAJECTORY_POLICY,
                )
                if exact_action not in (Action.RON, Action.TSUMO):
                    raise RuntimeError("exact terminal priority failed to select a legal win declaration")
                if action not in (Action.RON, Action.TSUMO):
                    raise RuntimeError("fast oracle declined a legal win declaration")
                record["metadata"].update({
                    "trajectory_action_id": action,
                    "trajectory_action_name": action_name(action),
                    "supplement_version": SUPPLEMENT_VERSION,
                    "source": "natural_all_fast_oracle_complete_hand",
                })
                records.append(record)
            state = _STEP(state, jnp.int32(action))
        else:
            raise RuntimeError(f"seed {seed} did not terminate within max_steps={max_steps}")
        if len(records) >= target:
            break

    if len(records) < target:
        raise RuntimeError(f"found only {len(records)} declaration states after scanning {len(scanned_seeds)} games")
    records = records[:target]
    identities = [(r["metadata"]["seed"], r["metadata"]["player"], r["metadata"]["step"]) for r in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate declaration-state identity")
    stats = {
        "games_scanned": len(scanned_seeds),
        "decision_steps_scanned": total_steps_scanned,
        "scanned_seed_start": scanned_seeds[0],
        "scanned_seed_end": scanned_seeds[-1],
        "represented_seeds": sorted({r["metadata"]["seed"] for r in records}),
        "records": len(records),
        "win_action_distribution": dict(sorted(Counter(r["metadata"]["trajectory_action_name"] for r in records).items())),
    }
    return records, stats


def generate_supplement(
    output_dir: Path,
    *,
    seed_start: int,
    target: int,
    max_games: int,
    max_steps: int,
    forbidden_seeds: set[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path, manifest_path = output_dir / "win_train.jsonl", output_dir / "manifest.json"
    runtime_path = output_dir / "runtime.json"
    if not overwrite and (data_path.exists() or manifest_path.exists() or runtime_path.exists()):
        raise FileExistsError(f"refusing to overwrite supplement artifacts in {output_dir}")
    started = time.perf_counter()
    records, stats = scan_win_states(
        seed_start=seed_start,
        target=target,
        max_games=max_games,
        max_steps=max_steps,
        forbidden_seeds=forbidden_seeds,
    )
    _write_jsonl(data_path, records)
    manifest = {
        "schema": SCHEMA_VERSION,
        "supplement_version": SUPPLEMENT_VERSION,
        "split": "train",
        "trajectory_policy": TRAJECTORY_POLICY,
        "scorer": ORACLE_SCORER,
        "scorer_version": ORACLE_SCORER_VERSION,
        "settings": {
            "seed_start": seed_start,
            "target": target,
            "max_games": max_games,
            "max_steps": max_steps,
        },
        "stats": stats,
        "files": {"win_train.jsonl": {"records": len(records), "sha256": _sha256(data_path)}},
        "runtime_measurement": "runtime.json is intentionally excluded from deterministic manifest hashing",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    runtime = {
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "remote_api_calls": 0,
        "remote_cost_usd": 0.0,
        "games_scanned": stats["games_scanned"],
        "decision_steps_scanned": stats["decision_steps_scanned"],
    }
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def combine_with_natural(
    *, natural_dir: Path, supplement_dir: Path, output_dir: Path, overwrite: bool = False
) -> dict[str, Any]:
    natural_dir, supplement_dir, output_dir = Path(natural_dir), Path(supplement_dir), Path(output_dir)
    natural_train, natural_eval = natural_dir / "train.jsonl", natural_dir / "eval.jsonl"
    supplement = supplement_dir / "win_train.jsonl"
    for required in (natural_train, natural_eval, natural_dir / "manifest.json", supplement, supplement_dir / "manifest.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    natural_seeds = _dataset_seeds(natural_train) | _dataset_seeds(natural_eval)
    supplement_seeds = _dataset_seeds(supplement)
    if natural_seeds & supplement_seeds:
        raise ValueError("natural and supplement seed sets overlap")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out, eval_out, manifest_out = output_dir / "train.jsonl", output_dir / "eval.jsonl", output_dir / "manifest.json"
    if not overwrite and any(path.exists() for path in (train_out, eval_out, manifest_out)):
        raise FileExistsError(f"refusing to overwrite combined artifacts in {output_dir}")
    train_out.write_bytes(natural_train.read_bytes() + supplement.read_bytes())
    shutil.copyfile(natural_eval, eval_out)
    natural_records = sum(1 for line in natural_train.open(encoding="utf-8") if line.strip())
    supplement_records = sum(1 for line in supplement.open(encoding="utf-8") if line.strip())
    eval_records = sum(1 for line in natural_eval.open(encoding="utf-8") if line.strip())
    manifest = {
        "schema": SCHEMA_VERSION,
        "version": "hk_oracle_train_v3_with_win_supplement_v1",
        "provenance": {
            "natural_manifest_sha256": _sha256(natural_dir / "manifest.json"),
            "supplement_manifest_sha256": _sha256(supplement_dir / "manifest.json"),
            "concatenation_order": ["natural_exact_v2_train", "win_supplement_v1"],
            "eval_contract": "byte-identical copy of natural exact-v2 eval",
        },
        "components": {
            "natural_train": {"records": natural_records, "sha256": _sha256(natural_train)},
            "win_supplement": {"records": supplement_records, "sha256": _sha256(supplement)},
            "natural_eval": {"records": eval_records, "sha256": _sha256(natural_eval)},
        },
        "outputs": {
            "train.jsonl": {"records": natural_records + supplement_records, "sha256": _sha256(train_out)},
            "eval.jsonl": {"records": eval_records, "sha256": _sha256(eval_out)},
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--target", type=int, default=128)
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--natural-dir", type=Path)
    parser.add_argument("--combined-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    forbidden: set[int] = set()
    if args.natural_dir:
        forbidden = _dataset_seeds(args.natural_dir / "train.jsonl") | _dataset_seeds(args.natural_dir / "eval.jsonl")
    manifest = generate_supplement(
        args.output_dir,
        seed_start=args.seed_start,
        target=args.target,
        max_games=args.max_games,
        max_steps=args.max_steps,
        forbidden_seeds=forbidden,
        overwrite=args.overwrite,
    )
    result: dict[str, Any] = {"supplement": manifest}
    if args.combined_dir:
        if not args.natural_dir:
            raise SystemExit("--combined-dir requires --natural-dir")
        result["combined"] = combine_with_natural(
            natural_dir=args.natural_dir,
            supplement_dir=args.output_dir,
            output_dir=args.combined_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
