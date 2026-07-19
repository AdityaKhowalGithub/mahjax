#!/usr/bin/env python3
"""Generate small, deterministic Hong Kong Mahjong SFT smoke datasets.

The labels come from the basic legal rule-based player. They are useful for
testing an SFT pipeline, but they are not expert-quality Mahjong supervision.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import jax
import jax.numpy as jnp

import mahjax
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.meld import Meld
from mahjax.hong_kong_mahjong.players import rule_based_player
from mahjax.hong_kong_mahjong.tile import Tile

SCHEMA_VERSION = "mahjax.hk_sft_smoke.v1"
RULESET = "hk_old_style_v1"
DATA_QUALITY = "smoke_baseline"
SYSTEM_PROMPT = (
    "You play Hong Kong Old Style Mahjong (HKOS v1). "
    "Reply with exactly one canonical action token from LEGAL_ACTIONS and no other text."
)

_WINDS = ("E", "S", "W", "N")
_FLOWERS = ("PLUM", "ORCHID", "CHRYSANTHEMUM", "BAMBOO", "SPRING", "SUMMER", "AUTUMN", "WINTER")

# Compile once per process, then reuse for every seed. This keeps the smoke
# generator practical on CPU-only Macs without changing trajectory semantics.
_SMOKE_ENV = mahjax.make("hong_kong_mahjong", round_mode="single")
_INIT_GAME = jax.jit(_SMOKE_ENV.init)
_STEP_GAME = jax.jit(_SMOKE_ENV.step)
_CHOOSE_ACTION = jax.jit(rule_based_player)


def tile_name(tile: int) -> str:
    """Return an unambiguous compact name for a tile type."""
    if 0 <= tile < 9:
        return f"{tile + 1}M"
    if 9 <= tile < 18:
        return f"{tile - 8}P"
    if 18 <= tile < 27:
        return f"{tile - 17}S"
    if 27 <= tile < 34:
        return ("E", "S", "W", "N", "WHITE", "GREEN", "RED")[tile - 27]
    raise ValueError(f"not a standard tile: {tile}")


def action_name(action: int) -> str:
    """Map a MahJax action id to one canonical SFT token."""
    if 0 <= action < 34:
        return f"DISCARD_{tile_name(action)}"
    if 34 <= action < 68:
        return f"SELF_KONG_{tile_name(action - 34)}"
    names = {
        Action.TSUMOGIRI: "TSUMOGIRI",
        Action.TSUMO: "TSUMO",
        Action.RON: "RON",
        Action.PON: "PON",
        Action.OPEN_KAN: "OPEN_KONG",
        Action.CHI_L: "CHOW_LEFT",
        Action.CHI_M: "CHOW_MIDDLE",
        Action.CHI_R: "CHOW_RIGHT",
        Action.PASS: "PASS",
    }
    if action not in names:
        raise ValueError(f"unsupported Hong Kong action id: {action}")
    return names[action]


def _tiles_from_counts(counts: Sequence[int]) -> str:
    tiles = [tile_name(tile) for tile, count in enumerate(counts) for _ in range(int(count))]
    return "[" + ",".join(tiles) + "]"


def _flowers_from_mask(mask: Sequence[bool]) -> str:
    return "[" + ",".join(name for name, present in zip(_FLOWERS, mask) if bool(present)) + "]"


def _meld_name(encoded: int) -> str:
    encoded_array = jnp.uint16(encoded)
    action = int(Meld.action(encoded_array))
    target = int(Meld.target(encoded_array))
    source = int(Meld.src(encoded_array))
    return f"{action_name(action)}({tile_name(target)},src={source})"


def _melds_for_player(state: Any, player: int) -> str:
    count = int(state.players.meld_counts[player])
    return "[" + ",".join(_meld_name(int(state.players.melds[player, i])) for i in range(count)) + "]"


def _river_for_player(state: Any, player: int) -> str:
    count = int(state.players.discard_counts[player])
    return "[" + ",".join(tile_name(int(state.players.river[player, i])) for i in range(count)) + "]"


def public_state_prompt(state: Any, legal_action_ids: Sequence[int]) -> str:
    """Serialize the acting player's private view and all public game state."""
    player = int(state.current_player)
    round_state = state.round_state
    lines = [
        f"RULESET={RULESET}",
        (
            f"ROUND={int(round_state.round)} PREVALENT={_WINDS[int(round_state.prevalent_wind)]} "
            f"DEALER=P{int(round_state.dealer)} ACTOR=P{player} "
            f"SEAT={_WINDS[int(round_state.seat_wind[player])]} WALL={Tile.NUM_TILE_ID - int(round_state.wall_index)}"
        ),
        (
            f"CONTEXT=last_player:P{int(round_state.last_player)},target:"
            f"{tile_name(int(round_state.target)) if int(round_state.target) >= 0 else 'NONE'},"
            f"after_kong:{int(round_state.after_kong)},robbing_kong:{int(round_state.robbing_kong)}"
        ),
        f"LAST_DRAW={tile_name(int(round_state.last_draw)) if int(round_state.last_draw) >= 0 else 'NONE'}",
        f"HAND={_tiles_from_counts(state.players.hand[player])}",
        f"ACTOR_FLOWERS={_flowers_from_mask(state.players.flowers[player])}",
    ]
    for seat in range(4):
        lines.append(
            f"P{seat}=score:{int(round_state.score[seat])},seat:{_WINDS[int(round_state.seat_wind[seat])]},"
            f"melds:{_melds_for_player(state, seat)},flowers:{_flowers_from_mask(state.players.flowers[seat])},"
            f"river:{_river_for_player(state, seat)}"
        )
    legal = ",".join(f"{action}:{action_name(action)}" for action in legal_action_ids)
    lines.append(f"LEGAL_ACTIONS=[{legal}]")
    return "\n".join(lines)


def _record(state: Any, action: int, seed: int, split: str, game_step: int) -> Dict[str, Any]:
    legal_ids = [int(i) for i in jnp.flatnonzero(state.legal_action_mask)]
    if action not in legal_ids:
        raise RuntimeError(f"heuristic selected illegal action {action} for seed {seed}, step {game_step}")
    token = action_name(action)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": public_state_prompt(state, legal_ids)},
            {"role": "assistant", "content": token},
        ],
        "metadata": {
            "schema": SCHEMA_VERSION,
            "ruleset": RULESET,
            "data_quality": DATA_QUALITY,
            "split": split,
            "seed": seed,
            "game_id": f"hk-smoke-{seed}",
            "seat": int(state.round_state.seat_wind[int(state.current_player)]),
            "player": int(state.current_player),
            "step": game_step,
            "action_id": action,
            "action_name": token,
            "legal_action_ids": legal_ids,
        },
    }


def generate_game(seed: int, split: str, max_steps: int) -> List[Dict[str, Any]]:
    """Generate one deterministic single-hand trajectory."""
    state = _INIT_GAME(jax.random.PRNGKey(seed))
    records: List[Dict[str, Any]] = []
    for game_step in range(max_steps):
        if bool(state.terminated) or bool(state.truncated):
            return records
        action_key = jax.random.fold_in(jax.random.PRNGKey(seed), game_step)
        action = int(_CHOOSE_ACTION(state, action_key))
        records.append(_record(state, action, seed, split, game_step))
        state = _STEP_GAME(state, jnp.int32(action))
    raise RuntimeError(f"seed {seed} did not terminate within --max-steps={max_steps}")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _distribution(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(record["metadata"]["action_name"]) for record in records)
    return dict(sorted(counts.items()))


def generate_dataset(
    output_dir: Path,
    *,
    seed: int,
    train_games: int,
    eval_games: int,
    max_steps: int,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate train/eval JSONL files and return the written manifest."""
    if train_games < 1 or eval_games < 1:
        raise ValueError("train_games and eval_games must both be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in ("train.jsonl", "eval.jsonl", "manifest.json")}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing files: {', '.join(existing)}")

    train_seeds = list(range(seed, seed + train_games))
    eval_seeds = list(range(seed + train_games, seed + train_games + eval_games))
    started = time.perf_counter()
    train_records = [record for game_seed in train_seeds for record in generate_game(game_seed, "train", max_steps)]
    eval_records = [record for game_seed in eval_seeds for record in generate_game(game_seed, "eval", max_steps)]
    _write_jsonl(paths["train.jsonl"], train_records)
    _write_jsonl(paths["eval.jsonl"], eval_records)

    all_records = train_records + eval_records
    manifest = {
        "schema": SCHEMA_VERSION,
        "ruleset": RULESET,
        "data_quality": DATA_QUALITY,
        "warning": "Smoke/baseline data from a simple legal heuristic; not expert-quality supervision.",
        "settings": {
            "base_seed": seed,
            "train_games": train_games,
            "eval_games": eval_games,
            "max_steps": max_steps,
            "round_mode": "single",
            "policy": "mahjax.hong_kong_mahjong.players.rule_based_player",
        },
        "splits": {
            "train": {
                "file": "train.jsonl",
                "game_seeds": train_seeds,
                "games": len(train_seeds),
                "records": len(train_records),
                "action_distribution": _distribution(train_records),
            },
            "eval": {
                "file": "eval.jsonl",
                "game_seeds": eval_seeds,
                "games": len(eval_seeds),
                "records": len(eval_records),
                "action_distribution": _distribution(eval_records),
            },
        },
        "totals": {"games": train_games + eval_games, "records": len(all_records)},
        "generation_seconds": round(time.perf_counter() - started, 3),
    }
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="First root game seed (default: 0)")
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=256)
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
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
