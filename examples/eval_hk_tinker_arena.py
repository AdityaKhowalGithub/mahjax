#!/usr/bin/env python3
"""Guarded paired gameplay arena for base and RL-checkpoint HK policies.

Gameplay outcomes are reported independently from the contextual-bandit proxy
evaluation.  The two learners start separate hands from identical game seeds
and player rotations, while all three opponents use the same frozen policy.
"""

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

import jax
import jax.numpy as jnp

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mahjax
from examples.eval_hk_tinker_sft import LEGAL_ACTION, normalize_action_output, parsed_response_text
from examples.generate_hk_rl_oracle_data import SYSTEM_PROMPT, action_name, public_state_prompt
from examples.run_hk_tinker_rl import DEFAULT_MODEL, DEFAULT_RENDERER, _load_key_for_explicit_run, validate_datasets
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.oracle import oracle_policy
from mahjax.hong_kong_mahjong.players import rule_based_player

SampleCompletion = Callable[[Sequence[Mapping[str, str]], int], Awaitable[str]]

_ENV = mahjax.make("hong_kong_mahjong", round_mode="single")
_INIT = jax.jit(_ENV.init)
_STEP = jax.jit(_ENV.step)
_RULE_BASED = jax.jit(rule_based_player)
_FAST_ORACLE = jax.jit(oracle_policy)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def text_fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def parse_game_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one comma-separated game seed")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("game seeds must be unique")
    return seeds


def validate_arena_inputs(train_jsonl: Path, eval_jsonl: Path, game_seeds: Sequence[int]) -> dict[str, Any]:
    dataset = validate_datasets(Path(train_jsonl), Path(eval_jsonl))
    frozen = set(dataset["train"]["game_seeds"]) | set(dataset["eval"]["game_seeds"])
    overlap = sorted(frozen & set(game_seeds))
    if overlap:
        raise ValueError(f"arena game seeds overlap frozen train/eval seeds: {overlap}")
    return {
        "train_jsonl": str(Path(train_jsonl).resolve()),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "train_sha256": file_sha256(train_jsonl),
        "eval_sha256": file_sha256(eval_jsonl),
        "frozen_train_seed_count": len(dataset["train"]["game_seeds"]),
        "frozen_eval_seed_count": len(dataset["eval"]["game_seeds"]),
        "arena_game_seeds": list(game_seeds),
        "arena_seed_fingerprint": text_fingerprint(",".join(map(str, game_seeds))),
        "seed_exclusion_passed": True,
    }


def _prompt_record(state: Any, legal_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": public_state_prompt(state, legal_ids)},
        ]
    }


def _deterministic_fallback(legal_ids: Sequence[int]) -> int:
    if not legal_ids:
        raise RuntimeError("active state has no legal actions")
    return min(legal_ids)


def _opponent_action(state: Any, game_seed: int, step: int, policy: str) -> int:
    if policy == "fast_oracle":
        return int(_FAST_ORACLE(state))
    if policy != "rule_based":
        raise ValueError("opponent policy must be 'rule_based' or 'fast_oracle'")
    key = jax.random.fold_in(jax.random.PRNGKey(game_seed), step * 4 + int(state.current_player))
    return int(_RULE_BASED(state, key))


async def play_hand(
    sample_learner: SampleCompletion,
    *,
    model_label: str,
    game_seed: int,
    learner_player: int,
    opponent_policy: str,
    sampling_seed: int,
    max_steps: int = 256,
) -> dict[str, Any]:
    """Play one complete hand with one sampled learner and three local opponents."""
    if learner_player not in range(4):
        raise ValueError("learner_player must be in [0, 3]")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    state = _INIT(jax.random.PRNGKey(game_seed))
    initial_dealer = int(state.round_state.dealer)
    initial_seat_wind = int(state.round_state.seat_wind[learner_player])
    decisions: list[dict[str, Any]] = []
    learner_decision = 0
    winning_action: int | None = None
    deal_in = False

    for step in range(max_steps):
        if bool(state.terminated) or bool(state.truncated):
            break
        actor = int(state.current_player)
        legal_ids = [int(item) for item in jnp.flatnonzero(state.legal_action_mask)]
        before = state
        if actor == learner_player:
            record = _prompt_record(state, legal_ids)
            decision_seed = sampling_seed + game_seed * 131 + learner_player * 17 + learner_decision
            started = time.perf_counter()
            raw_output = await sample_learner(record["messages"], decision_seed)
            latency = time.perf_counter() - started
            normalized = normalize_action_output(record, raw_output)
            legal_by_name = {action_name(action_id): action_id for action_id in legal_ids}
            predicted_name = normalized["prediction_action"]
            legal = bool(normalized["parseable_action"] and predicted_name in legal_by_name)
            action = legal_by_name[predicted_name] if legal else _deterministic_fallback(legal_ids)
            decisions.append({
                "decision": learner_decision,
                "sample_seed": decision_seed,
                "raw_output": raw_output,
                "prediction": normalized["prediction"],
                "prediction_action": predicted_name,
                "canonical_format": bool(normalized["canonical_format"]),
                "parseable_action": bool(normalized["parseable_action"]),
                "legal_action": legal,
                "invalid_output": not legal,
                "fallback_used": not legal,
                "executed_action_id": action,
                "executed_action_name": action_name(action),
                "legal_action_ids": legal_ids,
                "legal_action_names": list(legal_by_name),
                "latency_seconds": round(latency, 6),
            })
            learner_decision += 1
        else:
            action = _opponent_action(state, game_seed, step, opponent_policy)
            if action not in legal_ids:
                raise RuntimeError(f"{opponent_policy} opponent selected illegal action {action}")

        if action in (Action.RON, Action.TSUMO):
            winning_action = action
            if action == Action.RON and int(before.round_state.last_player) == learner_player and actor != learner_player:
                deal_in = True
        state = _STEP(state, jnp.int32(action))
    else:
        raise RuntimeError(f"hand {game_seed}/P{learner_player} exceeded max_steps={max_steps}")

    if not bool(state.terminated):
        raise RuntimeError(f"hand {game_seed}/P{learner_player} did not terminate")
    winners = [int(player) for player in jnp.flatnonzero(state.players.has_won)]
    won = learner_player in winners
    is_draw = not winners
    return {
        "model": model_label,
        "game_seed": game_seed,
        "learner_player": learner_player,
        "learner_seat_wind": initial_seat_wind,
        "dealer": initial_dealer,
        "opponent_policy": opponent_policy,
        "terminal_reward": float(state.rewards[learner_player]),
        "won": won,
        "ron": bool(won and winning_action == Action.RON),
        "tsumo": bool(won and winning_action == Action.TSUMO),
        "deal_in": deal_in,
        "draw": is_draw,
        "winner_players": winners,
        "terminal_steps": int(state.step_count),
        "decisions": decisions,
    }


async def run_paired_arena(
    game_seeds: Sequence[int],
    sample_base: SampleCompletion,
    sample_checkpoint: SampleCompletion,
    *,
    opponent_policy: str,
    sampling_seed: int,
    max_steps: int = 256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run matched initial hands for both models over all four player rotations."""
    hands: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for game_seed in game_seeds:
        for learner_player in range(4):
            base = await play_hand(
                sample_base,
                model_label="base",
                game_seed=game_seed,
                learner_player=learner_player,
                opponent_policy=opponent_policy,
                sampling_seed=sampling_seed,
                max_steps=max_steps,
            )
            checkpoint = await play_hand(
                sample_checkpoint,
                model_label="checkpoint",
                game_seed=game_seed,
                learner_player=learner_player,
                opponent_policy=opponent_policy,
                sampling_seed=sampling_seed,
                max_steps=max_steps,
            )
            hands.extend((base, checkpoint))
            paired.append({
                "game_seed": game_seed,
                "learner_player": learner_player,
                "learner_seat_wind": base["learner_seat_wind"],
                "base_reward": base["terminal_reward"],
                "checkpoint_reward": checkpoint["terminal_reward"],
                "reward_delta": checkpoint["terminal_reward"] - base["terminal_reward"],
            })
    return hands, paired


def _model_summary(hands: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    selected = [hand for hand in hands if hand["model"] == model]
    decisions = [decision for hand in selected for decision in hand["decisions"]]
    latencies = [float(decision["latency_seconds"]) for decision in decisions]
    rewards = [float(hand["terminal_reward"]) for hand in selected]
    return {
        "hands": len(selected),
        "terminal_reward_sum": sum(rewards),
        "mean_terminal_reward": statistics.fmean(rewards),
        "wins": sum(bool(hand["won"]) for hand in selected),
        "ron": sum(bool(hand["ron"]) for hand in selected),
        "tsumo": sum(bool(hand["tsumo"]) for hand in selected),
        "deal_ins": sum(bool(hand["deal_in"]) for hand in selected),
        "draws": sum(bool(hand["draw"]) for hand in selected),
        "decisions": len(decisions),
        "canonical_rate": statistics.fmean(float(row["canonical_format"]) for row in decisions) if decisions else 0.0,
        "legal_rate": statistics.fmean(float(row["legal_action"]) for row in decisions) if decisions else 0.0,
        "invalid_outputs": sum(bool(row["invalid_output"]) for row in decisions),
        "latency_seconds": {
            "sum": sum(latencies),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "max": max(latencies, default=0.0),
        },
    }


def seed_cluster_bootstrap(
    paired: Sequence[Mapping[str, Any]], *, samples: int = 2000, seed: int = 0
) -> dict[str, float | int]:
    if samples < 1:
        raise ValueError("bootstrap samples must be at least 1")
    clusters: dict[int, list[float]] = defaultdict(list)
    for row in paired:
        clusters[int(row["game_seed"])].append(float(row["reward_delta"]))
    means = [statistics.fmean(values) for _, values in sorted(clusters.items())]
    if not means:
        raise ValueError("cannot bootstrap empty paired arena results")
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choice(means) for _ in means) for _ in range(samples))
    return {
        "clusters": len(means),
        "samples": samples,
        "point_estimate": statistics.fmean(float(row["reward_delta"]) for row in paired),
        "ci95_lower": estimates[max(0, int(0.025 * samples) - 1)],
        "ci95_upper": estimates[min(samples - 1, int(0.975 * samples))],
    }


def summarize_arena(
    hands: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    return {
        "evaluation_type": "paired_gameplay_arena",
        "note": "Gameplay success is independent of contextual-bandit/oracle proxy success.",
        "base": _model_summary(hands, "base"),
        "checkpoint": _model_summary(hands, "checkpoint"),
        "paired_reward_delta": seed_cluster_bootstrap(paired, samples=bootstrap_samples, seed=bootstrap_seed),
        "paired_seed_seat_results": list(paired),
        "provenance": dict(provenance),
    }


def _ensure_output_available(output_dir: Path) -> None:
    paths = [Path(output_dir) / "hands.jsonl", Path(output_dir) / "summary.json"]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite arena artifacts: {', '.join(existing)}")


def write_results(output_dir: Path, hands: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    _ensure_output_available(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hands_path = Path(output_dir) / "hands.jsonl"
    with hands_path.open("w", encoding="utf-8") as output:
        for hand in hands:
            output.write(json.dumps(hand, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _local_sampler(which: str) -> SampleCompletion:
    async def sample(messages: Sequence[Mapping[str, str]], seed: int) -> str:
        del seed
        legal = LEGAL_ACTION.findall(messages[1]["content"])
        if not legal:
            return "INVALID"
        selected = legal[0] if which == "first" else legal[-1]
        return selected[1]

    return sample


async def run_remote(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                    max_tokens=args.max_output_tokens, temperature=0.0, seed=seed, stop=stop
                ),
                num_samples=1,
            )
            return parsed_response_text(renderer, response.sequences[0].tokens)

        return sample

    return await run_paired_arena(
        args.game_seeds,
        sampler(base_client),
        sampler(checkpoint_client),
        opponent_policy=args.opponent_policy,
        sampling_seed=args.sampling_seed,
        max_steps=args.max_steps,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--game-seeds", type=parse_game_seeds, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--local-smoke", action="store_true")
    mode.add_argument("--run", action="store_true", help="Explicitly launch paid Tinker sampling")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--opponent-policy", choices=("rule_based", "fast_oracle"), default="rule_based")
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_output_tokens < 1 or args.max_steps < 1 or args.bootstrap_samples < 1:
        raise ValueError("max-output-tokens, max-steps, and bootstrap-samples must be positive")
    dataset = validate_arena_inputs(args.train_jsonl, args.eval_jsonl, args.game_seeds)
    provenance = {
        **dataset,
        "base_model": args.base_model,
        "base_model_fingerprint": text_fingerprint(args.base_model),
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_fingerprint": text_fingerprint(args.checkpoint_path or "<local-smoke>"),
        "renderer": args.renderer,
        "opponent_policy": args.opponent_policy,
        "sampling_seed": args.sampling_seed,
        "temperature": 0.0,
        "max_output_tokens": args.max_output_tokens,
        "all_four_learner_players": True,
    }
    print(json.dumps({"validation": provenance}, indent=2, sort_keys=True))
    if args.validate_only:
        print("Local arena validation passed; Tinker was not imported or contacted.")
        return 0

    _ensure_output_available(args.output_dir)
    if args.local_smoke:
        hands, paired = asyncio.run(run_paired_arena(
            args.game_seeds,
            _local_sampler("first"),
            _local_sampler("last"),
            opponent_policy=args.opponent_policy,
            sampling_seed=args.sampling_seed,
            max_steps=args.max_steps,
        ))
    else:
        if not args.checkpoint_path:
            raise SystemExit("Refusing paid arena: --checkpoint-path is required")
        _load_key_for_explicit_run(args.env_file.resolve())
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("Refusing paid arena: set TINKER_API_KEY to a current credential first.")
        hands, paired = asyncio.run(run_remote(args))

    summary = summarize_arena(
        hands,
        paired,
        provenance=provenance,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.sampling_seed,
    )
    write_results(args.output_dir, hands, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
