#!/usr/bin/env python3
"""Validate HK oracle JSONL and optionally launch a guarded Tinker RLVR run."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.eval_hk_tinker_sft import LEGAL_ACTION, normalize_action_output
from examples.run_hk_tinker_sft import _load_key_for_explicit_run

EXPECTED_SCHEMA = "mahjax.hk_rl_oracle.v1"
EXPECTED_RULESET = "hk_old_style_v1"
EXPECTED_DATA_QUALITY = "oracle_privileged_reward"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RENDERER = "qwen3_disable_thinking"
CANONICAL_ACTION = re.compile(r"[A-Z][A-Z0-9_]*")
PROVENANCE_SCHEMA = "mahjax.hk_tinker_provenance.v1"


def _tile_name(tile: int) -> str:
    if 0 <= tile < 9:
        return f"{tile + 1}M"
    if 9 <= tile < 18:
        return f"{tile - 8}P"
    if 18 <= tile < 27:
        return f"{tile - 17}S"
    if 27 <= tile < 34:
        return ("E", "S", "W", "N", "WHITE", "GREEN", "RED")[tile - 27]
    raise ValueError(f"not a standard tile id: {tile}")


def hk_action_name(action_id: int) -> str:
    """Exact supported MahJax HK action-id to canonical-token mapping."""
    if 0 <= action_id < 34:
        return f"DISCARD_{_tile_name(action_id)}"
    if 34 <= action_id < 68:
        return f"SELF_KONG_{_tile_name(action_id - 34)}"
    fixed = {
        68: "TSUMOGIRI",
        70: "TSUMO",
        71: "RON",
        72: "PON",
        73: "OPEN_KONG",
        74: "CHOW_LEFT",
        75: "CHOW_MIDDLE",
        76: "CHOW_RIGHT",
        77: "PASS",
    }
    if action_id not in fixed:
        raise ValueError(f"unsupported Hong Kong action id: {action_id}")
    return fixed[action_id]


class OracleValidationError(ValueError):
    """Raised for oracle data that is unsafe to train or evaluate on."""


@dataclass(frozen=True)
class RLSettings:
    train_jsonl: Path
    eval_jsonl: Path
    log_path: Path
    model: str = DEFAULT_MODEL
    renderer: str = DEFAULT_RENDERER
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.05
    lora_rank: int = 16
    max_output_tokens: int = 12
    max_steps: int = 3
    temperature: float = 1.0
    loss_fn: str = "importance_sampling"
    loss_fn_config: Mapping[str, Any] | None = None
    shuffle_seed: int = 0
    malformed_penalty: float = -1.0

    def __post_init__(self) -> None:
        for name in ("group_size", "groups_per_batch", "lora_rank", "max_output_tokens", "max_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.temperature) or not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be finite and in (0, 2]")
        if self.kl_penalty_coef < 0:
            raise ValueError("kl_penalty_coef cannot be negative")
        if self.loss_fn not in {"importance_sampling", "cispo"}:
            raise ValueError("loss_fn must be 'importance_sampling' or 'cispo'")
        if not -1.0 <= self.malformed_penalty <= 1.0:
            raise ValueError("malformed_penalty must be in [-1, 1]")


class HKOracleMessageEnv:
    """Pickle-safe, cookbook-compatible message-level oracle environment."""

    def __init__(self, record: Mapping[str, Any], malformed_penalty: float):
        self.record = record
        self.malformed_penalty = malformed_penalty
        metadata = record["metadata"]
        self.example_id = f"{metadata['game_id']}:{metadata['player']}:{metadata['step']}"

    async def initial_observation(self) -> list[dict[str, str]]:
        return list(self.record["messages"])

    async def step(self, message: Mapping[str, Any]) -> Any:
        from tinker_cookbook.renderers import get_text_content
        from tinker_cookbook.rl.message_env import MessageStepResult

        scored = score_action(self.record, get_text_content(message), self.malformed_penalty)
        return MessageStepResult(
            reward=scored["oracle_reward"],
            episode_done=True,
            next_messages=[*self.record["messages"], message],
            metrics={
                "parseable": float(scored["parseable_action"]),
                "legal": float(scored["legal_action"]),
                "canonical": float(scored["canonical_format"]),
                "best_action": float(scored["best_action"]),
            },
        )


class HKGroupBuilder:
    def __init__(
        self,
        record: Mapping[str, Any],
        renderer: Any,
        group_size: int,
        malformed_penalty: float,
        max_output_tokens: int,
    ):
        self.record = record
        self.renderer = renderer
        self.group_size = group_size
        self.malformed_penalty = malformed_penalty
        self.max_output_tokens = max_output_tokens

    async def make_envs(self) -> list[Any]:
        from tinker_cookbook.rl.message_env import EnvFromMessageEnv

        return [
            EnvFromMessageEnv(
                self.renderer,
                HKOracleMessageEnv(self.record, self.malformed_penalty),
                failed_parse_reward=self.malformed_penalty,
                context_overflow_reward=self.malformed_penalty,
                max_generation_tokens=self.max_output_tokens,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: Sequence[Any], env_group: Sequence[Any]
    ) -> list[tuple[float, dict[str, float | int]]]:
        """All oracle reward is assigned by each one-step environment."""
        if len(trajectory_group) != len(env_group):
            raise ValueError("trajectory and environment group lengths must match")
        return [(0.0, {}) for _ in trajectory_group]

    async def cleanup(self) -> None:
        """Satisfy the cookbook lifecycle; this in-memory environment owns no resources."""
        return None

    def logging_tags(self) -> list[str]:
        return ["mahjax", "hk_rl_oracle"]


class HKDataset:
    def __init__(
        self,
        records: list[dict[str, Any]],
        renderer: Any,
        *,
        group_size: int,
        groups_per_batch: int,
        malformed_penalty: float,
        max_output_tokens: int,
        shuffle_seed: int,
    ):
        order = list(range(len(records)))
        random.Random(shuffle_seed).shuffle(order)
        self.groups_per_batch = groups_per_batch
        self.builders = [
            HKGroupBuilder(
                records[index], renderer, group_size, malformed_penalty, max_output_tokens
            )
            for index in order
        ]

    def get_batch(self, index: int) -> list[HKGroupBuilder]:
        start = index * self.groups_per_batch
        return self.builders[start : start + self.groups_per_batch]

    def __len__(self) -> int:
        return math.ceil(len(self.builders) / self.groups_per_batch)


@dataclass(frozen=True)
class HKDatasetBuilder:
    records: list[dict[str, Any]]
    model: str
    renderer_name: str
    group_size: int
    groups_per_batch: int
    malformed_penalty: float
    max_output_tokens: int
    shuffle_seed: int

    async def __call__(self) -> tuple[HKDataset, None]:
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer, model_name=self.model)
        return HKDataset(
            self.records,
            renderer,
            group_size=self.group_size,
            groups_per_batch=self.groups_per_batch,
            malformed_penalty=self.malformed_penalty,
            max_output_tokens=self.max_output_tokens,
            shuffle_seed=self.shuffle_seed,
        ), None


def _prompt_legal_actions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt = record["messages"][1]["content"]
    return [{"id": int(action_id), "name": name} for action_id, name in LEGAL_ACTION.findall(prompt)]


def _validate_record(  # noqa: C901 - keeping all cross-field schema checks together is clearer
    record: Any, path: Path, line_number: int, expected_split: str
) -> None:
    where = f"{path}:{line_number}"
    if not isinstance(record, dict):
        raise OracleValidationError(f"{where}: each line must be a JSON object")
    if record.get("schema") != EXPECTED_SCHEMA:
        raise OracleValidationError(f"{where}: schema must equal {EXPECTED_SCHEMA!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise OracleValidationError(f"{where}: messages must contain exactly system and user turns")
    for message, role in zip(messages, ("system", "user")):
        if not isinstance(message, dict) or message.get("role") != role:
            raise OracleValidationError(f"{where}: expected message role {role!r}")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise OracleValidationError(f"{where}: message content must be a non-empty string")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise OracleValidationError(f"{where}: metadata must be an object")
    required = {
        "ruleset": EXPECTED_RULESET,
        "data_quality": EXPECTED_DATA_QUALITY,
        "split": expected_split,
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise OracleValidationError(f"{where}: metadata.{field} must equal {expected!r}")
    for field in ("seed", "player", "step"):
        if not isinstance(metadata.get(field), int) or isinstance(metadata.get(field), bool):
            raise OracleValidationError(f"{where}: metadata.{field} must be an integer")
    for field in ("game_id", "trajectory_policy"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise OracleValidationError(f"{where}: metadata.{field} must be a non-empty string")

    legal_actions = record.get("legal_actions")
    if not isinstance(legal_actions, list) or not legal_actions:
        raise OracleValidationError(f"{where}: legal_actions must be a non-empty list")
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for action in legal_actions:
        if not isinstance(action, dict):
            raise OracleValidationError(f"{where}: each legal action must be an object")
        action_id, name = action.get("id"), action.get("name")
        if not isinstance(action_id, int) or isinstance(action_id, bool):
            raise OracleValidationError(f"{where}: legal action id must be an integer")
        if not isinstance(name, str) or not CANONICAL_ACTION.fullmatch(name):
            raise OracleValidationError(f"{where}: illegal canonical action name {name!r}")
        try:
            expected_name = hk_action_name(action_id)
        except ValueError as error:
            raise OracleValidationError(f"{where}: {error}") from error
        if name != expected_name:
            raise OracleValidationError(
                f"{where}: action id {action_id} must map to {expected_name!r}, not {name!r}"
            )
        if action_id in seen_ids or name in seen_names:
            raise OracleValidationError(f"{where}: legal action ids and names must be unique")
        seen_ids.add(action_id)
        seen_names.add(name)
    if legal_actions != _prompt_legal_actions(record):
        raise OracleValidationError(f"{where}: legal_actions must exactly match prompt LEGAL_ACTIONS order")

    rewards = record.get("action_rewards")
    if not isinstance(rewards, dict) or set(rewards) != seen_names:
        raise OracleValidationError(f"{where}: action_rewards keys must exactly equal legal action names")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           or not -1.0 <= float(value) <= 1.0 for value in rewards.values()):
        raise OracleValidationError(f"{where}: action_rewards values must be finite numbers in [-1, 1]")
    best = record.get("best_action_names")
    if not isinstance(best, list) or not best or len(set(best)) != len(best) or not set(best) <= seen_names:
        raise OracleValidationError(f"{where}: best_action_names must be a non-empty unique legal subset")
    maximum = max(float(value) for value in rewards.values())
    expected_best = {name for name, value in rewards.items() if math.isclose(float(value), maximum, abs_tol=1e-9)}
    if set(best) != expected_best:
        raise OracleValidationError(f"{where}: best_action_names does not match maximum action_rewards")


def read_oracle_jsonl(path: Path, expected_split: str) -> tuple[list[dict[str, Any]], set[int]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise OracleValidationError(f"{expected_split} JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    seeds: set[int] = set()
    identities: set[tuple[int, str, int, int]] = set()
    prompts: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise OracleValidationError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise OracleValidationError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            _validate_record(record, path, line_number, expected_split)
            metadata = record["metadata"]
            identity = (metadata["seed"], metadata["game_id"], metadata["player"], metadata["step"])
            if identity in identities:
                raise OracleValidationError(f"{path}:{line_number}: duplicate decision identity {identity!r}")
            prompt = json.dumps(record["messages"], sort_keys=True, separators=(",", ":"))
            if prompt in prompts:
                raise OracleValidationError(f"{path}:{line_number}: duplicate prompt within {expected_split} split")
            identities.add(identity)
            prompts.add(prompt)
            records.append(record)
            seeds.add(record["metadata"]["seed"])
    if not records:
        raise OracleValidationError(f"{expected_split} JSONL is empty: {path}")
    return records, seeds


def validate_datasets(train_jsonl: Path, eval_jsonl: Path) -> dict[str, Any]:
    train_path, eval_path = Path(train_jsonl).resolve(), Path(eval_jsonl).resolve()
    if train_path == eval_path:
        raise OracleValidationError("train and eval JSONL must be different files")
    train, train_seeds = read_oracle_jsonl(train_path, "train")
    evaluation, eval_seeds = read_oracle_jsonl(eval_path, "eval")
    overlap = train_seeds & eval_seeds
    if overlap:
        raise OracleValidationError(f"game seed leakage across train/eval: {sorted(overlap)}")
    return {
        "schema": EXPECTED_SCHEMA,
        "train": {"records": len(train), "game_seeds": sorted(train_seeds)},
        "eval": {"records": len(evaluation), "game_seeds": sorted(eval_seeds)},
    }


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _ordered_record_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def public_rl_config(settings: RLSettings) -> dict[str, Any]:
    public = asdict(settings)
    for key in ("train_jsonl", "eval_jsonl", "log_path"):
        public[key] = str(public[key])
    if public["loss_fn_config"] is not None:
        public["loss_fn_config"] = dict(public["loss_fn_config"])
    return public


def build_run_identity(settings: RLSettings) -> dict[str, Any]:
    train, train_seeds = read_oracle_jsonl(settings.train_jsonl, "train")
    evaluation, eval_seeds = read_oracle_jsonl(settings.eval_jsonl, "eval")

    def descriptor(path: Path, records: Sequence[Mapping[str, Any]], seeds: set[int]) -> dict[str, Any]:
        return {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "ordered_records_fingerprint": _ordered_record_fingerprint(records),
            "count": len(records),
            "seeds": sorted(seeds),
        }

    return {
        "base_model": settings.model,
        "renderer": settings.renderer,
        "datasets": {
            "train": descriptor(settings.train_jsonl, train, train_seeds),
            "eval": descriptor(settings.eval_jsonl, evaluation, eval_seeds),
        },
        "config": public_rl_config(settings),
    }


def _source_git_info(repo_dir: Path | None = None) -> dict[str, Any]:
    cwd = repo_dir or Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments], cwd=cwd, text=True, capture_output=True, check=False
        )

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"provenance file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid provenance JSON at {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"provenance at {path} must be a JSON object")
    return value


def preflight_log_path(settings: RLSettings, *, resume: bool) -> dict[str, Any]:
    """Refuse implicit cookbook resume and validate the only supported resume path."""
    log_path = settings.log_path
    existing = list(log_path.iterdir()) if log_path.is_dir() else []
    if not resume:
        if existing:
            raise RuntimeError(
                f"refusing fresh run in non-empty log directory {log_path}; "
                "choose a new --log-path or use --resume after auditing provenance"
            )
        return build_run_identity(settings)
    if not existing:
        raise RuntimeError(f"cannot resume empty or missing log directory: {log_path}")
    manifest = _load_json_object(log_path / "provenance.json")
    identity = build_run_identity(settings)
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "base_model": identity["base_model"],
        "renderer": identity["renderer"],
        "datasets": identity["datasets"],
        "config": identity["config"],
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(f"refusing resume: provenance mismatch in {', '.join(mismatches)}")
    if not manifest.get("state_checkpoint_path"):
        raise RuntimeError("refusing resume: existing provenance has no resumable state checkpoint")
    return identity


def _checkpoint_field(checkpoint: Any, field: str) -> Any:
    if checkpoint is None:
        return None
    if isinstance(checkpoint, Mapping):
        return checkpoint.get(field)
    return getattr(checkpoint, field, None)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


async def train_with_provenance(
    settings: RLSettings,
    *,
    train_main: Any,
    get_last_checkpoint: Any,
    resume: bool = False,
    config_builder: Any = None,
    git_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run training and atomically bind its final checkpoint to inputs/config."""
    identity = preflight_log_path(settings, resume=resume)
    if config_builder is None:
        config_builder = build_tinker_config
    started_wall = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    await train_main(config_builder(settings))
    sampler_checkpoint = get_last_checkpoint(str(settings.log_path), required_key="sampler_path")
    sampler_path = _checkpoint_field(sampler_checkpoint, "sampler_path")
    sampler_name = _checkpoint_field(sampler_checkpoint, "name")
    if not sampler_path or sampler_name != "final":
        raise RuntimeError(
            f"training returned successfully but no final sampler checkpoint was found in {settings.log_path}"
        )
    state_checkpoint = get_last_checkpoint(str(settings.log_path), required_key="state_path")
    state_path = _checkpoint_field(state_checkpoint, "state_path")
    ended_at = datetime.now(timezone.utc)
    manifest = {
        "schema": PROVENANCE_SCHEMA,
        "checkpoint_path": sampler_path,
        "sampler_checkpoint_path": sampler_path,
        "state_checkpoint_path": state_path,
        "base_model": identity["base_model"],
        "renderer": identity["renderer"],
        "train_sha256": identity["datasets"]["train"]["file_sha256"],
        "eval_sha256": identity["datasets"]["eval"]["file_sha256"],
        "datasets": identity["datasets"],
        "config": identity["config"],
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "runtime_seconds": round(time.perf_counter() - started_wall, 6),
        "source_git": dict(git_info) if git_info is not None else _source_git_info(),
    }
    _write_json_atomic(settings.log_path / "provenance.json", manifest)
    return manifest


def score_action(record: Mapping[str, Any], raw_output: str, malformed_penalty: float = -1.0) -> dict[str, Any]:
    parsed = normalize_action_output(record, raw_output)
    action = parsed["prediction_action"]
    legal = bool(parsed["parseable_action"] and action in record["action_rewards"])
    reward = float(record["action_rewards"][action]) if legal else float(malformed_penalty)
    return {
        **parsed,
        "legal_action": legal,
        "oracle_reward": reward,
        "best_action": bool(legal and action in record["best_action_names"]),
    }


def build_tinker_config(settings: RLSettings) -> Any:
    """Construct cookbook 0.5.2 objects; no client or remote call is created here."""
    from tinker_cookbook.rl.train import Config, KLReferenceConfig

    train_records, _ = read_oracle_jsonl(settings.train_jsonl, "train")

    return Config(
        learning_rate=settings.learning_rate,
        dataset_builder=HKDatasetBuilder(
            records=train_records,
            model=settings.model,
            renderer_name=settings.renderer,
            group_size=settings.group_size,
            groups_per_batch=settings.groups_per_batch,
            malformed_penalty=settings.malformed_penalty,
            max_output_tokens=settings.max_output_tokens,
            shuffle_seed=settings.shuffle_seed,
        ),
        model_name=settings.model,
        recipe_name="mahjax_hk_rl_oracle",
        renderer_name=settings.renderer,
        log_path=str(settings.log_path),
        max_tokens=settings.max_output_tokens,
        max_steps=settings.max_steps,
        lora_rank=settings.lora_rank,
        temperature=settings.temperature,
        kl_penalty_coef=settings.kl_penalty_coef,
        kl_reference_config=KLReferenceConfig(base_model=settings.model) if settings.kl_penalty_coef else None,
        loss_fn=settings.loss_fn,
        loss_fn_config=dict(settings.loss_fn_config) if settings.loss_fn_config else None,
        remove_constant_reward_groups=True,
        eval_every=0,
        save_every=0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, default=Path("logs/hk-tinker-rl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true", help="Explicitly launch paid remote RL training")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=3, help="Required paid-step safety cap")
    parser.add_argument("--temperature", type=float, default=1.0, help="RL rollout sampling temperature")
    parser.add_argument("--loss-fn", choices=("importance_sampling", "cispo"), default="importance_sampling")
    parser.add_argument("--loss-fn-config", type=json.loads)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--malformed-penalty", type=float, default=-1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly resume only when existing provenance exactly matches this run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = RLSettings(**{
        "train_jsonl": args.train_jsonl.resolve(), "eval_jsonl": args.eval_jsonl.resolve(),
        "log_path": args.log_path.resolve(), "model": args.model, "renderer": args.renderer,
        "group_size": args.group_size, "groups_per_batch": args.groups_per_batch,
        "learning_rate": args.learning_rate, "kl_penalty_coef": args.kl_penalty_coef,
        "lora_rank": args.lora_rank, "max_output_tokens": args.max_output_tokens,
        "max_steps": args.max_steps, "temperature": args.temperature,
        "loss_fn": args.loss_fn, "loss_fn_config": args.loss_fn_config,
        "shuffle_seed": args.shuffle_seed, "malformed_penalty": args.malformed_penalty,
    })
    summary = validate_datasets(settings.train_jsonl, settings.eval_jsonl)
    public = public_rl_config(settings)
    print(json.dumps({"dataset": summary, "training": public}, indent=2, sort_keys=True))
    if args.validate_only:
        print("Local validation passed; Tinker was not imported or contacted.")
        return 0
    try:
        preflight_log_path(settings, resume=args.resume)
    except RuntimeError as error:
        raise SystemExit(f"Refusing remote training: {error}") from error
    _load_key_for_explicit_run(args.env_file.resolve())
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Refusing remote training: set TINKER_API_KEY to a current credential first.")
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.rl import train

    manifest = asyncio.run(
        train_with_provenance(
            settings,
            train_main=train.main,
            get_last_checkpoint=checkpoint_utils.get_last_checkpoint,
            resume=args.resume,
        )
    )
    print(
        json.dumps(
            {
                "provenance_path": str(settings.log_path / "provenance.json"),
                "checkpoint_path": manifest["checkpoint_path"],
                "state_checkpoint_path": manifest["state_checkpoint_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
