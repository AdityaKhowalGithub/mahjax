#!/usr/bin/env python3
"""Validate HK smoke JSONL and optionally launch a small Tinker SFT run.

Validation is entirely local and deliberately imports no Tinker packages. A
remote run must be requested explicitly with ``--run`` and requires a current
credential in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

EXPECTED_SCHEMA = "mahjax.hk_sft_smoke.v1"
EXPECTED_RULESET = "hk_old_style_v1"
EXPECTED_DATA_QUALITY = "smoke_baseline"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RENDERER = "qwen3_disable_thinking"


class DatasetValidationError(ValueError):
    """Raised when an input is unsafe or incompatible with this adapter."""


@dataclass(frozen=True)
class TrainingSettings:
    train_jsonl: Path
    eval_jsonl: Path
    log_path: Path
    model: str = DEFAULT_MODEL
    renderer: str = DEFAULT_RENDERER
    lora_rank: int = 16
    batch_size: int = 8
    max_length: int = 1024
    num_epochs: int = 1
    max_steps: int | None = 3
    learning_rate: float = 1e-4
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        positive = {
            "lora_rank": self.lora_rank,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "num_epochs": self.num_epochs,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be at least 1 when provided")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def _read_jsonl(path: Path, expected_split: str) -> Tuple[List[Dict[str, Any]], Set[int]]:
    if not path.is_file():
        raise DatasetValidationError(f"{expected_split} JSONL does not exist: {path}")

    records: List[Dict[str, Any]] = []
    seeds: Set[int] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                raise DatasetValidationError(f"{path}:{line_number}: blank lines are not allowed")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            _validate_record(record, path, line_number, expected_split)
            records.append(record)
            seeds.add(record["metadata"]["seed"])

    if not records:
        raise DatasetValidationError(f"{expected_split} JSONL is empty: {path}")
    return records, seeds


def _validate_record(record: Any, path: Path, line_number: int, expected_split: str) -> None:
    location = f"{path}:{line_number}"
    if not isinstance(record, dict):
        raise DatasetValidationError(f"{location}: each line must be a JSON object")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise DatasetValidationError(f"{location}: messages must contain system, user, and assistant turns")
    expected_roles = ("system", "user", "assistant")
    for message, expected_role in zip(messages, expected_roles):
        if not isinstance(message, dict) or message.get("role") != expected_role:
            raise DatasetValidationError(f"{location}: expected message role {expected_role!r}")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise DatasetValidationError(f"{location}: message content must be a non-empty string")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise DatasetValidationError(f"{location}: metadata must be an object")
    expected_metadata = {
        "schema": EXPECTED_SCHEMA,
        "ruleset": EXPECTED_RULESET,
        "data_quality": EXPECTED_DATA_QUALITY,
        "split": expected_split,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise DatasetValidationError(f"{location}: metadata.{field} must equal {expected!r}")

    integer_fields = ("seed", "player", "seat", "step", "action_id")
    for field in integer_fields:
        if not isinstance(metadata.get(field), int) or isinstance(metadata.get(field), bool):
            raise DatasetValidationError(f"{location}: metadata.{field} must be an integer")
    if not isinstance(metadata.get("game_id"), str) or not metadata["game_id"]:
        raise DatasetValidationError(f"{location}: metadata.game_id must be a non-empty string")
    if not isinstance(metadata.get("action_name"), str) or not metadata["action_name"]:
        raise DatasetValidationError(f"{location}: metadata.action_name must be a non-empty string")
    legal_ids = metadata.get("legal_action_ids")
    if not isinstance(legal_ids, list) or not legal_ids or not all(isinstance(value, int) for value in legal_ids):
        raise DatasetValidationError(f"{location}: metadata.legal_action_ids must be a non-empty integer list")
    if metadata["action_id"] not in legal_ids:
        raise DatasetValidationError(f"{location}: selected action is not legal")
    assistant = messages[-1]["content"]
    if assistant != assistant.strip() or assistant != metadata["action_name"]:
        raise DatasetValidationError(f"{location}: assistant must contain exactly metadata.action_name")


def validate_datasets(train_jsonl: Path, eval_jsonl: Path) -> Dict[str, Any]:
    """Validate both files and enforce game-seed-level split isolation."""
    train_path = Path(train_jsonl).resolve()
    eval_path = Path(eval_jsonl).resolve()
    if train_path == eval_path:
        raise DatasetValidationError("train and eval JSONL must be different files")
    train_records, train_seeds = _read_jsonl(train_path, "train")
    eval_records, eval_seeds = _read_jsonl(eval_path, "eval")
    overlap = train_seeds & eval_seeds
    if overlap:
        raise DatasetValidationError(f"game seed leakage across train/eval: {sorted(overlap)}")
    return {
        "schema": EXPECTED_SCHEMA,
        "ruleset": EXPECTED_RULESET,
        "data_quality": EXPECTED_DATA_QUALITY,
        "train": {"records": len(train_records), "game_seeds": sorted(train_seeds)},
        "eval": {"records": len(eval_records), "game_seeds": sorted(eval_seeds)},
    }


def _public_settings(settings: TrainingSettings) -> Dict[str, Any]:
    public = asdict(settings)
    for field in ("train_jsonl", "eval_jsonl", "log_path"):
        public[field] = str(public[field])
    return public


def build_tinker_config(settings: TrainingSettings) -> Any:
    """Build the stable cookbook config without creating any service client."""
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=settings.model,
        renderer_name=settings.renderer,
        max_length=settings.max_length,
        batch_size=settings.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    class ExplicitSplitDatasetBuilder:
        def __call__(self) -> Tuple[Any, Any]:
            train_dataset, _ = FromConversationFileBuilder(
                common_config=common_config,
                file_path=str(settings.train_jsonl),
                test_size=0,
                shuffle_seed=settings.shuffle_seed,
            )()
            eval_dataset, _ = FromConversationFileBuilder(
                common_config=common_config,
                file_path=str(settings.eval_jsonl),
                test_size=0,
                shuffle_seed=settings.shuffle_seed,
            )()
            return train_dataset, eval_dataset

    return train.Config(
        log_path=str(settings.log_path),
        model_name=settings.model,
        recipe_name="mahjax_hk_sft_smoke",
        renderer_name=settings.renderer,
        dataset_builder=ExplicitSplitDatasetBuilder(),
        learning_rate=settings.learning_rate,
        num_epochs=settings.num_epochs,
        lora_rank=settings.lora_rank,
        save_every=0,
        eval_every=0,
        max_steps=settings.max_steps,
    )


def _load_key_for_explicit_run(env_file: Path) -> None:
    """Load only TINKER_API_KEY from a dotenv file without printing its value."""
    if os.environ.get("TINKER_API_KEY") or not env_file.is_file():
        return
    with env_file.open(encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if separator and name.strip() == "TINKER_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if value:
                    os.environ["TINKER_API_KEY"] = value
                return


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, default=Path("logs/hk-tinker-sft-smoke"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Credential file read only with --run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="Validate locally; never import or contact Tinker")
    mode.add_argument("--run", action="store_true", help="Explicitly launch paid remote training")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3, help="Paid-step safety cap (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = TrainingSettings(
        train_jsonl=args.train_jsonl.resolve(),
        eval_jsonl=args.eval_jsonl.resolve(),
        log_path=args.log_path.resolve(),
        model=args.model,
        renderer=args.renderer,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        shuffle_seed=args.shuffle_seed,
    )
    summary = validate_datasets(settings.train_jsonl, settings.eval_jsonl)
    print(json.dumps({"dataset": summary, "training": _public_settings(settings)}, indent=2, sort_keys=True))
    if args.validate_only:
        print("Local validation passed; Tinker was not imported or contacted.")
        return 0

    _load_key_for_explicit_run(args.env_file.resolve())
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Refusing remote training: set TINKER_API_KEY to a current credential first.")
    from tinker_cookbook.supervised import train

    asyncio.run(train.main(build_tinker_config(settings)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
