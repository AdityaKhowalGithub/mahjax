#!/usr/bin/env python3
"""Evaluate exact Hong Kong Mahjong actions through a Tinker sampler."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Sequence

from examples.run_hk_tinker_sft import _load_key_for_explicit_run, validate_datasets

CANONICAL_ACTION = re.compile(r"[A-Z][A-Z0-9_]*")
LEGAL_ACTION = re.compile(r"(\d+):([A-Z][A-Z0-9_]*)")
ID_AND_ACTION = re.compile(r"(\d+):([A-Z][A-Z0-9_]*)")
SampleCompletion = Callable[[Sequence[Mapping[str, str]], int], Awaitable[str]]


def load_eval_records(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        records = records[:limit]
    if not records:
        raise ValueError("evaluation selection is empty")
    return records


def legal_action_names(record: Mapping[str, Any]) -> List[str]:
    prompt = record["messages"][1]["content"]
    names = [name for _, name in LEGAL_ACTION.findall(prompt)]
    if not names:
        raise ValueError("record has no canonical legal action names in its prompt")
    return names


def normalize_action_output(record: Mapping[str, Any], raw_output: str) -> Dict[str, Any]:
    """Normalize canonical names, legal numeric ids, and consistent id:name pairs."""
    prediction = raw_output.strip()
    prompt = record["messages"][1]["content"]
    legal_by_id = {int(action_id): name for action_id, name in LEGAL_ACTION.findall(prompt)}
    canonical = bool(CANONICAL_ACTION.fullmatch(prediction))
    prediction_action: str | None = prediction if canonical else None
    parseable = canonical

    if prediction.isdigit():
        prediction_action = legal_by_id.get(int(prediction))
        parseable = prediction_action is not None
    else:
        id_and_action = ID_AND_ACTION.fullmatch(prediction)
        if id_and_action:
            action_id, action_name = int(id_and_action.group(1)), id_and_action.group(2)
            expected_name = legal_by_id.get(action_id)
            parseable = expected_name is not None and expected_name == action_name
            prediction_action = expected_name if parseable else None

    return {
        "prediction": prediction,
        "prediction_action": prediction_action,
        "canonical_format": canonical,
        "parseable_action": parseable,
    }


def parsed_response_text(renderer: Any, tokens: Sequence[int]) -> str:
    """Extract assistant text through the renderer, excluding response framing."""
    message, _ = renderer.parse_response(list(tokens))
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part.get("type") == "text")


async def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    sample_completion: SampleCompletion,
    *,
    seed: int,
    concurrency: int,
) -> List[Dict[str, Any]]:
    """Evaluate records with bounded concurrency and deterministic per-row seeds."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(index: int, record: Mapping[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            started = time.perf_counter()
            raw_output = await sample_completion(record["messages"][:2], seed + index)
            latency = time.perf_counter() - started
        normalized = normalize_action_output(record, raw_output)
        prediction = normalized["prediction"]
        prediction_action = normalized["prediction_action"]
        teacher = record["metadata"]["action_name"]
        legal_names = legal_action_names(record)
        legal = normalized["parseable_action"] and prediction_action in legal_names
        exact = normalized["parseable_action"] and prediction_action == teacher
        return {
            "index": index,
            "seed": record["metadata"]["seed"],
            "game_id": record["metadata"]["game_id"],
            "player": record["metadata"]["player"],
            "step": record["metadata"]["step"],
            "teacher_action": teacher,
            "legal_actions": legal_names,
            "raw_output": raw_output,
            "prediction": prediction,
            "prediction_action": prediction_action,
            "exact_match": exact,
            "canonical_format": normalized["canonical_format"],
            "parseable_action": normalized["parseable_action"],
            "legal_action": legal,
            "invalid_output": not normalized["parseable_action"],
            "latency_seconds": round(latency, 6),
        }

    return list(await asyncio.gather(*(evaluate_one(index, record) for index, record in enumerate(records))))


def summarize_results(results: Sequence[Mapping[str, Any]], model_ref: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
    count = len(results)
    if not count:
        raise ValueError("cannot summarize empty results")
    exact = sum(bool(row["exact_match"]) for row in results)
    canonical = sum(bool(row["canonical_format"]) for row in results)
    parseable = sum(bool(row["parseable_action"]) for row in results)
    legal = sum(bool(row["legal_action"]) for row in results)
    invalid = sum(bool(row["invalid_output"]) for row in results)
    latencies = [float(row["latency_seconds"]) for row in results]
    teacher_counts = Counter(str(row["teacher_action"]) for row in results)
    correct_counts = Counter(str(row["teacher_action"]) for row in results if row["exact_match"])
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        prediction = str(row["prediction_action"]) if row["parseable_action"] else "<INVALID>"
        confusion[str(row["teacher_action"])][prediction] += 1
    per_action = {
        action: {
            "count": teacher_counts[action],
            "correct": correct_counts[action],
            "accuracy": correct_counts[action] / teacher_counts[action],
        }
        for action in sorted(teacher_counts)
    }
    return {
        "model_ref": model_ref,
        "evaluated_count": count,
        "exact_teacher_action_matches": exact,
        "exact_teacher_action_match_rate": exact / count,
        "canonical_format_count": canonical,
        "canonical_format_rate": canonical / count,
        "parseable_action_count": parseable,
        "parseable_action_rate": parseable / count,
        "legal_action_count": legal,
        "legal_action_rate": legal / count,
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / count,
        "per_action": per_action,
        "confusion": {teacher: dict(sorted(predictions.items())) for teacher, predictions in sorted(confusion.items())},
        "latency_seconds": {
            "total": sum(latencies),
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[min(count - 1, int(0.95 * count))],
            "max": max(latencies),
        },
        "settings": dict(settings),
    }


def write_results(output_dir: Path, results: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_record_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if per_record_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation artifacts in {output_dir}")
    with per_record_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def run_remote(args: argparse.Namespace, records: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
    import tinker
    from tinker_cookbook import renderers

    service_client = tinker.ServiceClient()
    if args.model_path:
        sampling_client = await service_client.create_sampling_client_async(model_path=args.model_path)
        model_ref = args.model_path
    else:
        sampling_client = await service_client.create_sampling_client_async(base_model=args.base_model)
        model_ref = args.base_model
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.base_model)
    stop_sequences = renderer.get_stop_sequences()

    async def sample_completion(messages: Sequence[Mapping[str, str]], seed: int) -> str:
        prompt = renderer.build_generation_prompt(list(messages))
        response = await sampling_client.sample_async(
            prompt=prompt,
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_output_tokens,
                temperature=0.0,
                seed=seed,
                stop=stop_sequences,
            ),
            num_samples=1,
        )
        return parsed_response_text(renderer, response.sequences[0].tokens)

    results = await evaluate_records(records, sample_completion, seed=args.seed, concurrency=args.concurrency)
    return results, model_ref


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Used to verify seed-disjoint evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--base-model")
    model.add_argument("--model-path")
    parser.add_argument("--renderer", default="qwen3_disable_thinking")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_datasets(args.train_jsonl, args.eval_jsonl)
    records = load_eval_records(args.eval_jsonl, args.limit)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output directory: {args.output_dir}")
    _load_key_for_explicit_run(args.env_file.resolve())
    import os

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Refusing evaluation: set TINKER_API_KEY to a current credential first.")
    started = time.perf_counter()
    results, model_ref = asyncio.run(run_remote(args, records))
    settings = {
        "renderer": args.renderer,
        "temperature": 0.0,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "eval_jsonl": str(args.eval_jsonl.resolve()),
        "limit": args.limit,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    summary = summarize_results(results, model_ref, settings)
    write_results(args.output_dir, results, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
