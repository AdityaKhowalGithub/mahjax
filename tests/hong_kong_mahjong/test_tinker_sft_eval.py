import asyncio
import json

from examples.eval_hk_tinker_sft import (
    evaluate_records,
    normalize_action_output,
    parsed_response_text,
    summarize_results,
    write_results,
)


def _record(teacher="DISCARD_1M"):
    return {
        "messages": [
            {"role": "system", "content": "Choose one action."},
            {"role": "user", "content": "LEGAL_ACTIONS=[0:DISCARD_1M,1:DISCARD_2M]"},
            {"role": "assistant", "content": teacher},
        ],
        "metadata": {"seed": 10, "game_id": "game-10", "player": 0, "step": 0, "action_name": teacher},
    }


def test_metrics_cover_exact_legal_canonical_and_invalid_outputs(tmp_path):
    records = [_record(), _record(), _record()]
    outputs = iter(["DISCARD_1M", "DISCARD_2M\n", "not an action"])

    async def sample(messages, seed):
        assert [message["role"] for message in messages] == ["system", "user"]
        assert seed in (7, 8, 9)
        return next(outputs)

    results = asyncio.run(evaluate_records(records, sample, seed=7, concurrency=2))
    summary = summarize_results(results, "mock-model", {"temperature": 0.0})

    assert summary["evaluated_count"] == 3
    assert summary["exact_teacher_action_matches"] == 1
    assert summary["canonical_format_count"] == 2
    assert summary["legal_action_count"] == 2
    assert summary["invalid_output_count"] == 1
    assert summary["confusion"]["DISCARD_1M"] == {
        "<INVALID>": 1,
        "DISCARD_1M": 1,
        "DISCARD_2M": 1,
    }

    write_results(tmp_path, results, summary)
    saved = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(saved) == 3
    assert json.loads((tmp_path / "summary.json").read_text())["model_ref"] == "mock-model"


def test_teacher_action_can_be_non_discard():
    record = _record("PON")
    record["messages"][1]["content"] = "LEGAL_ACTIONS=[72:PON,77:PASS]"

    async def sample(messages, seed):
        return "PON"

    result = asyncio.run(evaluate_records([record], sample, seed=0, concurrency=1))[0]
    assert result["exact_match"]
    assert result["canonical_format"]
    assert result["legal_action"]


def test_sampled_tokens_are_parsed_by_renderer_before_scoring():
    class FakeRenderer:
        def parse_response(self, tokens):
            assert tokens == [1, 2, 3]
            return {"role": "assistant", "content": "DISCARD_1M"}, "clean-stop"

    assert parsed_response_text(FakeRenderer(), [1, 2, 3]) == "DISCARD_1M"


def test_numeric_and_id_name_outputs_normalize_without_hiding_format_failures():
    record = _record()
    numeric = normalize_action_output(record, "0")
    paired = normalize_action_output(record, "1:DISCARD_2M")
    inconsistent = normalize_action_output(record, "0:DISCARD_2M")

    assert numeric == {
        "prediction": "0",
        "prediction_action": "DISCARD_1M",
        "canonical_format": False,
        "parseable_action": True,
    }
    assert paired["prediction_action"] == "DISCARD_2M"
    assert paired["parseable_action"]
    assert not paired["canonical_format"]
    assert not inconsistent["parseable_action"]
    assert inconsistent["prediction_action"] is None
