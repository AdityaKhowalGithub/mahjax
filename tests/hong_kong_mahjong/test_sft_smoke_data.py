import json

from examples.generate_hk_sft_smoke import SCHEMA_VERSION, action_name, generate_dataset


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generation_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, seed=100, train_games=2, eval_games=1, max_steps=256)
    generate_dataset(second, seed=100, train_games=2, eval_games=1, max_steps=256)

    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert (first / "eval.jsonl").read_bytes() == (second / "eval.jsonl").read_bytes()


def test_schema_and_selected_actions_are_legal(tmp_path):
    output = tmp_path / "dataset"
    generate_dataset(output, seed=200, train_games=1, eval_games=1, max_steps=256)

    for record in _load_jsonl(output / "train.jsonl") + _load_jsonl(output / "eval.jsonl"):
        assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
        metadata = record["metadata"]
        assert metadata["schema"] == SCHEMA_VERSION
        assert metadata["data_quality"] == "smoke_baseline"
        assert metadata["action_id"] in metadata["legal_action_ids"]
        assert metadata["action_name"] == action_name(metadata["action_id"])
        assert record["messages"][2]["content"] == metadata["action_name"]
        assert record["messages"][2]["content"].strip() == record["messages"][2]["content"]
        user_prompt = record["messages"][1]["content"]
        assert f"{metadata['action_id']}:{metadata['action_name']}" in user_prompt
        assert "LAST_DRAW=" in user_prompt
        assert "ACTOR_FLOWERS=" in user_prompt
        assert "PRIVATE_FLOWERS=" not in user_prompt
        assert user_prompt.count("HAND=") == 1
        assert "DECK=" not in user_prompt
        assert "WALL_ORDER=" not in user_prompt
        assert "OPPONENT_HAND=" not in user_prompt
        for player in range(4):
            assert f"P{player}_HAND=" not in user_prompt


def test_whole_game_seeds_do_not_leak_across_splits(tmp_path):
    output = tmp_path / "dataset"
    manifest = generate_dataset(output, seed=300, train_games=3, eval_games=2, max_steps=256)
    train = _load_jsonl(output / "train.jsonl")
    eval_records = _load_jsonl(output / "eval.jsonl")

    train_seeds = {record["metadata"]["seed"] for record in train}
    eval_seeds = {record["metadata"]["seed"] for record in eval_records}
    assert train_seeds == set(manifest["splits"]["train"]["game_seeds"])
    assert eval_seeds == set(manifest["splits"]["eval"]["game_seeds"])
    assert train_seeds.isdisjoint(eval_seeds)
