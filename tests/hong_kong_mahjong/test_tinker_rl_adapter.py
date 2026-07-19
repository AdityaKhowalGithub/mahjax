import asyncio
import builtins
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.run_hk_tinker_rl import (
    HKGroupBuilder,
    OracleValidationError,
    RLSettings,
    build_tinker_config,
    hk_action_name,
    preflight_log_path,
    score_action,
    train_with_provenance,
    validate_datasets,
)


def _record(split: str, seed: int) -> dict:
    return {
        "schema": "mahjax.hk_rl_oracle.v1",
        "messages": [
            {"role": "system", "content": "Choose one canonical legal action only."},
            {"role": "user", "content": "State...\nLEGAL_ACTIONS: 0:DISCARD_1M, 13:DISCARD_5P, 77:PASS"},
        ],
        "metadata": {
            "ruleset": "hk_old_style_v1",
            "data_quality": "oracle_privileged_reward",
            "split": split,
            "seed": seed,
            "game_id": f"game-{seed}",
            "player": 0,
            "step": 3,
            "trajectory_policy": "uniform_legal",
        },
        "legal_actions": [
            {"id": 0, "name": "DISCARD_1M"},
            {"id": 13, "name": "DISCARD_5P"},
            {"id": 77, "name": "PASS"},
        ],
        "action_rewards": {"DISCARD_1M": -1.0, "DISCARD_5P": 0.25, "PASS": 0.25},
        "best_action_names": ["DISCARD_5P", "PASS"],
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_validate_oracle_data_and_seed_isolation(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    summary = validate_datasets(train, evaluation)
    assert summary["train"]["records"] == 1
    assert summary["eval"]["game_seeds"] == [20]

    _write(evaluation, [_record("eval", 10)])
    with pytest.raises(OracleValidationError, match="seed leakage"):
        validate_datasets(train, evaluation)


def test_validation_rejects_reward_or_prompt_mismatch(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    bad = _record("train", 10)
    bad["action_rewards"].pop("PASS")
    _write(train, [bad])
    _write(evaluation, [_record("eval", 20)])
    with pytest.raises(OracleValidationError, match="keys must exactly equal"):
        validate_datasets(train, evaluation)


def test_continuous_reward_ties_and_malformed_penalty() -> None:
    record = _record("train", 10)
    assert score_action(record, "DISCARD_5P")["oracle_reward"] == 0.25
    assert score_action(record, "77")["best_action"] is True
    malformed = score_action(record, "I choose PASS because it is safe")
    assert malformed["legal_action"] is False
    assert malformed["oracle_reward"] == -1.0


def test_guarded_defaults_and_validate_only_do_not_import_tinker(tmp_path: Path, monkeypatch, capsys) -> None:
    from examples import run_hk_tinker_rl

    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    settings = RLSettings(train, evaluation, tmp_path / "logs")
    assert settings.group_size == 4
    assert settings.max_output_tokens == 12
    assert settings.learning_rate == 1e-5
    assert settings.kl_penalty_coef == 0.05
    assert settings.max_steps == 3
    assert settings.temperature == 1.0

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tinker" or name.startswith("tinker_cookbook"):
            raise AssertionError(f"validate-only imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert run_hk_tinker_rl.main([
        "--train-jsonl", str(train), "--eval-jsonl", str(evaluation), "--validate-only"
    ]) == 0
    assert "not imported or contacted" in capsys.readouterr().out


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "cispo"])
def test_configurable_supported_losses(tmp_path: Path, loss_fn: str) -> None:
    settings = RLSettings(tmp_path / "t", tmp_path / "e", tmp_path / "l", loss_fn=loss_fn)
    assert settings.loss_fn == loss_fn


@pytest.mark.parametrize(
    ("action_id", "name"),
    [(0, "DISCARD_1M"), (33, "DISCARD_RED"), (34, "SELF_KONG_1M"), (68, "TSUMOGIRI"), (77, "PASS")],
)
def test_exact_hk_action_mapping(action_id: int, name: str) -> None:
    assert hk_action_name(action_id) == name


@pytest.mark.parametrize(("action_id", "name"), [(69, "RIICHI"), (78, "DUMMY"), (77, "RON")])
def test_validation_rejects_unsupported_or_impossible_action_pairs(
    tmp_path: Path, action_id: int, name: str
) -> None:
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    bad = _record("train", 10)
    bad["messages"][1]["content"] = f"LEGAL_ACTIONS: {action_id}:{name}"
    bad["legal_actions"] = [{"id": action_id, "name": name}]
    bad["action_rewards"] = {name: 1.0}
    bad["best_action_names"] = [name]
    _write(train, [bad])
    _write(evaluation, [_record("eval", 20)])
    with pytest.raises(OracleValidationError, match="unsupported|must map"):
        validate_datasets(train, evaluation)


def test_duplicate_identity_and_prompt_are_rejected(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    duplicate_identity = _record("train", 10)
    _write(train, [duplicate_identity, duplicate_identity])
    _write(evaluation, [_record("eval", 20)])
    with pytest.raises(OracleValidationError, match="duplicate decision identity"):
        validate_datasets(train, evaluation)

    second = _record("train", 11)
    second["messages"] = duplicate_identity["messages"]
    _write(train, [duplicate_identity, second])
    with pytest.raises(OracleValidationError, match="duplicate prompt"):
        validate_datasets(train, evaluation)


def test_group_builder_uses_malformed_penalty_for_context_overflow() -> None:
    class Renderer:
        @staticmethod
        def get_stop_sequences() -> list[str]:
            return []

    builder = HKGroupBuilder(_record("train", 10), Renderer(), 1, -0.75, 12)
    env = asyncio.run(builder.make_envs())[0]
    assert env.failed_parse_reward == -0.75
    assert env.context_overflow_reward == -0.75


def test_group_builder_completes_real_cookbook_group_rollout() -> None:
    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.rollouts import do_group_rollout

    class Termination:
        is_clean = True

    class Renderer:
        @staticmethod
        def get_stop_sequences() -> list[str]:
            return []

        @staticmethod
        def build_generation_prompt(messages, **kwargs):
            return tinker.ModelInput.from_ints([1])

        @staticmethod
        def parse_response(tokens):
            return {"role": "assistant", "content": "PASS"}, Termination()

    async def policy(model_input, stop, *, max_tokens=None):
        return TokensWithLogprobs(tokens=[77], maybe_logprobs=[-0.1], stop_reason="stop")

    builder = HKGroupBuilder(_record("train", 10), Renderer(), 2, -1.0, 12)
    group = asyncio.run(do_group_rollout(builder, policy))
    assert group.get_total_rewards() == [0.25, 0.25]
    assert group.final_rewards_G == [0.0, 0.0]


def test_config_dataset_builder_is_module_level_and_pickleable(tmp_path: Path) -> None:
    train, evaluation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    config = build_tinker_config(RLSettings(train, evaluation, tmp_path / "logs"))
    restored = pickle.loads(pickle.dumps(config.dataset_builder))
    assert type(restored).__module__ == "examples.run_hk_tinker_rl"
    assert restored.group_size == 4
    assert config.temperature == 1.0


@pytest.mark.parametrize("temperature", [0.0, -0.1, 2.01, float("inf"), float("nan")])
def test_rollout_temperature_safety_bound(tmp_path: Path, temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        RLSettings(tmp_path / "train", tmp_path / "eval", tmp_path / "logs", temperature=temperature)


def test_successful_training_atomically_writes_complete_provenance(tmp_path: Path) -> None:
    train, evaluation, logs = tmp_path / "train.jsonl", tmp_path / "eval.jsonl", tmp_path / "logs"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    settings = RLSettings(train.resolve(), evaluation.resolve(), logs.resolve())
    calls: list[tuple[str, str]] = []

    async def fake_train_main(config) -> None:
        assert config == "local-config"
        logs.mkdir()

    def fake_get_last_checkpoint(log_path: str, required_key: str):
        calls.append((log_path, required_key))
        if required_key == "sampler_path":
            return SimpleNamespace(name="final", sampler_path="tinker://run/sampler_weights/final")
        return SimpleNamespace(name="final", state_path="tinker://run/training_state/final")

    manifest = asyncio.run(train_with_provenance(
        settings,
        train_main=fake_train_main,
        get_last_checkpoint=fake_get_last_checkpoint,
        config_builder=lambda value: "local-config",
        git_info={"commit": "abc123", "dirty": False},
    ))
    stored = json.loads((logs / "provenance.json").read_text())
    assert stored == manifest
    assert stored["schema"] == "mahjax.hk_tinker_provenance.v1"
    assert stored["checkpoint_path"] == "tinker://run/sampler_weights/final"
    assert stored["state_checkpoint_path"] == "tinker://run/training_state/final"
    assert stored["datasets"]["train"]["count"] == 1
    assert stored["datasets"]["eval"]["seeds"] == [20]
    assert stored["datasets"]["train"]["file_sha256"].startswith("sha256:")
    assert stored["datasets"]["train"]["ordered_records_fingerprint"].startswith("sha256:")
    assert stored["config"]["model"] == "Qwen/Qwen3-8B"
    assert stored["started_at"] <= stored["ended_at"]
    assert stored["runtime_seconds"] >= 0
    assert calls == [(str(logs.resolve()), "sampler_path"), (str(logs.resolve()), "state_path")]
    assert not list(logs.glob(".provenance.json.*.tmp"))


def test_training_fails_without_final_sampler_and_publishes_no_manifest(tmp_path: Path) -> None:
    train, evaluation, logs = tmp_path / "train.jsonl", tmp_path / "eval.jsonl", tmp_path / "logs"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    settings = RLSettings(train.resolve(), evaluation.resolve(), logs.resolve())

    async def fake_train_main(config) -> None:
        logs.mkdir()

    with pytest.raises(RuntimeError, match="no final sampler checkpoint"):
        asyncio.run(train_with_provenance(
            settings,
            train_main=fake_train_main,
            get_last_checkpoint=lambda log_path, required_key: None,
            config_builder=lambda value: "local-config",
            git_info={"commit": None, "dirty": None},
        ))
    assert not (logs / "provenance.json").exists()


def test_fresh_and_resume_log_directory_guards(tmp_path: Path) -> None:
    train, evaluation, logs = tmp_path / "train.jsonl", tmp_path / "eval.jsonl", tmp_path / "logs"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    settings = RLSettings(train.resolve(), evaluation.resolve(), logs.resolve())
    logs.mkdir()
    (logs / "unrelated.txt").write_text("do not resume me")
    with pytest.raises(RuntimeError, match="non-empty log directory"):
        preflight_log_path(settings, resume=False)
    with pytest.raises(RuntimeError, match="provenance file does not exist"):
        preflight_log_path(settings, resume=True)


def test_resume_requires_exact_existing_provenance_binding(tmp_path: Path) -> None:
    train, evaluation, logs = tmp_path / "train.jsonl", tmp_path / "eval.jsonl", tmp_path / "logs"
    _write(train, [_record("train", 10)])
    _write(evaluation, [_record("eval", 20)])
    settings = RLSettings(train.resolve(), evaluation.resolve(), logs.resolve())

    async def fake_train_main(config) -> None:
        logs.mkdir()

    def checkpoint(log_path: str, required_key: str):
        return SimpleNamespace(name="final", **{required_key: f"tinker://run/{required_key}/final"})

    asyncio.run(train_with_provenance(
        settings,
        train_main=fake_train_main,
        get_last_checkpoint=checkpoint,
        config_builder=lambda value: "local-config",
        git_info={"commit": "abc123", "dirty": False},
    ))
    assert preflight_log_path(settings, resume=True)["datasets"]["train"]["count"] == 1

    train.write_text(json.dumps(_record("train", 10), sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        preflight_log_path(settings, resume=True)
