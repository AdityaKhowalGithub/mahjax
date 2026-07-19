import builtins
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

import examples.run_hk_tinker_sft as adapter
from examples.run_hk_tinker_sft import (
    DEFAULT_MODEL,
    DEFAULT_RENDERER,
    DatasetValidationError,
    TrainingSettings,
    main,
    validate_datasets,
)


def _record(split, seed):
    return {
        "messages": [
            {"role": "system", "content": "Choose one legal action."},
            {"role": "user", "content": "LEGAL_ACTIONS=[0:DISCARD_1M]"},
            {"role": "assistant", "content": "DISCARD_1M"},
        ],
        "metadata": {
            "schema": "mahjax.hk_sft_smoke.v1",
            "ruleset": "hk_old_style_v1",
            "data_quality": "smoke_baseline",
            "split": split,
            "seed": seed,
            "game_id": f"hk-smoke-{seed}",
            "seat": 0,
            "player": 0,
            "step": 0,
            "action_id": 0,
            "action_name": "DISCARD_1M",
            "legal_action_ids": [0],
        },
    }


def _write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_validate_only_does_not_import_tinker(tmp_path, monkeypatch, capsys):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write(train_path, [_record("train", 10)])
    _write(eval_path, [_record("eval", 20)])
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tinker" or name.startswith("tinker_cookbook"):
            raise AssertionError("validate-only imported Tinker")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(adapter, "_load_key_for_explicit_run", lambda path: pytest.fail(f"read env file: {path}"))
    assert main(["--train-jsonl", str(train_path), "--eval-jsonl", str(eval_path), "--validate-only"]) == 0
    output = capsys.readouterr().out
    assert "Local validation passed" in output
    assert f'"model": "{DEFAULT_MODEL}"' in output
    assert f'"renderer": "{DEFAULT_RENDERER}"' in output
    assert '"lora_rank": 16' in output
    assert '"batch_size": 8' in output
    assert '"max_length": 1024' in output
    assert '"num_epochs": 1' in output
    assert '"max_steps": 3' in output


def test_validation_rejects_game_seed_leakage(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write(train_path, [_record("train", 10)])
    _write(eval_path, [_record("eval", 10)])

    with pytest.raises(DatasetValidationError, match="seed leakage"):
        validate_datasets(train_path, eval_path)


def test_validation_rejects_illegal_label_and_non_smoke_schema(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    bad_action = _record("train", 10)
    bad_action["metadata"]["action_id"] = 1
    _write(train_path, [bad_action])
    _write(eval_path, [_record("eval", 20)])
    with pytest.raises(DatasetValidationError, match="selected action is not legal"):
        validate_datasets(train_path, eval_path)

    bad_quality = _record("train", 10)
    bad_quality["metadata"]["data_quality"] = "expert"
    _write(train_path, [bad_quality])
    with pytest.raises(DatasetValidationError, match="smoke_baseline"):
        validate_datasets(train_path, eval_path)


def test_training_settings_validate_optional_max_steps(tmp_path):
    settings = TrainingSettings(
        train_jsonl=tmp_path / "train.jsonl",
        eval_jsonl=tmp_path / "eval.jsonl",
        log_path=tmp_path / "logs",
        max_steps=2,
    )
    assert settings.model == "Qwen/Qwen3-8B"
    assert settings.renderer == "qwen3_disable_thinking"
    assert settings.lora_rank == 16
    assert settings.batch_size == 8
    assert settings.max_length == 1024
    assert settings.num_epochs == 1
    assert settings.max_steps == 2

    with pytest.raises(ValueError, match="max_steps"):
        TrainingSettings(
            train_jsonl=tmp_path / "train.jsonl",
            eval_jsonl=tmp_path / "eval.jsonl",
            log_path=tmp_path / "logs",
            max_steps=0,
        )


def test_run_refuses_when_current_credential_is_missing(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write(train_path, [_record("train", 10)])
    _write(eval_path, [_record("eval", 20)])
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="Refusing remote training"):
        main(
            [
                "--train-jsonl",
                str(train_path),
                "--eval-jsonl",
                str(eval_path),
                "--env-file",
                str(tmp_path / "missing.env"),
                "--run",
            ]
        )


def test_run_loads_only_expected_dotenv_field_without_logging_it(tmp_path, monkeypatch, capsys):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    env_path = tmp_path / ".env"
    _write(train_path, [_record("train", 10)])
    _write(eval_path, [_record("eval", 20)])
    placeholder = "unit-test-placeholder"
    env_path.write_text(f"IGNORED=value\nTINKER_API_KEY='{placeholder}'\n", encoding="utf-8")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setattr(adapter, "build_tinker_config", lambda settings: settings)
    called = []

    async def fake_main(config):
        called.append(config)

    fake_supervised = ModuleType("tinker_cookbook.supervised")
    fake_supervised.train = SimpleNamespace(main=fake_main)
    fake_package = ModuleType("tinker_cookbook")
    fake_package.supervised = fake_supervised
    monkeypatch.setitem(sys.modules, "tinker_cookbook", fake_package)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised", fake_supervised)

    assert (
        main(
            [
                "--train-jsonl",
                str(train_path),
                "--eval-jsonl",
                str(eval_path),
                "--env-file",
                str(env_path),
                "--run",
            ]
        )
        == 0
    )
    assert os.environ["TINKER_API_KEY"] == placeholder
    assert called
    assert placeholder not in capsys.readouterr().out


def test_build_config_composes_explicit_splits_with_installed_stable_api(tmp_path, monkeypatch):
    import chz
    from tinker_cookbook.supervised.data import FromConversationFileBuilder

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    settings = TrainingSettings(train_jsonl=train_path, eval_jsonl=eval_path, log_path=tmp_path / "logs")

    def fake_build(builder):
        marker = "train-dataset" if builder.file_path == str(train_path) else "eval-dataset"
        return marker, None

    monkeypatch.setattr(FromConversationFileBuilder, "__call__", fake_build)
    config = adapter.build_tinker_config(settings)
    train_dataset, eval_dataset = config.dataset_builder()

    assert train_dataset == "train-dataset"
    assert eval_dataset == "eval-dataset"
    assert config.model_name == "Qwen/Qwen3-8B"
    assert config.renderer_name == "qwen3_disable_thinking"
    assert config.lora_rank == 16
    assert config.num_epochs == 1
    assert config.max_steps == 3
    assert config.eval_every == 0
    serialized = chz.asdict(config)
    assert serialized["model_name"] == "Qwen/Qwen3-8B"
    assert serialized["max_steps"] == 3
