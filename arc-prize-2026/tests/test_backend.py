"""Tests for the model backend, in particular the KV-cache contract.

arc_solver itself cannot be imported here (it needs peft, which the Kaggle
image has and this machine does not), so these exercise arc_backend directly.
That is where the risky part lives: the DFS reuses one cache across sibling
branches, and the two cache flavours transformers has shipped behave
differently under exactly that access pattern.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import arc_backend


def test_backend_name_matches_availability():
    assert arc_backend.name() == ("unsloth" if arc_backend.HAS_UNSLOTH else "transformers")


def test_lora_spec_covers_embeddings_and_head():
    assert "embed_tokens" in arc_backend.LORA["target_modules"]
    assert "lm_head" in arc_backend.LORA["target_modules"]
    assert arc_backend.LORA["r"] == 256
    assert arc_backend.LORA["use_rslora"] is True


def _training_arguments():
    from transformers import TrainingArguments
    return TrainingArguments


def test_eval_strategy_key_is_one_transformers_accepts():
    import inspect
    try:
        TrainingArguments = _training_arguments()
    except Exception as e:
        pytest.skip(f"transformers.TrainingArguments unavailable here: {e}")
    key = arc_backend.eval_strategy_key()
    assert key in inspect.signature(TrainingArguments.__init__).parameters


def test_eval_strategy_key_falls_back_when_probing_fails(monkeypatch):
    monkeypatch.setattr(arc_backend, "eval_strategy_key",
                        arc_backend.eval_strategy_key)
    assert arc_backend.eval_strategy_key() in ("eval_strategy", "evaluation_strategy")


def test_plain_path_turns_on_gradient_checkpointing(monkeypatch):
    monkeypatch.setattr(arc_backend, "HAS_UNSLOTH", False)
    out = arc_backend.adjust_train_args({"bf16": True, "gradient_checkpointing": False})
    assert out["gradient_checkpointing"] is True
    assert out["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert out["bf16"] is True


def test_unsloth_path_keeps_the_reference_settings(monkeypatch):
    monkeypatch.setattr(arc_backend, "HAS_UNSLOTH", True)
    given = {"bf16": True, "gradient_checkpointing": False}
    assert arc_backend.adjust_train_args(given) == given


def test_adjust_train_args_does_not_mutate_the_caller(monkeypatch):
    monkeypatch.setattr(arc_backend, "HAS_UNSLOTH", False)
    given = {"gradient_checkpointing": False}
    arc_backend.adjust_train_args(given)
    assert given == {"gradient_checkpointing": False}


def test_the_cache_kind_is_reported_once(monkeypatch, capsys):
    monkeypatch.setattr(arc_backend, "_cache_reported", False)
    arc_backend.prepare_cache(FakeMutableCache(), 0)
    first = capsys.readouterr().out
    arc_backend.prepare_cache(FakeMutableCache(), 0)
    second = capsys.readouterr().out
    assert "kv cache" in first and "croppable" in first
    assert second == ""


def test_an_uncroppable_cache_says_so(monkeypatch, capsys):
    monkeypatch.setattr(arc_backend, "_cache_reported", False)
    arc_backend.prepare_cache(((1, 2),), 0)
    assert "no crop()" in capsys.readouterr().out


def test_lora_rank_comes_from_config():
    import arc_config
    assert arc_backend.LORA["r"] == arc_config.LORA_RANK


def fake_peft(monkeypatch, probe):
    """Install a throwaway peft package whose torchao probe behaves as given."""
    import types
    import_utils = types.ModuleType("peft.import_utils")
    import_utils.is_torchao_available = probe

    lora_torchao = types.ModuleType("peft.tuners.lora.torchao")
    lora_torchao.is_torchao_available = probe

    peft = types.ModuleType("peft")
    peft.import_utils = import_utils
    tuners = types.ModuleType("peft.tuners")
    lora = types.ModuleType("peft.tuners.lora")
    lora.torchao = lora_torchao

    for name, mod in [
        ("peft", peft), ("peft.import_utils", import_utils),
        ("peft.tuners", tuners), ("peft.tuners.lora", lora),
        ("peft.tuners.lora.torchao", lora_torchao),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return import_utils, lora_torchao


def test_a_raising_torchao_probe_is_forced_to_false(monkeypatch):
    def raises():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported"
        )

    import_utils, lora_torchao = fake_peft(monkeypatch, raises)
    patched = arc_backend.neutralise_optional_backend_probes()

    assert import_utils.is_torchao_available() is False
    assert lora_torchao.is_torchao_available() is False
    assert "peft.tuners.lora.torchao" in patched


def test_a_healthy_probe_is_left_alone(monkeypatch):
    import_utils, _ = fake_peft(monkeypatch, lambda: True)
    assert arc_backend.neutralise_optional_backend_probes() == []
    assert import_utils.is_torchao_available() is True


def test_a_probe_that_answers_false_is_left_alone(monkeypatch):
    import_utils, _ = fake_peft(monkeypatch, lambda: False)
    assert arc_backend.neutralise_optional_backend_probes() == []


def test_no_peft_at_all_is_not_an_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("peft"):
            raise ImportError("no peft here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert arc_backend.neutralise_optional_backend_probes() == []


class FakeMutableCache:
    """Stands in for transformers' DynamicCache: grows in place per forward."""

    def __init__(self):
        self.length = 0

    def append(self):
        self.length += 1

    def crop(self, n):
        self.length = min(self.length, n)


@pytest.fixture(autouse=True)
def _reset_cache_report(monkeypatch):
    monkeypatch.setattr(arc_backend, "_cache_reported", True)


def test_prepare_cache_crops_a_mutable_cache_back_to_the_prefix():
    cache = FakeMutableCache()
    for _ in range(5):
        cache.append()
    arc_backend.prepare_cache(cache, 3)
    assert cache.length == 3


def test_prepare_cache_leaves_a_shorter_cache_alone():
    cache = FakeMutableCache()
    cache.append()
    arc_backend.prepare_cache(cache, 4)
    assert cache.length == 1


def test_prepare_cache_passes_tuples_through_untouched():
    legacy = ((1, 2), (3, 4))
    assert arc_backend.prepare_cache(legacy, 1) is legacy


def test_prepare_cache_survives_a_crop_that_refuses():
    class Stubborn:
        def crop(self, n):
            raise IndexError("cannot crop a static cache")

    cache = Stubborn()
    assert arc_backend.prepare_cache(cache, 2) is cache


def test_sibling_branches_start_from_the_same_prefix():
    """The behaviour the DFS depends on.

    Without cropping, exploring one continuation of a prefix would leave the
    cache holding that continuation, and the next sibling would be decoded as
    if it followed the first one instead of the shared prefix.
    """
    cache = FakeMutableCache()
    prefix_len = 4
    for _ in range(prefix_len):
        cache.append()

    seen = []
    for _ in range(3):
        arc_backend.prepare_cache(cache, prefix_len)
        seen.append(cache.length)
        cache.append()
        cache.append()

    assert seen == [prefix_len] * 3


class FakeConfig:
    def __init__(self):
        self.use_cache = False


class FakeModel:
    """A model in the state the Trainer leaves behind."""

    def __init__(self):
        self.config = FakeConfig()
        self.checkpointing = True
        self.mode = "train"

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def eval(self):
        self.mode = "eval"
        return self


def test_inference_mode_restores_the_kv_cache(monkeypatch):
    """The Trainer disables use_cache alongside gradient checkpointing, and the
    DFS cannot decode without a cache."""
    monkeypatch.setattr(arc_backend, "HAS_UNSLOTH", False)
    model = FakeModel()
    out = arc_backend.for_inference(model)
    assert out is model
    assert model.checkpointing is False
    assert model.config.use_cache is True
    assert model.mode == "eval"


def test_inference_mode_tolerates_a_model_without_those_hooks(monkeypatch):
    monkeypatch.setattr(arc_backend, "HAS_UNSLOTH", False)

    class Bare:
        def eval(self):
            self.mode = "eval"
            return self

    model = arc_backend.for_inference(Bare())
    assert model.mode == "eval"
