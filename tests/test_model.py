import json
import sys
import types

import pytest

from app.evaluate import token_overlap
from app.train import build_lora_config, inject_lora


def test_lora_config_uses_gpt2_c_attn():
    config = build_lora_config()
    assert set(config.target_modules) == {"c_attn"}
    assert config.fan_in_fan_out is True
    assert config.r == 8


class FakeBaseModel:
    pass


def test_lora_injection_verifies_adapter_and_trainable_parameters(monkeypatch):
    fake_peft = types.ModuleType("peft")

    class FakeModel:
        def named_modules(self):
            yield "transformer.h.0.attn.c_attn.lora_A", object()

        def parameters(self):
            yield types.SimpleNamespace(numel=lambda: 16, requires_grad=True)

    fake_peft.get_peft_model = lambda base, config: FakeModel()
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    model = inject_lora(FakeBaseModel(), object())
    assert "lora_A" in next(model.named_modules())[0]


def test_token_overlap_is_normalized():
    assert token_overlap("passport application identity", "passport identity") == 1.0
    assert token_overlap("", "passport") == 0.0


def test_evaluation_missing_model_fails_clearly(tmp_path):
    from app.evaluate import evaluate

    with pytest.raises(FileNotFoundError) as exc:
        evaluate(model_dir=tmp_path / "missing", results_path=tmp_path / "results.json")
    assert "python -m app.train" in str(exc.value)


def test_merge_missing_adapter_fails_clearly(tmp_path):
    from app.merge import merge_adapter

    with pytest.raises(FileNotFoundError) as exc:
        merge_adapter(tmp_path / "adapter", tmp_path / "merged")
    assert "python -m app.train" in str(exc.value)
