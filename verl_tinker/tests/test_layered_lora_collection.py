"""Per-unit LoRA collection for rollout weight sync (FSDP1 + VeRL's LoRA wrap policy).

A stand-in FSDP unit reproduces how VeRL wraps a PEFT model for LoRA: every trainable
leaf (``lora_A.default`` / ``lora_B.default``) and every decoder layer is its own unit.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import LlamaConfig, LlamaForCausalLM

from verl.utils import device as verl_device
from verl.utils import fsdp_utils
from verl_tinker.backends import colocated


class Unit(torch.nn.Module):
    """FSDP1 naming without sharding: the wrapped module sits under ``_fsdp_wrapped_module``."""

    def __init__(self, module):
        super().__init__()
        self._fsdp_wrapped_module = module


def lora_model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    return get_peft_model(LlamaForCausalLM(config),
                          LoraConfig(r=2, target_modules="all-linear", init_lora_weights=False))


def wrap_like_verl(peft_model, *, skip_layer=None):
    """VeRL's LoRA policy: each trainable leaf, each decoder layer, then the root."""
    layers = peft_model.base_model.model.model.layers
    skipped = set() if skip_layer is None else set(layers[skip_layer].modules())
    leaves = [(parent, name, child) for parent in peft_model.modules() if parent not in skipped
              for name, child in parent.named_children()
              if not list(child.children()) and getattr(child, "weight", None) is not None
              and child.weight.requires_grad]
    for parent, name, child in leaves:
        setattr(parent, name, Unit(child))
    for index in range(len(layers)):
        if index != skip_layer:
            layers[index] = Unit(layers[index])
    return Unit(peft_model)


@pytest.fixture
def summoned(monkeypatch):
    units = []

    @contextmanager
    def summon(module, writeback=True, **_):
        assert writeback is False
        units.append(module)
        yield

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda module: 1 if isinstance(module, Unit) else 0)
    monkeypatch.setattr(FSDP, "summon_full_params", staticmethod(summon))
    monkeypatch.setattr(verl_device, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    return units


def test_upstream_collector_misses_individually_wrapped_lora_leaves(summoned):
    root = wrap_like_verl(lora_model())
    assert colocated._layered_lora_params.upstream(root) == {}


def test_collects_every_adapter_tensor_one_lora_unit_at_a_time(summoned):
    model = lora_model()
    reference = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model).items()}
    root = wrap_like_verl(model)

    collected = fsdp_utils.collect_lora_params(root, layered_summon=True, base_sync_done=True)

    assert fsdp_utils.layered_summon_lora_params is colocated._layered_lora_params
    assert set(collected) == set(reference) and len(reference) == 2 * 7 * 2  # layers x linears x (A, B)
    assert all(torch.equal(collected[key], reference[key]) for key in reference)
    # Only the LoRA leaves are gathered, never a decoder layer or the root.
    assert len(summoned) == len(reference)
    assert all(not isinstance(unit._fsdp_wrapped_module, type(model.base_model.model.model.layers[0]))
               for unit in summoned)


def test_adapter_tensors_outside_any_lora_unit_fail_loudly(summoned):
    # Layer 0 left unwrapped: its adapter belongs to the root unit, which is never gathered alone.
    root = wrap_like_verl(lora_model(), skip_layer=0)
    with pytest.raises(RuntimeError, match="missed 14 of 28 adapter tensors"):
        colocated._layered_lora_params(root)
