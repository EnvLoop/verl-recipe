"""EnvLoop VeRL patches for fused log-prob kernels (``verl_tinker.verl_patches``), on CPU."""

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

from verl_tinker import verl_patches


class _Head(torch.nn.Module):
    def __init__(self, hidden=8, vocab=16):
        super().__init__()
        self.config = PretrainedConfig(tie_word_embeddings=False)
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList([_Block(hidden) for _ in range(2)])
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)


class _Block(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden)


def _lora_head(target_modules):
    torch.manual_seed(0)
    model = get_peft_model(_Head(), LoraConfig(r=4, lora_alpha=8, target_modules=target_modules))
    for name, param in model.named_parameters():
        if "lora_B" in name:  # PEFT initializes B to zero; make the delta visible.
            torch.nn.init.normal_(param)
    return model.base_model.model


def test_effective_weight_adds_the_active_lora_delta_and_trains_a_and_b():
    model = _lora_head(["lm_head"])
    head = model.lm_head
    hidden = torch.randn(3, 8)

    weight = verl_patches.effective_lm_head_weight(head)

    torch.testing.assert_close(hidden @ weight.T, head(hidden))
    weight.sum().backward()
    assert head.lora_A["default"].weight.grad is not None and head.lora_B["default"].weight.grad is not None
    assert head.base_layer.weight.grad is None  # frozen base weight
    head.enable_adapters(False)
    try:
        torch.testing.assert_close(verl_patches.effective_lm_head_weight(head), head.base_layer.weight)
    finally:
        head.enable_adapters(True)


def test_effective_weight_without_lora_is_the_plain_weight():
    model = _lora_head(["proj"])

    assert verl_patches.effective_lm_head_weight(model.lm_head) is model.lm_head.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VeRL's torch fused backend uses a Triton cross-entropy")
def test_fused_torch_backend_log_probs_include_the_lm_head_adapter():
    model = _lora_head(["lm_head", "proj"]).cuda()
    hidden = torch.randn(1, 5, 8, device="cuda")
    input_ids = torch.randint(0, 16, (1, 5), device="cuda")

    class _Out(tuple):
        hidden_states = None

    owner = SimpleNamespace(model=lambda ids, **_: _Out((hidden,)), lm_head=model.lm_head)

    out = verl_patches.forward_with_torch_backend(owner, input_ids=input_ids, temperature=1.0)

    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    expected = torch.log_softmax(model.lm_head(hidden).float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(out.log_probs.float(), expected, rtol=1e-4, atol=1e-4)


def test_fsdp2_wrap_targets_leave_lm_head_and_its_adapter_to_the_root_unit():
    from verl.utils import fsdp_utils

    model = _lora_head(["lm_head", "proj"])

    targets = fsdp_utils._select_fsdp2_wrap_targets(model, ["_Block"])

    assert fsdp_utils._select_fsdp2_wrap_targets is verl_patches._select_fsdp2_wrap_targets_root_lm_head
    names = {name for name, module in model.named_modules() if any(module is target for target in targets)}
    assert names == {"embed_tokens", "layers.0", "layers.1"}


def test_patches_are_installed_where_verl_looks_them_up():
    from verl.models.transformers import qwen3_5

    verl_patches.apply()  # idempotent

    assert qwen3_5.forward_with_torch_backend is verl_patches.forward_with_torch_backend
    assert qwen3_5.forward_with_triton_backend is verl_patches.forward_with_triton_backend


@pytest.mark.parametrize("dropout", [0.1])
def test_lora_dropout_on_lm_head_is_refused(dropout):
    torch.manual_seed(0)
    model = get_peft_model(_Head(), LoraConfig(r=4, lora_alpha=8, target_modules=["lm_head"], lora_dropout=dropout))

    with pytest.raises(NotImplementedError, match="lora_dropout"):
        verl_patches.effective_lm_head_weight(model.base_model.model.lm_head)
