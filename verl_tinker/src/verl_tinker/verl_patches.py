"""EnvLoop: VeRL fixes for fused log-prob kernels, loaded in every worker via ``model.external_lib``.

VeRL's fused log-prob path (``use_fused_kernels``, needed for 64K+ contexts so the
``[tokens, vocab]`` logits are never materialized) reads ``self.lm_head.weight``
directly instead of calling ``lm_head``. Two consequences at VeRL ``cfacd76``
(still present on upstream main 2026-10-16):

1. FSDP2 wraps an untied ``lm_head`` as its own unit; a weight read outside the
   unit's forward is never unsharded and the post-backward hook fails with
   ``'FSDPParam' object has no attribute '_unsharded_param'`` (Qwen3.8-27B).
   Fix: keep ``lm_head`` in the root unit, which FSDP2 unshards for the whole
   forward/backward of the root (2.5 GB for Qwen3.8-27B).
2. When PEFT wraps ``lm_head`` (LoRA on the unembedding, Tinker's
   ``train_unembed``), ``.weight`` is the frozen base weight and the adapter is
   silently ignored. Fix: the qwen3_5 fused forwards use the effective weight
   ``W + scaling * B @ A`` (differentiable in A and B).

The two forward functions are VeRL's ``verl/models/transformers/qwen3_5.py``
(Apache-2.0, Bytedance Ltd. and/or its affiliates) with only the
``vocab_weights`` line changed.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.distributed.tensor import DTensor

_APPLIED = False


def effective_lm_head_weight(head: torch.nn.Module) -> torch.Tensor:
    """``head.weight`` plus any active, unmerged LoRA delta (PEFT ``lora.Linear``)."""
    weight = head.weight
    if isinstance(weight, DTensor):
        weight = weight.full_tensor()
    lora_a = getattr(head, "lora_A", None)
    if not lora_a or getattr(head, "disable_adapters", False) or getattr(head, "merged", False):
        return weight
    for name in getattr(head, "active_adapters", ()):
        if name not in lora_a:
            continue
        if getattr(head, "use_dora", {}).get(name, False):
            raise NotImplementedError("fused log-prob kernels do not support DoRA on lm_head")
        dropout = head.lora_dropout[name]
        if not isinstance(dropout, torch.nn.Identity) and getattr(dropout, "p", 0.0) != 0.0:
            raise NotImplementedError("fused log-prob kernels need lora_dropout=0 on lm_head")
        a, b = lora_a[name].weight, head.lora_B[name].weight
        a = a.full_tensor() if isinstance(a, DTensor) else a
        b = b.full_tensor() if isinstance(b, DTensor) else b
        weight = weight + (b @ a).to(weight.dtype) * head.scaling[name]
    return weight


def _select_fsdp2_wrap_targets_root_lm_head(model, fsdp_transformer_layer_cls_to_wrap):
    """VeRL's FSDP2 wrap targets minus ``lm_head`` (and its LoRA children), left to the root unit."""
    lm_heads = [module for name, module in model.named_modules() if name.rsplit(".", 1)[-1] == "lm_head"]
    inside = {id(child) for head in lm_heads for child in head.modules()}
    return [module for module in _ORIGINAL_SELECT(model, fsdp_transformer_layer_cls_to_wrap)
            if id(module) not in inside]


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO
    from verl.utils.experimental.torch_functional import FusedLinearForPPO
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, ulysses_pad_and_slice_inputs

    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    vocab_weights = effective_lm_head_weight(self.lm_head)  # EnvLoop: was self.lm_head.weight

    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if shift_labels is None and ulysses_sequence_parallel_size > 1:
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
        )
    hidden_states = hidden_states.to(vocab_weights.dtype)
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=vocab_weights,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(log_probs=log_probs, entropy=entropy, hidden_states=outputs.hidden_states)


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
    from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, ulysses_pad_and_slice_inputs

    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")
    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if shift_labels is None and ulysses_sequence_parallel_size > 1:
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
        )

    vocab_weights = effective_lm_head_weight(self.lm_head)  # EnvLoop: was self.lm_head.weight
    hidden_states = hidden_states.to(vocab_weights.dtype)
    log_probs, entropy = linear_cross_entropy(hidden_states, vocab_weights, rolled_labels, temperature, "none")
    return Qwen3_5CausalLMOutputForPPO(log_probs=log_probs, entropy=entropy, hidden_states=outputs.hidden_states)


def apply() -> None:
    """Idempotent; runs on import (``model.external_lib: verl_tinker.verl_patches``)."""
    global _APPLIED, _ORIGINAL_SELECT
    if _APPLIED:
        return
    from verl.models.transformers import qwen3_5
    from verl.utils import fsdp_utils

    _ORIGINAL_SELECT = fsdp_utils._select_fsdp2_wrap_targets
    fsdp_utils._select_fsdp2_wrap_targets = _select_fsdp2_wrap_targets_root_lm_head
    # monkey_patch.patch_forward_with_backends imports these by name at patch time.
    qwen3_5.forward_with_torch_backend = forward_with_torch_backend
    qwen3_5.forward_with_triton_backend = forward_with_triton_backend
    _APPLIED = True


_ORIGINAL_SELECT = None
apply()
