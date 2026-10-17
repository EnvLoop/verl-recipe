# Copyright 2026 EnvLoop
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Separate q/k/v LoRAs on the fused Gated DeltaNet ``in_proj_qkv`` (Qwen3.5), the reference layout.

HF transformers (and vLLM) hold one ``linear_attn.in_proj_qkv`` Linear whose output is Q‖K‖V with unequal
widths (key, key, value). The reference platform trains three independent LoRAs on its slices
(``in_proj_q/k/v``, rank r each; tinker-cookbook ``weights/_merge_qwen3_5.SPLIT_QKV_SUFFIXES``); a plain PEFT
LoRA on the fused Linear shares one ``A`` and has ``2 * hidden * r`` fewer parameters per layer.

The base Linear stays as it is (same parameter, same ``in_proj_qkv.weight`` name, so HF loading, FSDP, the
rollout base-weight sync and VeRL's sequence-parallel patch are untouched): its class is swapped to the
marker :class:`SplitQKVLinear`, for which a PEFT custom LoRA layer (PEFT's documented
``LoraConfig._register_custom_module`` extension point) holds three ``A`` and three ``B`` Linears::

    in_proj_qkv(x) = W x + scaling * cat(B_q A_q x, B_k A_k x, B_v A_v x)

Parameter names: ``in_proj_qkv.lora_A.<adapter>.{q,k,v}.weight`` (FSDP / checkpoints) and, for vLLM and the
exported PEFT adapter, ``in_proj_{q,k,v}.lora_A.weight`` (:func:`split_peft_name`). A legacy adapter with one
fused ``in_proj_qkv`` LoRA converts exactly into this layout by copying ``A`` and slicing ``B``
(:func:`split_legacy_tensors`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

import torch
from torch import nn

SLICES = ("q", "k", "v")
FUSED = "in_proj_qkv"

_SPLIT_PARAM = re.compile(
    r"^(?P<prefix>.*\.)in_proj_qkv\.lora_(?P<ab>[AB])(?P<adapter>\.[^.]+)?\.(?P<slice>[qkv])\.weight$"
)
_FUSED_PARAM = re.compile(r"^(?P<prefix>.*\.)in_proj_qkv\.lora_(?P<ab>[AB])(?P<adapter>\.[^.]+)?\.weight$")


class SplitQKVLinear(nn.Linear):
    """Marker class for a fused q/k/v Linear; ``split_sizes`` are its output widths (q, k, v)."""

    split_sizes: tuple[int, int, int]


def mark_split_qkv(model: nn.Module) -> int:
    """Swap every Gated DeltaNet ``in_proj_qkv`` to :class:`SplitQKVLinear` in place; returns how many."""
    marked = 0
    for module in model.modules():
        linear = getattr(module, FUSED, None)
        key_dim, value_dim = getattr(module, "key_dim", None), getattr(module, "value_dim", None)
        if type(linear) is not nn.Linear or not isinstance(key_dim, int) or not isinstance(value_dim, int):
            continue
        if linear.out_features != 2 * key_dim + value_dim:
            raise ValueError(f"unexpected {FUSED} width {linear.out_features} for key {key_dim} / value {value_dim}")
        linear.__class__ = SplitQKVLinear
        linear.split_sizes = (key_dim, key_dim, value_dim)
        marked += 1
    return marked


class _SplitA(nn.Module):
    """Three ``A`` (r x in) applied to the same input; output is their concatenation (3r)."""

    def __init__(self, in_features: int, r: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(in_features, r, bias=False) for _ in SLICES)

    @property
    def weight(self) -> torch.Tensor:  # PEFT reads .weight for dtype and delta weights
        return torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)


class _SplitB(nn.Module):
    """Three ``B`` (width_i x r), each applied to its own rank-r chunk of the input."""

    def __init__(self, r: int, split_sizes: Iterable[int]):
        super().__init__()
        self.r = r
        self.q, self.k, self.v = (nn.Linear(r, width, bias=False) for width in split_sizes)

    @property
    def weight(self) -> torch.Tensor:  # block-diagonal: B @ A is the fused delta weight
        return torch.block_diag(self.q.weight, self.k.weight, self.v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = x.split(self.r, dim=-1)
        return torch.cat([self.q(q), self.k(k), self.v(v)], dim=-1)


def _lora_linear_cls():
    from peft.tuners.lora.layer import Linear

    return Linear


class SplitQKVLoraLinear(_lora_linear_cls()):
    """PEFT LoRA layer for :class:`SplitQKVLinear`: per-slice LoRAs, everything else is PEFT's ``Linear``."""

    def update_layer(self, adapter_name, r, *args, init_lora_weights=True, **kwargs):
        super().update_layer(adapter_name, r, *args, init_lora_weights=False, **kwargs)
        if kwargs.get("use_dora") or kwargs.get("lora_bias"):
            raise NotImplementedError("split q/k/v LoRA supports plain LoRA only")
        self.lora_A[adapter_name] = _SplitA(self.in_features, r)
        self.lora_B[adapter_name] = _SplitB(r, self.get_base_layer().split_sizes)
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False or adapter_name not in self.lora_A:
            return
        lora_a, lora_b = self.lora_A[adapter_name], self.lora_B[adapter_name]
        if not isinstance(lora_a, _SplitA):  # PEFT's own init during super().update_layer
            return super().reset_lora_parameters(adapter_name, init_lora_weights)
        for name in SLICES:
            weight = getattr(lora_a, name).weight
            if init_lora_weights is True:  # same as PEFT / nn.Linear for each A
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            elif str(init_lora_weights).lower() == "gaussian":
                nn.init.normal_(weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"unsupported init_lora_weights={init_lora_weights!r} for split q/k/v LoRA")
            nn.init.zeros_(getattr(lora_b, name).weight)


def lora_config_class():
    """``peft.LoraConfig`` subclass that routes :class:`SplitQKVLinear` targets to :class:`SplitQKVLoraLinear`."""
    from peft import LoraConfig

    class SplitQKVLoraConfig(LoraConfig):
        def __post_init__(self):
            super().__post_init__()
            self._register_custom_module({SplitQKVLinear: SplitQKVLoraLinear})

    SplitQKVLoraConfig.__name__ = "LoraConfig"  # keeps saved peft_type / repr identical
    return SplitQKVLoraConfig


def split_peft_name(name: str) -> str:
    """``…in_proj_qkv.lora_A[.adapter].q.weight`` -> ``…in_proj_q.lora_A.weight`` (PEFT file naming, no adapter).

    Other names are returned unchanged.
    """
    match = _SPLIT_PARAM.match(name)
    if match is None:
        return name
    return f"{match['prefix']}in_proj_{match['slice']}.lora_{match['ab']}.weight"


def split_legacy_tensors(tensors: Mapping[str, torch.Tensor], split_sizes: tuple[int, int, int]) -> dict:
    """PEFT tensors with fused ``in_proj_qkv`` LoRAs -> the exact ``in_proj_{q,k,v}`` form (A copied, B sliced)."""
    out = {}
    for name, tensor in tensors.items():
        match = _FUSED_PARAM.match(name)
        if match is None:
            out[name] = tensor
            continue
        parts = [tensor] * 3 if match["ab"] == "A" else list(tensor.split(list(split_sizes), dim=0))
        for slice_name, part in zip(SLICES, parts, strict=True):
            out[f"{match['prefix']}in_proj_{slice_name}.lora_{match['ab']}{match['adapter'] or ''}.weight"] = (
                part.clone()
            )
    return out


def legacy_split_names(state_dict_keys: Iterable[str]) -> dict[str, list[str]]:
    """Fused ``in_proj_qkv`` LoRA keys of a checkpoint -> the keys of the split layer they become.

    ``…in_proj_qkv.lora_A.default.weight`` -> ``…in_proj_qkv.lora_A.default.{q,k,v}.weight``.
    """
    out = {}
    for key in state_dict_keys:
        match = _FUSED_PARAM.match(key)
        if match is not None:
            base = f"{match['prefix']}in_proj_qkv.lora_{match['ab']}{match['adapter'] or ''}"
            out[key] = [f"{base}.{name}.weight" for name in SLICES]
    return out


def full_tensor(parts: list) -> torch.Tensor:
    """One parameter from every rank's sharded checkpoint entry (DTensor, ShardedTensor or plain)."""
    first = parts[0]
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        DTensor = ()  # noqa: N806
    if isinstance(first, DTensor):
        (placement,) = first.placements
        if type(placement).__name__ == "Replicate":
            return first.to_local()
        return torch.cat([part.to_local() for part in parts], dim=placement.dim)
    if hasattr(first, "local_shards") and hasattr(first, "metadata"):  # FSDP1 ShardedTensor
        full = torch.zeros(tuple(first.size()), dtype=first.dtype)
        for part in parts:
            for shard in part.local_shards():
                index = tuple(
                    slice(o, o + n)
                    for o, n in zip(shard.metadata.shard_offsets, shard.metadata.shard_sizes, strict=True)
                )
                full[index] = shard.tensor.to("cpu")
        return full
    return first


def convert_legacy_full(full: Mapping[str, torch.Tensor], names: Mapping[str, list[str]], split_sizes) -> dict:
    """Full legacy tensors -> full tensors of the split layer (``split_sizes(key) -> (q, k, v)``); exact."""
    out = {}
    for key, tensor in full.items():
        targets = names.get(key)
        if targets is None:
            out[key] = tensor
            continue
        parts = [tensor] * 3 if ".lora_A" in key else list(tensor.split(list(split_sizes(key)), dim=0))
        for target, part in zip(targets, parts, strict=True):
            out[target] = part.clone()
    return out


def convert_legacy_optimizer_state(
    full_states: Mapping[str, Mapping[str, object]], names: Mapping[str, list[str]], split_sizes
) -> dict:
    """Per-parameter Adam state of a legacy checkpoint (full tensors, keyed by parameter name) -> the split layer's.

    ``B``'s moments are sliced like ``B``; ``A``'s moments are copied to q, k and v (``A`` itself is copied,
    so this is the equivalent state); ``step`` and other scalars are kept. Other parameters are unchanged.
    """
    out = {}
    for key, state in full_states.items():
        targets = names.get(key)
        if targets is None:
            out[key] = dict(state)
            continue
        per_slice = [dict() for _ in targets]
        for field, value in state.items():
            is_moment = isinstance(value, torch.Tensor) and value.dim() == 2
            parts = value.split(list(split_sizes(key)), dim=0) if is_moment and ".lora_B" in key else [value] * 3
            for slot, part in zip(per_slice, parts, strict=True):
                slot[field] = part.clone() if isinstance(part, torch.Tensor) else part
        out.update(zip(targets, per_slice, strict=True))
    return out


__all__ = [
    "convert_legacy_full",
    "convert_legacy_optimizer_state",
    "full_tensor",
    "legacy_split_names",
    "SplitQKVLinear",
    "SplitQKVLoraLinear",
    "lora_config_class",
    "mark_split_qkv",
    "split_legacy_tensors",
    "split_peft_name",
]
