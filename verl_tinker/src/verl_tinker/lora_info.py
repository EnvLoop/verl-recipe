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

"""The server's real LoRA configuration, as reported to Tinker clients.

The adapter is created once at server start from
``actor_rollout_ref.model.{lora_rank, lora_alpha, target_modules,
exclude_modules}`` (FSDP / VeOmni) or ``actor_rollout_ref.model.lora.{rank,
alpha, target_modules}`` (Megatron-style). EnvLoop sets the rank through
``TINKER_LORA_RANK`` in its guest yaml. A rank of 0 means full-parameter
training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

# PEFT module-name families used to derive Tinker's train_attn / train_mlp /
# train_unembed flags from ``target_modules``.
_ATTN_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "query_key_value",
        "c_attn",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "in_proj",
        "out_proj",
        "self_attn",
        "attn",
        "attention",
    }
)
_MLP_MODULES = frozenset(
    {
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "fc1",
        "fc2",
        "c_fc",
        "w1",
        "w2",
        "w3",
        "dense_h_to_4h",
        "dense_4h_to_h",
        "mlp",
        "experts",
        "shared_experts",
    }
)
_UNEMBED_MODULES = frozenset({"lm_head", "embed_out", "output_layer"})


@dataclass(frozen=True)
class LoraInfo:
    rank: int
    alpha: Optional[float]
    target_modules: Any
    train_attn: bool
    train_mlp: bool
    train_unembed: bool

    @property
    def is_lora(self) -> bool:
        return self.rank > 0

    def metadata(self) -> dict[str, Any]:
        """Tinker-shaped LoRA fields (``WeightsInfoResponse`` / training-run metadata)."""
        if not self.is_lora:
            return {"is_lora": False, "lora_rank": None}
        return {
            "is_lora": True,
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "train_unembed": self.train_unembed,
            "train_mlp": self.train_mlp,
            "train_attn": self.train_attn,
        }


FULL_PARAMETER = LoraInfo(rank=0, alpha=None, target_modules=None, train_attn=True, train_mlp=True, train_unembed=True)


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, DictConfig):
        return OmegaConf.select(node, key, default=default)
    if isinstance(node, Mapping):
        return node.get(key, default)
    return default


def _module_leaves(target_modules: Any) -> list[str]:
    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        return [target_modules]
    return [str(name) for name in target_modules]


def train_flags_for_target_modules(target_modules: Any) -> tuple[bool, bool, bool]:
    """``(train_attn, train_mlp, train_unembed)`` implied by PEFT ``target_modules``.

    PEFT's ``"all-linear"`` adapts every linear layer except the output
    (unembedding) layer.
    """
    if target_modules is None or target_modules == "all-linear":
        return True, True, False
    leaves = [name.rsplit(".", 1)[-1] for name in _module_leaves(target_modules)]
    return (
        any(name in _ATTN_MODULES for name in leaves),
        any(name in _MLP_MODULES for name in leaves),
        any(name in _UNEMBED_MODULES for name in leaves),
    )


def configured_lora_info(config: Any) -> LoraInfo:
    """LoRA configuration the server was started with (rank 0 = full-parameter)."""
    model = _get(_get(config, "actor_rollout_ref"), "model")
    if not isinstance(model, (DictConfig, Mapping)):
        return FULL_PARAMETER

    lora = _plain(_get(model, "lora")) or {}
    rank = int(_get(lora, "rank", 0) or 0)
    if rank <= 0:
        rank = int(_get(model, "lora_rank", 0) or 0)
    if rank <= 0:
        return FULL_PARAMETER

    alpha = _get(lora, "alpha")
    if alpha is None:
        alpha = _get(model, "lora_alpha")
    target_modules = _plain(_get(lora, "target_modules"))
    if target_modules is None:
        target_modules = _plain(_get(model, "target_modules", "all-linear"))
    train_attn, train_mlp, train_unembed = train_flags_for_target_modules(target_modules)
    return LoraInfo(
        rank=rank,
        alpha=float(alpha) if alpha is not None else None,
        target_modules=target_modules,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
    )
