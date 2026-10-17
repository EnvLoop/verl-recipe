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

"""vLLM general plugin: serve separate q/k/v LoRAs on the Qwen3.5 Gated DeltaNet ``in_proj_qkvz``.

Adapters trained with ``verl_tinker.lora_split`` (the reference layout) carry ``in_proj_{q,k,v}`` LoRAs, each with
its own ``A``. vLLM's LoRA layer for the fused ``in_proj_qkvz`` already has four output slices [q, k, v, z], each
with its own ``lora_a`` / ``lora_b`` buffers at rank r; it only knows the module names ``in_proj_qkv`` and
``in_proj_z`` (``packed_modules_mapping``) and copies the one ``in_proj_qkv`` ``A`` into the q/k/v slices.

This plugin (vLLM 0.26 and 0.31, idempotent):

* sets ``in_proj_qkvz: [in_proj_q, in_proj_k, in_proj_v, in_proj_z, in_proj_qkv]`` on the Qwen3.5 model classes,
  so both layouts are expected adapter modules;
* before vLLM packs an adapter's LoRAs, rewrites a legacy fused ``in_proj_qkv`` LoRA into exact ``in_proj_{q,k,v}``
  LoRAs (``A`` copied, ``B`` sliced by the layer's output sizes), and after packing keeps the four slice entries
  [q, k, v, z], which vLLM then loads one-to-one into its slices.

No upstream vLLM version supports per-slice LoRAs on this layer by name (checked 0.26.0 - 0.31.0).
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import pathlib

logger = logging.getLogger(__name__)

SPLIT = ("in_proj_q", "in_proj_k", "in_proj_v")
ENTRY_POINT = f"{__name__}:register"
# What the patches rely on in the installed vLLM tree (relative to the ``vllm`` package).
HOOKS = {
    "model_executor/models/qwen3_5.py": '"in_proj_qkvz"',
    "lora/model_manager.py": "def _create_merged_loras_inplace(",
}
PACKED = [*SPLIT, "in_proj_z", "in_proj_qkv"]


def _patch_mapping() -> None:
    from vllm.model_executor.models import qwen3_5

    for name in dir(qwen3_5):
        cls = getattr(qwen3_5, name)
        mapping = getattr(cls, "packed_modules_mapping", None)
        if isinstance(cls, type) and isinstance(mapping, dict) and "in_proj_qkvz" in mapping:
            cls.packed_modules_mapping = {**mapping, "in_proj_qkvz": list(PACKED)}


def _split_legacy(manager, lora_model) -> None:
    from vllm.lora.lora_weights import LoRALayerWeights

    for packed, members in manager.packed_modules.items():
        if not packed.endswith(".in_proj_qkvz"):
            continue
        prefix = packed[: -len("in_proj_qkvz")]
        fused = lora_model.loras.get(prefix + "in_proj_qkv")
        if fused is None or any(prefix + name in lora_model.loras for name in SPLIT):
            continue
        sizes = list(manager.modules[packed].output_sizes[:3])  # full (not TP-sharded) q, k, v widths
        for name, lora_b in zip(SPLIT, fused.lora_b.split(sizes, dim=0), strict=True):
            lora_model.loras[prefix + name] = LoRALayerWeights(
                prefix + name, fused.rank, fused.lora_alpha, fused.lora_a.clone(), lora_b.clone(), fused.scaling
            )
        del lora_model.loras[prefix + "in_proj_qkv"]


def _keep_slice_entries(manager, lora_model) -> None:
    for packed in manager.packed_modules:
        lora = lora_model.loras.get(packed)
        if not packed.endswith(".in_proj_qkvz") or lora is None or len(lora.lora_a) != len(PACKED):
            continue
        for field in ("lora_a", "lora_b", "lora_alphas", "scaling"):
            setattr(lora, field, list(getattr(lora, field))[:4])  # q, k, v, z (in_proj_qkv was split)


def _patch_manager() -> None:
    from vllm.lora.model_manager import LoRAModelManager

    original = LoRAModelManager._create_merged_loras_inplace
    if getattr(original, "_envloop_split_qkv", False):
        return

    def _create_merged_loras_inplace(self, lora_model):
        _split_legacy(self, lora_model)
        result = original(self, lora_model)
        _keep_slice_entries(self, lora_model)
        return result

    _create_merged_loras_inplace._envloop_split_qkv = True
    LoRAModelManager._create_merged_loras_inplace = _create_merged_loras_inplace


def register() -> None:
    try:
        _patch_mapping()
        _patch_manager()
    except ImportError:  # a vLLM without Qwen3.5 or LoRA support has nothing to patch
        return
    logger.info("verl_tinker: per-slice q/k/v LoRA enabled for Qwen3.5 in_proj_qkvz")


def enabled(vllm_root: pathlib.Path | None) -> bool:
    """vLLM will run ``register`` in its engine-core and worker processes. Reads metadata and files only.

    True when this package's ``vllm.general_plugins`` entry point is installed, ``VLLM_PLUGINS`` (vLLM's
    allowlist; unset loads every plugin) does not exclude it, and the installed vLLM has the hooks it patches.
    """
    names = [ep.name for ep in importlib.metadata.entry_points(group="vllm.general_plugins") if ep.value == ENTRY_POINT]
    allowed = os.environ.get("VLLM_PLUGINS")
    if not names or (allowed is not None and not set(names) & set(allowed.split(","))):
        return False
    return vllm_root is not None and all(
        (vllm_root / relative).is_file() and marker in (vllm_root / relative).read_text()
        for relative, marker in HOOKS.items()
    )
