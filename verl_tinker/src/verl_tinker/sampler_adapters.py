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

"""Sample saved sampler checkpoints as extra vLLM LoRA adapters beside the training weights.

The colocated rollout holds the current actor weights as LoRA ``VLLM_LORA_INT_ID`` (123). A
saved sampler checkpoint whose weights are no longer in the rollout (an older checkpoint of
this session, or another session's checkpoint the EnvLoop gateway staged into this head's
checkpoint root) is served by loading its PEFT export (``<tag>/lora_adapter``) as a separate
adapter and naming it per request. The training weights and adapter 123 are never touched.

Slot management is vLLM's own: each request carries a path-backed ``LoRARequest``, which vLLM's
``LRUCacheWorkerLoRAManager`` loads on first use and evicts least-recently-used across
``max_loras`` GPU slots and ``max_cpu_loras`` CPU slots (the scheduler never batches more adapters
than slots). The current weights (123) are pushed as tensors and cannot be reloaded from a path, so
they are pinned (``pin_lora``) after every weight sync. This module only names adapters
(``SamplerAdapterIds``: a stable ``lora_int_id`` per directory) and checks compatibility before a
request reaches the engine, where a failed adapter load would fail inside the engine step.

Modelled on SkyRL (Apache-2.0, NovaSky-AI/SkyRL): ``skyrl/tinker/extra/external_inference.py``
(adapters referenced by name from a local directory) and
``skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py`` (``add_lora`` with a
name -> ``lora_int_id`` map, the adapter resolved per generate request).
"""

from __future__ import annotations

import contextvars
import json
import re
from pathlib import Path
from typing import Any

ADAPTER_PATH_PREFIX = "sampler-adapter://"
ADAPTER_DIR = "lora_adapter"
# vLLM's allowed ``max_lora_rank`` values (``vllm.config.lora.MaxLoRARanks``).
VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
# First id for sampler adapters; the current-weights adapter is VeRL's VLLM_LORA_INT_ID = 123.
FIRST_ADAPTER_INT_ID = 1000
UNEMBED_MODULES = frozenset({"lm_head"})
# Separate linear-attention q/k/v LoRAs (instead of a fused in_proj_qkv); loadable only by a recipe whose
# vLLM reports the ``lora_split_qkv`` capability.
SPLIT_QKV_MODULES = frozenset({"in_proj_q", "in_proj_k", "in_proj_v"})
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}")


class SamplerAdapterError(ValueError):
    """Base class; ``code`` is the typed error code returned to the gateway."""

    code = "SAMPLER_ADAPTER_INVALID"


class SamplerAdapterNotFound(SamplerAdapterError):
    code = "SAMPLER_ADAPTER_NOT_FOUND"


class SamplerAdapterIncompatible(SamplerAdapterError):
    code = "SAMPLER_ADAPTER_INCOMPATIBLE"


def vllm_max_lora_rank(rank: int) -> int:
    """The smallest vLLM ``max_lora_rank`` that holds ``rank`` (VeRL's get_vllm_max_lora_rank)."""
    for value in VLLM_MAX_LORA_RANKS:
        if rank <= value:
            return value
    raise SamplerAdapterIncompatible(f"LoRA rank {rank} exceeds vLLM's maximum {VLLM_MAX_LORA_RANKS[-1]}")


def adapter_tag(model_path: str | None) -> str | None:
    """The checkpoint-root tag named by ``sampler-adapter://<tag>``, else None."""
    if not isinstance(model_path, str) or not model_path.startswith(ADAPTER_PATH_PREFIX):
        return None
    tag = model_path[len(ADAPTER_PATH_PREFIX) :]
    if not _TAG.fullmatch(tag) or tag in {".", ".."}:
        raise SamplerAdapterError(f"Invalid sampler adapter path: {model_path!r}")
    return tag


def adapter_dir(checkpoint_root: str | Path, tag: str) -> Path:
    """``<root>/<tag>/lora_adapter``; never resolves outside the checkpoint root."""
    root = Path(checkpoint_root).resolve()
    path = (root / tag / ADAPTER_DIR).resolve()
    if root not in path.parents:
        raise SamplerAdapterError(f"Invalid sampler adapter tag: {tag!r}")
    return path


def read_adapter_config(path: str | Path) -> dict:
    path = Path(path)
    config_file, weights_file = path / "adapter_config.json", path / "adapter_model.safetensors"
    if not config_file.is_file() or not weights_file.is_file():
        raise SamplerAdapterNotFound(f"No PEFT adapter at {path.parent.name}")
    try:
        config = json.loads(config_file.read_text())
    except (OSError, ValueError) as exc:
        raise SamplerAdapterError("adapter_config.json is not valid JSON") from exc
    if not isinstance(config, dict):
        raise SamplerAdapterError("adapter_config.json must be an object")
    return config


def adapter_modules(config: dict) -> frozenset[str]:
    modules = config.get("target_modules") or ()
    if isinstance(modules, str):
        # PEFT allows a regex or "all-linear"; EnvLoop exports always list module names.
        raise SamplerAdapterIncompatible("adapter target_modules must be an explicit list")
    return frozenset(str(module).rsplit(".", 1)[-1] for module in modules)


def check_adapter_compatible(
    config: dict, *, max_lora_rank: int, unembed_enabled: bool, split_qkv_enabled: bool = False
) -> None:
    """The single head-side compatibility rule (mirrored by the gateway's sampling dispatch).

    vLLM wraps every supported linear layer of the base model with LoRA regardless of which
    adapter is active, so attention-only or MLP-only adapters load on any LoRA head. What the
    head's engine fixes at start is the maximum rank and whether the unembedding (``lm_head``)
    carries LoRA; an adapter outside either is incompatible (the gateway sends it elsewhere).
    """
    if config.get("peft_type", "LORA") != "LORA":
        raise SamplerAdapterIncompatible(f"unsupported adapter type {config.get('peft_type')!r}")
    rank = config.get("r")
    if type(rank) is not int or rank < 1:
        raise SamplerAdapterIncompatible("adapter rank (r) must be a positive integer")
    if rank > max_lora_rank:
        raise SamplerAdapterIncompatible(
            f"adapter rank {rank} exceeds this session's sampler max_lora_rank {max_lora_rank}"
        )
    modules = adapter_modules(config)
    if not modules:
        raise SamplerAdapterIncompatible("adapter has no target_modules")
    if modules & UNEMBED_MODULES and not unembed_enabled:
        raise SamplerAdapterIncompatible(
            "adapter trains the unembedding (lm_head) but this session's sampler was started without lm_head LoRA"
        )
    if modules & SPLIT_QKV_MODULES and not split_qkv_enabled:
        raise SamplerAdapterIncompatible(
            "adapter has separate q/k/v linear-attention LoRAs, which this session's sampler cannot load"
        )
    if config.get("use_dora") or config.get("modules_to_save"):
        raise SamplerAdapterIncompatible("DoRA adapters and modules_to_save are not supported")


class SamplerAdapterIds:
    """A stable vLLM ``lora_int_id`` per adapter directory (never VeRL's current-weights id 123)."""

    def __init__(self):
        self._ids: dict[str, int] = {}

    def spec(self, path: str | Path) -> dict:
        key = str(Path(path))
        lora_int_id = self._ids.setdefault(key, FIRST_ADAPTER_INT_ID + len(self._ids))
        return {"lora_name": f"sampler-{lora_int_id}", "lora_int_id": lora_int_id, "lora_path": key}


# ==================== vLLM server side (used by backends/vllm_multi_lora.py) ====================

# The LoRARequest a rollout server's ``generate`` decodes the current request with.
REQUEST_LORA: contextvars.ContextVar[Any] = contextvars.ContextVar("verl_tinker_request_lora", default=None)


class LoRAOverrideEngine:
    """Delegate to vLLM's engine; ``generate`` uses the adapter chosen for the current request."""

    def __init__(self, engine):
        object.__setattr__(self, "_engine", engine)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def __setattr__(self, name, value):
        setattr(self._engine, name, value)

    def generate(self, *args, lora_request=None, **kwargs):
        override = REQUEST_LORA.get()
        return self._engine.generate(*args, lora_request=override or lora_request, **kwargs)


def multi_lora_engine_args(args, engine_kwargs: dict | None) -> None:
    """Re-apply ``max_loras`` / ``max_cpu_loras`` / ``max_lora_rank`` that VeRL's launch_server overwrote."""
    engine_kwargs = engine_kwargs or {}
    for key in ("max_loras", "max_cpu_loras"):
        value = engine_kwargs.get(key)
        if value is not None:
            setattr(args, key, int(value))
    if getattr(args, "max_cpu_loras", None) is not None and getattr(args, "max_loras", None) is not None:
        args.max_cpu_loras = max(int(args.max_cpu_loras), int(args.max_loras))
    rank = engine_kwargs.get("max_lora_rank")
    if rank is not None and getattr(args, "enable_lora", False):
        args.max_lora_rank = max(int(args.max_lora_rank or 0), vllm_max_lora_rank(int(rank)))
