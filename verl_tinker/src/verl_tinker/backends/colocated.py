# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Tinker server engine: FSDP actor + optional rollout on the same Ray placement group.

Mirrors the worker-init logic in VeRL's ``RayPPOTrainer.init_workers()``
(verl/trainer/ppo/ray_trainer.py), but runs independently on the Tinker server
side. Creates Ray worker groups for actor/rollout, initializes vLLM rollout
replicas directly on the SAME GPU pool, and manages a
sleep/wake state machine so FSDP and vLLM can share the GPUs.

This is the only supported Tinker server engine. It owns the cross-request
serialization lock for synchronous training and checkpoint operations.
"""

import asyncio
import math
import logging
import threading
import time
from copy import deepcopy
from enum import Enum
from typing import Any, Optional

import ray
from omegaconf import DictConfig, OmegaConf

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.protocol import DataProtoFuture
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

from ..config_utils import is_no_rollout_deployment, needs_reference_policy
from ..schemas import ServerCapabilities
from ._loss import is_ref_in_actor, make_branching_loss
from .model_lifecycle import ModelLifecycle, ModelRole

logger = logging.getLogger("ray")

# Error code of a sampler request whose rollout engine could not be started (EnvLoop).
ROLLOUT_ENGINE_START_FAILED = "ROLLOUT_ENGINE_START_FAILED"


class RolloutState(str, Enum):
    """Whether the rollout replicas exist (``server.rollout_start``)."""

    DISABLED = "disabled"  # no-rollout deployment
    NOT_STARTED = "not_started"  # lazy: nothing has needed the sampler yet
    READY = "ready"
    FAILED = "failed"


class RolloutStartError(RuntimeError):
    """The rollout replicas failed to start; later sampler requests fail fast with the same error."""

    code = ROLLOUT_ENGINE_START_FAILED


def effective_clip_grad(grad_clip_norm: Optional[float]) -> Optional[float]:
    """Map Tinker ``AdamParams.grad_clip_norm`` onto VeRL's ``optim.clip_grad``.

    ``None`` keeps the server-start ``clip_grad``; ``> 0`` clips to that global
    norm; ``0`` / negative disables clipping. VeRL always calls
    ``clip_grad_norm_(max_norm=clip_grad)`` (and asserts it is not ``None``), so
    "disabled" is ``inf``: the clip coefficient ``inf / (norm + eps)`` is
    clamped to 1 and gradients are left unchanged, while the returned pre-clip
    norm is still computed.
    """
    if grad_clip_norm is None:
        return None
    value = float(grad_clip_norm)
    if value != value:  # NaN
        raise ValueError("grad_clip_norm must be a number, got NaN")
    return value if value > 0 else float("inf")


def _on_sender_device(weight):
    """``weight`` on the current accelerator; host tensors are copied there."""
    if getattr(getattr(weight, "device", None), "type", None) != "cpu":
        return weight
    from verl.utils.device import get_device_id, get_device_name

    return weight.to(f"{get_device_name()}:{get_device_id()}")


def _patch_large_weight_ipc_device() -> None:
    """Send CPU-resident oversized weights from the current device.

    VeRL streams rollout weights through a fixed-size device bucket, but a
    weight larger than the bucket is shared directly with
    ``reduce_tensor(weight)``. With an FSDP1 actor the gathered full weights are
    CPU tensors, so that handle is a CPU-storage handle without a device slot and
    the vLLM receiver fails in ``rebuild_ipc`` (``list_args[6] = device_id``:
    IndexError). Qwen3.8-27B's ~2.5 GB embedding exceeds the default 2 GB bucket.
    Copy such a weight to the sender's device first; the copy lives until the
    receiver acknowledges the transfer. Runs at import, i.e. in every Ray worker
    that loads this module's worker classes.
    """
    try:
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
    except Exception:  # noqa: BLE001 - a VeRL without bucketed transfer has nothing to patch
        return
    original = BucketedWeightSender._direct_send_large_weight
    if getattr(original, "_envloop_device_patch", False):
        return

    def _direct_send_large_weight(self, name, weight):
        return original(self, name, _on_sender_device(weight))

    _direct_send_large_weight._envloop_device_patch = True
    BucketedWeightSender._direct_send_large_weight = _direct_send_large_weight


_patch_large_weight_ipc_device()


def _clean_fsdp_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "").replace("._fsdp_wrapped_module", "")


def _layered_lora_params(fsdp_module):
    """Collect the LoRA adapter one FSDP unit at a time; every adapter tensor or an error.

    Replaces VeRL's ``layered_summon_lora_params``, which tests the *unit-relative*
    parameter names for ``lora_``. VeRL's LoRA wrap policy wraps each trainable LoRA
    leaf (``...q_proj.lora_A.default``) as its own FSDP unit, whose only parameter is
    ``weight``, so that collector finds nothing and VeRL falls back to summoning the
    whole model (Qwen3.8-27B: ~54 GB per GPU, an OOM on 80 GB cards next to vLLM).
    Here the test uses the full name, only units that own LoRA tensors are gathered
    (a few MB each), and the result must contain every adapter tensor of the model.
    """
    from collections import OrderedDict
    from contextlib import nullcontext

    from peft.utils.save_and_load import get_peft_model_state_dict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.utils.device import get_torch_device
    from verl.utils.fsdp_utils import fsdp_version

    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    expected = set(get_peft_model_state_dict(peft_model, state_dict={
        _clean_fsdp_name(name): param for name, param in fsdp_module.named_parameters()
        if "lora_" in _clean_fsdp_name(name)
    }))
    lora_params = OrderedDict()
    for name, submodule in fsdp_module.named_modules():
        if name == "" or fsdp_version(submodule) == 0:
            continue
        prefix = _clean_fsdp_name(name)
        # FSDP1 strips wrapper prefixes from parameter names only inside summon_full_params.
        nested = {_clean_fsdp_name(n) for n, m in submodule.named_modules() if n != "" and fsdp_version(m) > 0}

        def own(unit=submodule, nested=nested, prefix=prefix):
            for n, param in unit.named_parameters():
                n = _clean_fsdp_name(n)
                if any(n.startswith(f"{child}.") for child in nested):
                    continue
                full = f"{prefix}.{n}"
                if "lora_" in full:
                    yield full, param

        if next(own(), None) is None:
            continue
        is_fsdp1 = fsdp_version(submodule) == 1
        if is_fsdp1:
            submodule._is_root = True  # as VeRL's collector: lets an inner unit summon on its own
        try:
            with FSDP.summon_full_params(submodule, writeback=False) if is_fsdp1 else nullcontext():
                for key, param in get_peft_model_state_dict(peft_model, state_dict=dict(own())).items():
                    full = param.full_tensor() if hasattr(param, "full_tensor") else param
                    lora_params[key] = full.detach().cpu()
        finally:
            if is_fsdp1:
                submodule._is_root = False
        get_torch_device().empty_cache()
    missing = expected - set(lora_params)
    if missing or not lora_params:
        raise RuntimeError(f"layered LoRA collection missed {len(missing)} of {len(expected)} adapter tensors, "
                           f"e.g. {sorted(missing)[:3]}")
    return lora_params


def _patch_layered_lora_collection() -> None:
    """Install ``_layered_lora_params`` for ``collect_lora_params(layered_summon=True)``."""
    try:
        from verl.utils import fsdp_utils
    except Exception:  # noqa: BLE001 - nothing to patch without VeRL's FSDP utilities
        return
    if getattr(fsdp_utils.layered_summon_lora_params, "_envloop_layered_patch", False):
        return
    _layered_lora_params._envloop_layered_patch = True
    _layered_lora_params.upstream = fsdp_utils.layered_summon_lora_params
    fsdp_utils.layered_summon_lora_params = _layered_lora_params


_patch_layered_lora_collection()


def lora_init_seed(model_config) -> Optional[int]:
    """Tinker ``LoraConfig.seed`` (EnvLoop: ``actor_rollout_ref.model.lora.seed``), or None."""
    lora = getattr(model_config, "lora", None) or {}
    seed = lora.get("seed") if hasattr(lora, "get") else None
    if seed is None or getattr(model_config, "lora_adapter_path", None) is not None:
        return None
    return int(seed)


def _patch_lora_build() -> None:
    """Build the LoRA adapter like the reference: per-slice q/k/v LoRAs and a seeded init.

    * Split q/k/v (``lora_split``): every Gated DeltaNet ``in_proj_qkv`` is marked before
      ``get_peft_model`` and VeRL's ``LoraConfig`` is swapped for one that routes it to the split
      LoRA layer, so the fused projection trains three independent LoRAs (q, k, v) like the
      reference. The base weight and its name are unchanged. Not for adapters loaded from
      ``lora_adapter_path`` (PEFT rebuilds those from their own config);
      ``actor_rollout_ref.model.lora.split_qkv: false`` keeps the fused LoRA (previous layout).
    * Seed: VeRL creates the adapter with ``get_peft_model`` (``lora_A`` Kaiming uniform from the
      global torch RNG, ``lora_B`` zero) before FSDP wrapping, and rank 0's initialised weights are
      what FSDP broadcasts to every rank. Seeding the RNGs (CPU and the current accelerator) for
      just that call gives the same adapter for the same seed (``LoraConfig.seed``); the global
      RNG state is restored afterwards. Without a seed nothing changes.

    Runs at import, like the patch above.
    """
    try:
        from verl.workers.engine.fsdp import transformer_impl
    except Exception:  # noqa: BLE001 - a VeRL without the FSDP engine has nothing to patch
        return
    from contextlib import ExitStack

    from ..lora_split import lora_config_class, mark_split_qkv

    FSDPEngine = transformer_impl.FSDPEngine
    original = FSDPEngine._build_lora_module
    if getattr(original, "_envloop_lora_patch", False):
        return

    def _build_lora_module(self, module):
        import torch

        with ExitStack() as stack:
            lora = getattr(self.model_config, "lora", None) or {}
            split = lora.get("split_qkv", True) if hasattr(lora, "get") else True
            if split and getattr(self.model_config, "lora_adapter_path", None) is None and mark_split_qkv(module):
                default_config = transformer_impl.LoraConfig
                transformer_impl.LoraConfig = lora_config_class()
                stack.callback(setattr, transformer_impl, "LoraConfig", default_config)
            seed = lora_init_seed(self.model_config)
            if seed is not None:
                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                stack.enter_context(torch.random.fork_rng(devices=devices))
                torch.manual_seed(seed)
            return original(self, module)

    _build_lora_module._envloop_lora_patch = True
    _build_lora_module._envloop_seed_patch = True
    FSDPEngine._build_lora_module = _build_lora_module

    original_params = FSDPEngine.get_per_tensor_param
    if getattr(original_params, "_envloop_lora_patch", False):
        return

    def get_per_tensor_param(self, *args, **kwargs):
        """Adapter tensors for vLLM in PEFT file naming (``in_proj_{q,k,v}`` for the split LoRAs)."""
        from ..lora_split import split_peft_name

        params, peft_config = original_params(self, *args, **kwargs)
        if peft_config is None:
            return params, peft_config
        return ((split_peft_name(name), tensor) for name, tensor in params), peft_config

    get_per_tensor_param._envloop_lora_patch = True
    FSDPEngine.get_per_tensor_param = get_per_tensor_param


_patch_lora_build()


def _assign_full_params(model, full: dict) -> int:
    """Copy full tensors into a (FSDP1 or FSDP2) model's parameters by name; returns how many were set."""
    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    def clean(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")

    assigned = 0
    with torch.no_grad():
        if any(isinstance(p, DTensor) for p in model.parameters()):  # FSDP2
            for name, param in model.named_parameters():
                tensor = full.get(clean(name))
                if tensor is None:
                    continue
                local = distribute_tensor(
                    tensor.to(param.device, param.dtype), param.device_mesh, param.placements, src_data_rank=None
                ).to_local()
                param.to_local().copy_(local)
                assigned += 1
            return assigned
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        prefixes = {id(module): name for name, module in model.named_modules()}
        for unit in FSDP.fsdp_modules(model):  # each LoRA leaf is its own FSDP1 unit
            with FSDP.summon_full_params(unit, recurse=False, writeback=True):
                for name, param in unit.named_parameters():
                    key = clean(f"{prefixes[id(unit)]}.{name}" if prefixes[id(unit)] else name)
                    tensor = full.get(key)
                    if tensor is not None and tuple(param.shape) == tuple(tensor.shape):
                        param.copy_(tensor.to(param.device, param.dtype))
                        assigned += 1
    return assigned


def _full_state_value(parts: list, shape) -> object:
    """One optimizer-state entry from every rank: DTensor (FSDP2), flat local views (FSDP1) or a scalar."""
    import torch

    from ..lora_split import full_tensor

    first = parts[0]
    if not isinstance(first, torch.Tensor) or first.dim() == 0:
        return first
    if type(first).__name__ == "DTensor":
        return full_tensor(parts)
    if tuple(first.shape) == tuple(shape):  # unsharded (one rank)
        return first
    flat = torch.cat([part.reshape(-1) for part in parts])  # FSDP1 use_orig_params: 1-D local views in rank order
    return flat[: math.prod(shape)].reshape(shape)


def _local_state_value(value, param, world_size: int, rank: int):
    """A full optimizer-state tensor -> this rank's piece, shaped like the live (sharded) parameter."""
    import torch
    from torch.distributed.tensor import DTensor, distribute_tensor

    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return value
    if isinstance(param, DTensor):  # FSDP2
        return distribute_tensor(value.to(param.device), param.device_mesh, param.placements, src_data_rank=None)
    flat = value.reshape(-1)  # FSDP1: every LoRA leaf is its own unit, padded to a multiple of world_size
    chunk = -(-flat.numel() // world_size)
    local = flat[rank * chunk : (rank + 1) * chunk]
    if local.numel() != param.numel():
        raise ValueError("unexpected FSDP1 shard layout for a LoRA parameter")
    return local.reshape(param.shape).to(param.device)


def _load_legacy_optimizer(manager, local_path: str, full_params: dict, names: dict, sizes) -> None:
    """Restore a fused-q/k/v checkpoint's Adam state into the split layer (``lora_split``).

    ``full_params``: the checkpoint's full LoRA tensors by name (in checkpoint order, the optimizer's order).
    """
    import os

    import torch

    from ..lora_split import convert_legacy_optimizer_state

    world, rank = manager.world_size, manager.rank
    saved = [
        torch.load(os.path.join(local_path, f"optim_world_size_{world}_rank_{r}.pt"), weights_only=False,
                   map_location="cpu")
        for r in range(world)
    ]
    order = [index for group in saved[0]["param_groups"] for index in group["params"]]
    if len(order) != len(full_params):
        raise ValueError("optimizer parameters do not line up with the checkpoint's LoRA parameters")
    full_states = {}
    for index, (name, tensor) in zip(order, full_params.items(), strict=True):
        states = [sd["state"].get(index, {}) for sd in saved]
        full_states[name] = {
            field: _full_state_value([state[field] for state in states], tensor.shape) for field in states[0]
        }
    converted = convert_legacy_optimizer_state(full_states, names, sizes)

    def clean(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")

    name_of = {id(p): clean(n) for n, p in manager.model.named_parameters()}
    current = manager.optimizer.state_dict()
    params = [p for group in manager.optimizer.param_groups for p in group["params"]]
    state = {}
    for index, param in enumerate(params):
        entry = converted.get(name_of.get(id(param)))
        if entry:
            state[index] = {k: _local_state_value(v, param, world, rank) for k, v in entry.items()}
    manager.optimizer.load_state_dict({"state": state, "param_groups": current["param_groups"]})


def _patch_legacy_lora_checkpoints() -> None:
    """Resume checkpoints written before the split q/k/v LoRA (one fused ``in_proj_qkv`` LoRA).

    Their adapter converts exactly (``lora_split``: ``A`` copied to q, k and v, ``B`` sliced): every
    rank reads all ranks' model (and optimizer) shards (one node), converts the full tensors and
    copies them into its own parameters. Adam state converts the same way: ``B``'s moments are
    sliced, ``A``'s copied to q, k and v, ``step`` kept, so the next step updates ``B`` exactly as
    the fused layout would (``A`` gets per-slice gradients from then on). Everything else (extra
    state, RNG) loads as before. Checkpoints in the current layout take VeRL's own path.
    """
    try:
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    except Exception:  # noqa: BLE001
        return
    original = FSDPCheckpointManager.load_checkpoint
    if getattr(original, "_envloop_legacy_lora", False):
        return

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        import os

        import torch

        from ..lora_split import SplitQKVLinear, convert_legacy_full, full_tensor, legacy_split_names

        model_file = os.path.join(local_path or "", f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        if local_path is None or not self.should_load_model or not os.path.isfile(model_file):
            return original(self, local_path, hdfs_path, del_local_after_load)
        own = torch.load(model_file, weights_only=False, map_location="cpu")
        names = legacy_split_names(own)
        split_layers = {
            name.replace("_fsdp_wrapped_module.", "").removesuffix(".base_layer"): module.split_sizes
            for name, module in self.model.named_modules()
            if isinstance(module, SplitQKVLinear)
        }
        if not names or not split_layers:
            return original(self, local_path, hdfs_path, del_local_after_load)
        shards = [own] + [
            torch.load(os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{rank}.pt"),
                       weights_only=False, map_location="cpu")
            for rank in range(self.world_size) if rank != self.rank
        ]
        shards.insert(self.rank, shards.pop(0))
        full = {key: full_tensor([shard[key] for shard in shards]) for key in own}

        def sizes(key: str):
            prefix = key.split(".in_proj_qkv.")[0] + ".in_proj_qkv"
            return split_layers[prefix]

        converted = convert_legacy_full(full, names, sizes)
        assigned = _assign_full_params(self.model, converted)
        if assigned != len(converted):
            raise RuntimeError(f"legacy LoRA checkpoint: set {assigned} of {len(converted)} adapter tensors")
        restored = "not requested"
        if self.should_load_optimizer:
            _load_legacy_optimizer(self, local_path, full, names, sizes)
            restored = "restored"
        logger.warning(
            "Loaded a fused-q/k/v LoRA checkpoint from %s as separate q/k/v LoRAs (%d tensors); optimizer state %s",
            local_path, assigned, restored,
        )
        contents = self.checkpoint_load_contents
        self.checkpoint_load_contents = [item for item in contents if item not in ("model", "optimizer")]
        try:
            return original(self, local_path, hdfs_path, del_local_after_load)
        finally:
            self.checkpoint_load_contents = contents

    load_checkpoint._envloop_legacy_lora = True
    FSDPCheckpointManager.load_checkpoint = load_checkpoint


_patch_legacy_lora_checkpoints()


class TinkerServerActorRolloutRefWorker(TinkerActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to_actor(self, device, model=True, optimizer=True, grad=True):
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to_ref(self, device, model=True, optimizer=False, grad=False):
        if self.ref is None:
            return
        self.ref.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step_with_grad_clip(self, optim_step_params=None, grad_clip_norm=None):
        """Tinker optimizer step with a per-step global grad-norm clip.

        Temporarily overrides the actor engine's ``optimizer_config.clip_grad``
        (a VeRL mutable field) for this step only; the engine's own
        ``optimizer_step`` clips the global norm of all parameters that have a
        gradient (the trainable LoRA / full parameters) and returns the
        pre-clip norm, which the Tinker worker reports as ``grad_norm``.
        """
        assert "actor" in self.role, "optimizer_step_with_grad_clip only supports the actor role"
        clip_grad = effective_clip_grad(grad_clip_norm)
        optimizer_config = getattr(getattr(self.actor, "engine", None), "optimizer_config", None)
        if clip_grad is None:
            return self.actor.optimizer_step(optim_step_params=optim_step_params)
        if optimizer_config is None or not hasattr(optimizer_config, "clip_grad"):
            raise NotImplementedError(
                f"{type(getattr(self.actor, 'engine', None)).__name__} does not expose optimizer_config.clip_grad; "
                "per-step grad_clip_norm is unsupported for this actor strategy"
            )
        previous = optimizer_config.clip_grad
        optimizer_config.clip_grad = clip_grad
        try:
            return self.actor.optimizer_step(optim_step_params=optim_step_params)
        finally:
            optimizer_config.clip_grad = previous

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer_state(self):
        """Fresh optimizer state after a weights-only load.

        A VeRL checkpoint restores parameters and optimizer state together, but
        Tinker's ``create_training_client_from_state`` (``load_weights`` with
        ``optimizer=false``) must continue from the saved weights with a new
        optimizer. Clearing the torch optimizer's per-parameter state (Adam
        moments and step counts) makes the next step a first step; parameters
        and hyperparameters are unchanged. Gradients are cleared as well.
        """
        assert "actor" in self.role, "reset_optimizer_state only supports the actor role"
        optimizer = getattr(getattr(self.actor, "engine", None), "optimizer", None)
        if optimizer is None or not hasattr(optimizer, "state"):
            raise NotImplementedError(
                f"{type(getattr(self.actor, 'engine', None)).__name__} does not expose a torch optimizer; "
                "weights-only checkpoint loads are unsupported for this actor strategy"
            )
        optimizer.state.clear()
        optimizer.zero_grad()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_actor_for_weight_sync(self):
        """Release actor training-only memory before waking rollout weights.

        The naive checkpoint engine still needs actor parameters on GPU as the
        source of the transfer. Move only optimizer state to CPU and leave both
        parameters and gradients untouched so a caller's gradient-accumulation
        state is preserved. Optimizer CPU copies are non-blocking on some
        backends, so synchronize and clear the allocator cache before vLLM
        attempts to remap its weight allocation.
        """
        self.actor.to(device="cpu", model=False, optimizer=True, grad=False)
        aggressive_empty_cache(force_sync=True)


class NoRolloutWorker(TinkerServerActorRolloutRefWorker):
    """TinkerActorRolloutRefWorker that skips rollout (vLLM) initialization."""

    def _build_rollout(self, **kwargs):
        pass


class ColocatedBackend:
    """Creates and manages VeRL RayWorkerGroups on a single GPU pool.

    Initialization order:
    1. _build_role_cls()          — build role -> worker class mapping
    2. _spawn_worker_groups()     — create resource pool, colocate, spawn
    3. _init_worker_groups()      — init actor/rollout, ref
    4. _init_rollout_replicas()   — vLLM replicas, checkpoint manager

    With ``server.rollout_start: lazy`` (EnvLoop) step 4 is deferred until the
    first request that needs the sampler (``start_rollout``, or ``update_weights``),
    so a session that only trains never loads a second model copy.

    After initialization, provides operation methods (generate,
    compute_log_prob, forward_backward, optim_step, etc.) that route to the appropriate
    worker group or replica.

    Synchronous training/checkpoint operations are serialized by
    ``_engine_lock``. Tinker router handlers run request work in threads,
    and two concurrent ``/forward_backward`` + ``/save_weights_for_sampler``
    calls can otherwise race on shared replica sleep/wake state.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self._engine_lock = threading.Lock()
        self._no_rollout_deployment: bool = is_no_rollout_deployment(config)
        self.use_kl_loss: bool = config.actor_rollout_ref.actor.get("use_kl_loss", False)
        self.use_kl_in_reward: bool = config.get("algorithm", {}).get("use_kl_in_reward", False)
        self._ref_in_actor: bool = is_ref_in_actor(config)
        # A LoRA actor can always serve base-model reference log-probs by
        # disabling its adapter, even when the server-side training algorithm
        # did not otherwise request a reference policy.
        self.use_reference_policy: bool = needs_reference_policy(config) or self._ref_in_actor
        self._enable_offload = bool(config.get("server", {}).get("enable_offload", True))

        self.actor_rollout_wg: Optional[RayWorkerGroup] = None
        self.ref_policy_wg: Optional[RayWorkerGroup] = None
        self.checkpoint_manager = None
        self.rollout_replicas = []
        self._server_manager = None
        self._resource_pool = None
        self._lifecycle: ModelLifecycle | None = None
        self._replica_wake_lock = asyncio.Lock()
        self._profile_step = 0
        self._lazy_rollout = config.get("server", {}).get("rollout_start", "eager") == "lazy"
        self._rollout_state = RolloutState.DISABLED if self._no_rollout_deployment else RolloutState.NOT_STARTED
        self._rollout_start_error: str | None = None

        # Ray workers import model.external_lib through HFModelConfig, but rollout
        # registries are process-local, so the backend must import them before
        # resolving custom rollout replica classes.
        external_libs = config.get("external_libs", None)
        if external_libs is not None:
            import_external_libs(list(external_libs))

        role_cls, actor_role = self._build_role_cls()
        all_wg = self._spawn_worker_groups(role_cls)
        self._init_worker_groups(all_wg, actor_role)
        self._lifecycle = ModelLifecycle.create(
            enable_offload=self._enable_offload,
            has_rollout=not self._no_rollout_deployment,
            has_ref=bool(self.ref_policy_wg is not None and not self._ref_in_actor),
            actor_awake=True,
        )
        if self._no_rollout_deployment:
            logger.info("No-rollout deployment: skipping rollout replica initialization")
        elif self._lazy_rollout:
            logger.info("Lazy rollout: replicas start on the first request that needs the sampler")
        else:
            self._prepare_model_roles(set(), reason="before rollout init")
            self._init_rollout_replicas()
            self._rollout_state = RolloutState.READY

    # ==================== Backend protocol — properties ====================

    @property
    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            use_kl_loss=self.use_kl_loss,
            use_kl_in_reward=self.use_kl_in_reward,
            use_critic=False,
            use_reference_policy=self.use_reference_policy,
            no_rollout_deployment=self._no_rollout_deployment,
        )

    @property
    def world_size(self) -> int:
        return self.actor_rollout_wg.world_size

    @property
    def no_rollout_deployment(self) -> bool:
        return self._no_rollout_deployment

    @property
    def server_manager(self):
        return self._server_manager

    @property
    def rollout_state(self) -> RolloutState:
        return self._rollout_state

    @property
    def rollout_start_error(self) -> str | None:
        return self._rollout_start_error

    @property
    def rollout_loads_base_weights(self) -> bool:
        """A started rollout holds the base model before any sync: vLLM read it from disk (not ``dummy``)
        and the warm-up sleep kept it, which VeRL does only for LoRA adapters (sleep level 1; full weights
        sleep at level 2 and are discarded)."""
        model = self.config.actor_rollout_ref.model
        lora = model.get("lora", None) or {}
        lora_rank = int(lora.get("rank", 0) or 0) or int(model.get("lora_rank", 0) or 0)
        lora_as_adapter = lora_rank > 0 and not lora.get("merge", False)
        return lora_as_adapter and "dummy" not in str(self.config.actor_rollout_ref.rollout.get("load_format", "dummy"))

    # ==================== Worker init ====================

    def _build_role_cls(self) -> tuple[dict, Role]:
        """Build role -> RayClassWithInitArgs mapping."""
        config = self.config

        # LoRA: ref reuses actor weights with adapters disabled (no extra memory) → ActorRollout.
        # Full fine-tune: ref is a separate frozen copy inside the same worker (2x memory) → ActorRolloutRef.
        # Ref always shares actor_rollout_wg.
        ref_in_actor = is_ref_in_actor(config)
        if self._no_rollout_deployment:
            if needs_reference_policy(config):
                raise ValueError("no_rollout_deployment does not support reference policy")
            actor_role = Role.Actor
        else:
            actor_role = (
                Role.ActorRollout if ref_in_actor or not needs_reference_policy(config) else Role.ActorRolloutRef
            )

        # ``ref.enable`` is a verl-tinker deployment switch, not an ActorConfig
        # field. Keep it in the server config for readable role selection, but
        # never pass it to VeRL's ref-config dataclass constructor.
        worker_config = deepcopy(config.actor_rollout_ref)
        if "ref" in worker_config and "enable" in worker_config.ref:
            del worker_config.ref.enable

        worker_cls = NoRolloutWorker if self._no_rollout_deployment else TinkerServerActorRolloutRefWorker
        role_cls = {
            str(actor_role): RayClassWithInitArgs(
                cls=ray.remote(worker_cls),
                config=worker_config,
                role=str(actor_role),
            ),
        }

        return role_cls, actor_role

    def _spawn_worker_groups(self, role_cls: dict) -> dict[str, RayWorkerGroup]:
        """Create GPU resource pool, colocate workers, and spawn per-role groups.

        Uses a single resource pool instead of RayPPOTrainer's ResourcePoolManager.
        """
        config = self.config

        self._resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=3,
            name_prefix="gpu_serve",
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict=role_cls)
        wg_kwargs = {}
        if OmegaConf.select(config, "global_profiler.steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.to_container(
                OmegaConf.select(config, "global_profiler.steps"), resolve=True
            )
            if OmegaConf.select(config, "global_profiler.tool") == "nsys":
                worker_nsight_options = OmegaConf.select(
                    config,
                    "global_profiler.global_tool_config.nsys.worker_nsight_options",
                )
                if worker_nsight_options is None:
                    raise ValueError("worker_nsight_options must be set when using nsys with profile_steps")
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(worker_nsight_options)
        wg = RayWorkerGroup(resource_pool=self._resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
        return wg.spawn(prefix_set=role_cls.keys())

    def _init_worker_groups(self, all_wg: dict, actor_role: Role):
        """Initialize worker groups in the correct order."""
        config = self.config

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # Replace the actor's loss_fn (which init_model just set to
        # ppo_loss at engine_workers.py:579-582) with a wrapper that
        # branches on a TD sentinel ``__loss_mode__``. Runtime hot-swap
        # via ``set_loss_fn`` per-step proved unreliable in this codepath
        # — the actor kept running ppo_loss despite a successful swap
        # call. Setting the wrapper ONCE at init avoids any race between
        # ``set_loss_fn`` and ``forward_backward`` dispatch and any decorator
        # quirks. ``_datums_to_sft_td`` tags its TDs with
        # ``__loss_mode__="sft"``; RL TDs leave it unset and the wrapper
        # falls back to ppo_loss.
        self.actor_rollout_wg.set_loss_fn(make_branching_loss(config))

        # Ref policy always shares actor_rollout_wg:
        # - LoRA: same weights, adapters disabled for ref forward
        # - Full fine-tune: ActorRolloutRef worker holds both actor and ref model state
        if needs_reference_policy(config):
            self.ref_policy_wg = self.actor_rollout_wg

    def _prepare_model_roles(self, required_roles: set[ModelRole], *, reason: str):
        if self._lifecycle is None:
            return

        logger.info("[engine] lifecycle prepare roles=%s reason=%s", sorted(r.value for r in required_roles), reason)
        self._lifecycle.prepare(
            required_roles,
            sleep_role=lambda role: self._sleep_model_role(role, reason=reason),
            wake_role=lambda role: self._wake_model_role(role, reason=reason),
        )

    async def _prepare_model_roles_async(self, required_roles: set[ModelRole], *, reason: str):
        if self._lifecycle is None or not self._lifecycle.enable_offload:
            return

        required = required_roles & self._lifecycle.available_roles
        logger.info("[engine] lifecycle prepare roles=%s reason=%s", sorted(r.value for r in required), reason)

        for role in list(self._lifecycle.awake_roles - required):
            self._sleep_model_role(role, reason=reason)
            self._lifecycle.mark_asleep(role)

        for role in required - self._lifecycle.awake_roles:
            await self._wake_model_role_async(role, reason=reason)
            self._lifecycle.mark_awake(role)

    def _sleep_model_role(self, role: ModelRole, *, reason: str):
        if role == ModelRole.ACTOR:
            self._move_actor("cpu", reason=f"sleep actor {reason}")
        elif role == ModelRole.REF:
            self._move_ref("cpu", reason=f"sleep ref {reason}")
        elif role == ModelRole.ROLLOUT and self.checkpoint_manager is not None:
            logger.info("[engine] sleeping rollout %s", reason)
            self.checkpoint_manager.sleep_replicas()

    def _wake_model_role(self, role: ModelRole, *, reason: str):
        if role == ModelRole.ACTOR:
            self._move_actor("device", reason=f"wake actor {reason}")
        elif role == ModelRole.REF:
            self._move_ref("device", reason=f"wake ref {reason}")
        elif role == ModelRole.ROLLOUT and self.checkpoint_manager is not None:
            logger.info("[engine] waking rollout %s", reason)
            result = self.checkpoint_manager.wake_up_replicas()
            if asyncio.iscoroutine(result):
                self._run_all([result])

    async def _wake_model_role_async(self, role: ModelRole, *, reason: str):
        if role == ModelRole.ROLLOUT and self.checkpoint_manager is not None:
            logger.info("[engine] waking rollout %s", reason)
            result = self.checkpoint_manager.wake_up_replicas()
            if asyncio.iscoroutine(result):
                await result
            return
        self._wake_model_role(role, reason=reason)

    def _move_actor(self, device: str, *, reason: str):
        if self.actor_rollout_wg is None:
            return
        logger.info("[engine] moving actor to %s: %s", device, reason)
        self.actor_rollout_wg.to_actor(device=device, model=True, optimizer=True, grad=True)

    def _move_ref(self, device: str, *, reason: str):
        if self.ref_policy_wg is None or self._ref_in_actor:
            return
        logger.info("[engine] moving ref to %s: %s", device, reason)
        self.ref_policy_wg.to_ref(device=device, model=True, optimizer=False, grad=False)

    def _init_rollout_replicas(self):
        """Initialize vLLM rollout replicas and checkpoint manager.

        Creates replicas directly using the same formula as AgentLoopManager._initialize_llm_servers,
        but without creating AgentLoopManager (which lives on the client side).
        """
        config = self.config
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        rollout_world_size = (
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        world_size = self.actor_rollout_wg.world_size
        num_replicas = world_size // rollout_world_size

        replica_class = get_rollout_replica_class(rollout_config.name)

        self.rollout_replicas = [
            replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        self._run_all([r.init_hybrid(self.actor_rollout_wg) for r in self.rollout_replicas])

        from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

        servers = {r.server_address: r._server_handle for r in self.rollout_replicas}
        load_balancer = ray.remote(GlobalRequestLoadBalancer).remote(
            servers=servers,
            full_determinism=getattr(rollout_config, "full_determinism", False),
        )
        self._server_manager = LLMServerClient(config, load_balancer)

        ckpt_engine_config = omega_conf_to_dataclass(rollout_config.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=ckpt_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.rollout_replicas,
        )

        if not self._enable_offload:
            if self._lifecycle is not None:
                self._lifecycle.mark_awake(ModelRole.ROLLOUT)
            logger.info("[engine] server offload disabled: leaving actor/ref/rollout resident after init")
            return

        # Bring the engine into a deterministic fully-slept post-init state.
        #
        # The bug being avoided: verl's ``rollout.update_weights`` only resumes
        # the ``weights`` tag, leaving vLLM's KV-cache pool unmapped. A
        # subsequent ``sleep(level=2)`` then crashes with
        # ``cumem_allocator.cpp:209 CUDA Error: invalid argument`` because
        # ``cuMemUnmap`` is called on the never-mapped KV-cache. RL flows
        # avoid this naturally — /asample re-maps KV-cache before the next
        # sleep — but SFT clients (cookbook sl_basic, sdft, ...) call
        # /forward_backward as the very first request and hit the unmapped
        # KV-cache case immediately, plus the same trap exists for any
        # client doing two update_weights cycles back to back.
        #
        # Workaround: at init we (1) warm each replica with a one-token
        # generate so both ``weights`` AND ``kv_cache`` are mapped, then
        # (2) drive ``sleep_replicas`` once. Step (2) succeeds because step
        # (1) ensured everything is mapped, and the engine ends init in
        # rollout marked asleep in the lifecycle tracker. Subsequent training
        # calls early-return; the first /asample (or full ``update_weights``)
        # re-wakes vLLM and rebuilds the invariant ``sleep is preceded by
        # full wake``.
        self._warmup_replicas()
        self.checkpoint_manager.sleep_replicas()
        if self._lifecycle is not None:
            self._lifecycle.mark_asleep(ModelRole.ROLLOUT)
        logger.info("[engine] post-init transition: warmup + sleep done")

    def start_rollout(self, *, sync_weights: bool, reason: str) -> bool:
        """Start the rollout replicas if they are not running yet (``server.rollout_start: lazy``).

        Runs the same init as an eager server (actor offloaded, replicas launched,
        warmed and slept), then, with ``sync_weights``, pushes the actor's current
        weights. Returns True when this call started the rollout. A failed start is
        not retried: this and later calls raise ``RolloutStartError``.
        """
        with self._engine_lock:
            started = self._start_rollout_locked(reason=reason)
            if sync_weights:
                self._update_weights_locked()
            return started

    def _start_rollout_locked(self, *, reason: str) -> bool:
        if self._rollout_state is RolloutState.READY:
            return False
        if self._rollout_state is RolloutState.DISABLED:
            raise RuntimeError("The rollout is not available in no_rollout_deployment mode")
        if self._rollout_state is RolloutState.FAILED:
            raise RolloutStartError(self._rollout_start_error)
        logger.info("[engine] starting rollout replicas: %s", reason)
        started_at = time.monotonic()
        try:
            self._prepare_model_roles(set(), reason="before rollout init")
            self._init_rollout_replicas()
        except Exception as exc:
            self._rollout_state = RolloutState.FAILED
            self._rollout_start_error = (
                f"{ROLLOUT_ENGINE_START_FAILED}: the sampling engine failed to start ({type(exc).__name__}: {exc})"
            )
            logger.exception("[engine] rollout start failed")
            raise RolloutStartError(self._rollout_start_error) from exc
        self._rollout_state = RolloutState.READY
        logger.info("[engine] rollout replicas started in %.1fs", time.monotonic() - started_at)
        return True

    def _is_vexact_rollout(self) -> bool:
        """vexact's VeXactServer.generate hard-rejects ``max_tokens`` /
        ``max_new_tokens`` in ``sampling_params`` (see open-vexact
        async_server.py:220) and derives its own bound from
        ``rollout.{response_length, prompt_length}``. Used by the
        generate / warmup paths to strip those keys before dispatch.
        """
        return self.config.actor_rollout_ref.rollout.get("name") == "vexact"

    @staticmethod
    def _sanitize_sampling_params_for_vexact(sampling_params: dict) -> dict:
        """Drop the two keys vexact's assertion rejects. Returns a new dict
        so we never mutate the caller's params (the tinker SDK reuses the
        same dict across multiple sample_async calls)."""
        if not sampling_params:
            return sampling_params
        return {
            k: v
            for k, v in sampling_params.items()
            if k not in ("max_tokens", "max_new_tokens", "include_stop_str_in_output")
        }

    def _warmup_replicas(self):
        """Issue one tiny generate per rollout replica to populate vLLM's
        KV-cache pool before any sleep transition can be requested."""
        n = len(self.rollout_replicas)
        if n == 0:
            return
        warmup_params: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
        }
        if not self._is_vexact_rollout():
            # vLLM honours ``max_tokens=1`` to keep warmup cheap; vexact
            # derives its own response budget from rollout config and
            # raises if we pass ``max_tokens`` here.
            warmup_params["max_tokens"] = 1
        # Concurrent dispatch: GlobalRequestLoadBalancer's least-requests
        # policy fans the N concurrent generates out across the N replicas
        # (each acquire bumps the in-flight count before the next picks).
        self._run_all(
            [
                self._server_manager.generate(
                    request_id=f"engine-warmup-{i}",
                    prompt_ids=[1, 2, 3, 4],
                    sampling_params=warmup_params,
                )
                for i in range(n)
            ]
        )
        logger.info(f"[engine] warmed {n} rollout replica(s) for CuMemAllocator")

    def _run_all(self, tasks: list):
        """Run a list of async tasks synchronously."""

        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())

    @staticmethod
    def _wait_for_nonblocking_result(result):
        """Materialize non-blocking VeRL worker-group results."""
        if isinstance(result, DataProtoFuture):
            return result.get()
        if isinstance(result, ray.ObjectRef):
            return ray.get(result)
        if isinstance(result, list) and result and all(isinstance(item, ray.ObjectRef) for item in result):
            return ray.get(result)
        return result

    # ==================== Profiling ====================

    def _next_profile_step(self) -> int:
        self._profile_step += 1
        return self._profile_step

    def _should_profile_step(self, step: int) -> bool:
        steps = OmegaConf.select(self.config, "global_profiler.steps")
        if steps is None:
            return False
        return step in set(list(steps))

    def _global_profiler_settings(self) -> dict[str, Any]:
        steps = OmegaConf.select(self.config, "global_profiler.steps")
        if steps is not None:
            steps = list(steps)
        return {
            "tool": OmegaConf.select(self.config, "global_profiler.tool"),
            "steps": steps,
            "save_path": OmegaConf.select(self.config, "global_profiler.save_path"),
        }

    def _start_profile(self, step: int) -> None:
        if self.actor_rollout_wg is None:
            logger.info("[profiler] skip start step=%s reason=no actor_rollout_wg", step)
            return
        logger.info("[profiler] starting actor profiler step=%s settings=%s", step, self._global_profiler_settings())
        self.actor_rollout_wg.start_profile(role="actor")
        logger.info("[profiler] start dispatched step=%s", step)

    def _stop_profile(self, step: int) -> None:
        if self.actor_rollout_wg is None:
            logger.info("[profiler] skip stop step=%s reason=no actor_rollout_wg", step)
            return
        logger.info("[profiler] stopping actor profiler step=%s", step)
        self.actor_rollout_wg.stop_profile()
        logger.info("[profiler] stopped actor profiler step=%s", step)

    # ==================== Replica lifecycle ====================

    async def _ensure_replicas_awake_for_generation(self):
        """Wake rollout replicas once before allowing concurrent generation.

        ``generate`` can be called by many /asample requests at the same time.
        Only the slept -> awake transition needs serialization; once vLLM is
        resident, the requests should flow through the async server concurrently.
        """
        if self._no_rollout_deployment:
            return
        if not self._enable_offload:
            return

        async with self._replica_wake_lock:
            await self._prepare_model_roles_async({ModelRole.ROLLOUT}, reason="before generation")

    # ==================== Operations ====================

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Dispatch a single generation request to a rollout replica (least-requests)."""
        if self._rollout_state is RolloutState.FAILED:
            raise RolloutStartError(self._rollout_start_error)
        if self._server_manager is None:
            raise RuntimeError("No rollout replicas available")

        await self._ensure_replicas_awake_for_generation()

        if self._is_vexact_rollout():
            sampling_params = self._sanitize_sampling_params_for_vexact(sampling_params)

        return await self._server_manager.generate(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )

    def compute_log_prob(self, data):
        with self._engine_lock:
            self._prepare_model_roles({ModelRole.ACTOR}, reason="compute_log_prob")
            return self.actor_rollout_wg.compute_log_prob(data)

    def compute_ref_log_prob(self, data):
        with self._engine_lock:
            if self._ref_in_actor:
                self._prepare_model_roles({ModelRole.ACTOR}, reason="compute_ref_log_prob")
                tu.assign_non_tensor(data, no_lora_adapter=True)
                return self.actor_rollout_wg.compute_log_prob(data)
            if self.ref_policy_wg is None:
                raise RuntimeError("Reference policy not initialized (actor_rollout_ref.ref.enable is false)")
            self._prepare_model_roles({ModelRole.REF}, reason="compute_ref_log_prob")
            return self.ref_policy_wg.compute_ref_log_prob(data)

    def forward_backward(self, data):
        with self._engine_lock:
            profile_step = self._next_profile_step()
            should_profile = self._should_profile_step(profile_step)
            logger.info(
                "[profiler] forward_backward step=%s should_profile=%s settings=%s",
                profile_step,
                should_profile,
                self._global_profiler_settings(),
            )
            self._prepare_model_roles({ModelRole.ACTOR}, reason="forward_backward")
            if should_profile:
                self._start_profile(profile_step)
            try:
                result = self.actor_rollout_wg.forward_backward(data)
                return self._wait_for_nonblocking_result(result)
            finally:
                if should_profile:
                    self._stop_profile(profile_step)

    def optim_step(self, optim_step_params: dict[str, Any] | None = None, grad_clip_norm: float | None = None):
        """One optimizer step.

        ``grad_clip_norm`` follows Tinker ``AdamParams``: ``> 0`` clips the
        global grad norm to that value, ``0`` disables clipping, ``None`` keeps
        the server-start ``actor.optim.clip_grad``.
        """
        with self._engine_lock:
            self._prepare_model_roles({ModelRole.ACTOR}, reason="optim_step")
            if grad_clip_norm is not None:
                return self.actor_rollout_wg.optimizer_step_with_grad_clip(
                    optim_step_params=optim_step_params,
                    grad_clip_norm=grad_clip_norm,
                )
            if optim_step_params is None:
                return self.actor_rollout_wg.optimizer_step()
            return self.actor_rollout_wg.optimizer_step(optim_step_params=optim_step_params)

    def update_weights(self):
        """Sleep replicas and sync FSDP weights (which wakes them).

        For the naive backend, checkpoint_manager.update_weights() internally
        does: resume(weights) → push FSDP params → resume(kv_cache), which
        requires replicas to be in the slept state and leaves them awake.
        When server offload is enabled, release actor optimizer memory before
        that sequence while keeping actor parameters and gradients resident.

        Raises RuntimeError in no_rollout_deployment mode (no rollout replicas to sync).
        A lazy rollout that has not started yet is started first.
        """
        with self._engine_lock:
            if self._no_rollout_deployment:
                raise RuntimeError("update_weights is not available in no_rollout_deployment mode")
            self._start_rollout_locked(reason="update_weights")
            self._update_weights_locked()

    def _update_weights_locked(self):
        self._prepare_model_roles({ModelRole.ACTOR}, reason="update_weights")
        if self._enable_offload:
            logger.info("[engine] offloading actor optimizer before rollout weight sync")
            self.actor_rollout_wg.prepare_actor_for_weight_sync()
        self.checkpoint_manager.update_weights()
        if self._enable_offload:
            self._move_actor("cpu", reason="after update_weights")
            if self._lifecycle is not None:
                self._lifecycle.mark_asleep(ModelRole.ACTOR)
                self._lifecycle.mark_awake(ModelRole.ROLLOUT)

    def save_checkpoint(self, local_path: str, global_step: int = 0, max_ckpt_to_keep=None, **kwargs):
        with self._engine_lock:
            self._prepare_model_roles({ModelRole.ACTOR}, reason="save_checkpoint")
            self.actor_rollout_wg.save_checkpoint(
                local_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
            )

    def load_checkpoint(self, checkpoint_path: str, zero_optimizer_grad: bool = False, **kwargs):
        with self._engine_lock:
            self._prepare_model_roles({ModelRole.ACTOR}, reason="load_checkpoint")
            self.actor_rollout_wg.load_checkpoint(checkpoint_path)
            if zero_optimizer_grad:
                # Weights-only load (Tinker create_training_client_from_state): the VeRL
                # checkpoint also restored the optimizer moments, so start a fresh optimizer.
                self.actor_rollout_wg.reset_optimizer_state()
