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

"""Unit tests for ColocatedBackend.

Tests the colocated-deployment sleep/wake lifecycle, no-rollout mode
behavior, async generate delegation, and serialized synchronous engine
operations.
"""

import asyncio
import logging
import threading
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl_tinker.backends._loss import is_ref_in_actor
from verl_tinker.backends.colocated import (
    ROLLOUT_ENGINE_START_FAILED,
    ColocatedBackend,
    NoRolloutWorker,
    RolloutStartError,
    RolloutState,
    TinkerServerActorRolloutRefWorker,
)
from verl_tinker.backends.model_lifecycle import ModelLifecycle, ModelRole

from verl.protocol import DataProtoFuture
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker

_BACKEND_MODULE = "verl_tinker.backends.colocated"


def _actor_config(strategy="fsdp", param_offload=False, optimizer_offload=False):
    if strategy == "veomni":
        return {
            "_target_": "verl.workers.config.actor.VeOmniActorConfig",
            "strategy": "veomni",
            "use_kl_loss": False,
            "rollout_n": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "veomni": {
                "_target_": "verl.workers.config.engine.VeOmniEngineConfig",
                "param_offload": param_offload,
                "optimizer_offload": optimizer_offload,
            },
        }
    return {
        "_target_": "verl.workers.config.actor.FSDPActorConfig",
        "strategy": "fsdp",
        "use_kl_loss": False,
        "rollout_n": 1,
        "ppo_micro_batch_size_per_gpu": 1,
        "fsdp_config": {
            "_target_": "verl.workers.config.engine.FSDPEngineConfig",
            "param_offload": param_offload,
            "optimizer_offload": optimizer_offload,
        },
    }


def _make_config(
    backend="naive",
    no_rollout_deployment=False,
    actor_strategy="fsdp",
    param_offload=False,
    optimizer_offload=False,
):
    """Create a minimal OmegaConf config for ColocatedBackend."""
    cfg = {
        "server": {"no_rollout_deployment": no_rollout_deployment},
        "algorithm": {},
        "actor_rollout_ref": {
            "actor": _actor_config(actor_strategy, param_offload, optimizer_offload),
            "model": {"path": "/fake/model"},
            "rollout": {
                "name": "vllm",
                "checkpoint_engine": {"backend": backend},
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        },
        "trainer": {"n_gpus_per_node": 8, "nnodes": 1},
    }
    return OmegaConf.create(cfg)


@pytest.mark.parametrize(
    "model_overrides",
    [
        {"lora": {"rank": 8}},
        {"lora_rank": 8},
        {"lora_adapter_path": "/fake/adapter"},
    ],
)
def test_is_ref_in_actor_detects_verl_lora_config(model_overrides):
    config = _make_config()
    config.actor_rollout_ref.model.update(model_overrides)

    assert is_ref_in_actor(config) is True


@pytest.mark.parametrize(
    "model_overrides",
    [
        {},
        {"lora": None, "lora_rank": 0},
        {"lora": {"rank": 0}, "lora_adapter_path": None},
    ],
)
def test_is_ref_in_actor_rejects_disabled_lora_config(model_overrides):
    config = _make_config()
    config.actor_rollout_ref.model.update(model_overrides)

    assert is_ref_in_actor(config) is False


def test_init_imports_model_external_libs_before_workers_and_rollout():
    config = _make_config()
    config.actor_rollout_ref.model.external_lib = [
        "vexact.integrations.verl.register",
        "vexact.integrations.verl.fsdp_enable_invariant",
    ]
    config.external_libs = config.actor_rollout_ref.model.external_lib
    call_order = []

    with (
        patch(
            f"{_BACKEND_MODULE}.import_external_libs", side_effect=lambda _: call_order.append("import")
        ) as mock_import,
        patch.object(
            ColocatedBackend, "_build_role_cls", side_effect=lambda: call_order.append("build") or ({}, "actor")
        ),
        patch.object(ColocatedBackend, "_spawn_worker_groups", side_effect=lambda _: call_order.append("spawn") or {}),
        patch.object(ColocatedBackend, "_init_worker_groups", side_effect=lambda *_: call_order.append("init_workers")),
        patch.object(
            ColocatedBackend, "_prepare_model_roles", side_effect=lambda *_args, **_kwargs: call_order.append("offload")
        ),
        patch.object(ColocatedBackend, "_init_rollout_replicas", side_effect=lambda: call_order.append("init_rollout")),
        patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=False),
        patch(f"{_BACKEND_MODULE}.is_ref_in_actor", return_value=False),
    ):
        ColocatedBackend(config)

    mock_import.assert_called_once_with(config.actor_rollout_ref.model.external_lib)
    assert call_order == ["import", "build", "spawn", "init_workers", "offload", "init_rollout"]


def test_lora_exposes_reference_capability_without_reference_config():
    config = _make_config()

    with (
        patch.object(ColocatedBackend, "_build_role_cls", return_value=({}, "actor")),
        patch.object(ColocatedBackend, "_spawn_worker_groups", return_value={}),
        patch.object(ColocatedBackend, "_init_worker_groups"),
        patch.object(ColocatedBackend, "_prepare_model_roles"),
        patch.object(ColocatedBackend, "_init_rollout_replicas"),
        patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=False),
        patch(f"{_BACKEND_MODULE}.is_ref_in_actor", return_value=True),
    ):
        backend = ColocatedBackend(config)

    assert backend.capabilities.use_reference_policy is True


def test_init_rollout_replicas_wraps_load_balancer_as_ray_actor():
    config = _make_config()
    config.actor_rollout_ref.rollout.full_determinism = True
    backend = _make_backend(config)
    backend.actor_rollout_wg.world_size = 1
    backend._enable_offload = False
    replica = SimpleNamespace(
        server_address="http://rollout-0",
        _server_handle=MagicMock(name="server_handle"),
        init_hybrid=MagicMock(return_value="init_hybrid"),
    )
    replica_class = MagicMock(return_value=replica)
    remote_actor_class = MagicMock(name="remote_actor_class")
    load_balancer_handle = remote_actor_class.remote.return_value

    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    with (
        patch(f"{_BACKEND_MODULE}.get_rollout_replica_class", return_value=replica_class),
        patch(f"{_BACKEND_MODULE}.omega_conf_to_dataclass", return_value=MagicMock()),
        patch(f"{_BACKEND_MODULE}.CheckpointEngineManager"),
        patch(f"{_BACKEND_MODULE}.ray.remote", return_value=remote_actor_class) as remote,
        patch.object(backend, "_run_all"),
    ):
        backend._init_rollout_replicas()

    remote.assert_called_once_with(GlobalRequestLoadBalancer)
    remote_actor_class.remote.assert_called_once_with(
        servers={replica.server_address: replica._server_handle},
        full_determinism=True,
    )
    assert backend.server_manager._load_balancer is load_balancer_handle


def _make_backend(config):
    """Create a ColocatedBackend with mocked __init__, pre-populating state."""
    # Bypass __init__ to avoid Ray/GPU calls
    backend = object.__new__(ColocatedBackend)
    backend.config = config
    backend._engine_lock = threading.Lock()
    backend._no_rollout_deployment = config.get("server", {}).get("no_rollout_deployment", False)
    backend.use_kl_loss = config.actor_rollout_ref.actor.get("use_kl_loss", False)
    backend.use_kl_in_reward = config.get("algorithm", {}).get("use_kl_in_reward", False)
    backend.use_reference_policy = False
    backend._ref_in_actor = False
    backend._enable_offload = bool(config.get("server", {}).get("enable_offload", True))
    backend._lifecycle = ModelLifecycle.create(
        enable_offload=backend._enable_offload,
        has_rollout=not backend._no_rollout_deployment,
        has_ref=False,
        actor_awake=True,
    )
    backend.actor_rollout_wg = MagicMock(name="actor_rollout_wg")
    backend.actor_rollout_wg.world_size = 8
    backend.ref_policy_wg = None
    backend.checkpoint_manager = None if backend._no_rollout_deployment else MagicMock(name="checkpoint_manager")
    backend.rollout_replicas = []
    backend._server_manager = None
    backend._resource_pool = None
    backend._replica_wake_lock = asyncio.Lock()
    backend._profile_step = 0
    backend._lazy_rollout = False
    backend._rollout_state = RolloutState.DISABLED if backend._no_rollout_deployment else RolloutState.READY
    backend._rollout_start_error = None
    return backend


def _make_update_actor_td() -> TensorDict:
    input_ids = torch.nested.as_nested_tensor(
        [
            torch.tensor([10, 11, 12, 13, 14]),
            torch.tensor([20, 21, 22, 23, 24]),
        ],
        layout=torch.jagged,
    )
    prompts = torch.nested.as_nested_tensor(
        [
            torch.tensor([10, 11]),
            torch.tensor([20, 21, 22]),
        ],
        layout=torch.jagged,
    )
    responses = torch.nested.as_nested_tensor(
        [
            torch.tensor([12, 13, 14]),
            torch.tensor([23, 24]),
        ],
        layout=torch.jagged,
    )
    td = TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": torch.nested.as_nested_tensor(
                [torch.arange(5), torch.arange(5)],
                layout=torch.jagged,
            ),
            "prompts": prompts,
            "responses": responses,
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "response_mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            "old_log_probs": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor_data(td, "max_response_len", 3)
    tu.assign_non_tensor_data(td, "compute_loss", True)
    tu.assign_non_tensor_data(td, "calculate_entropy", True)
    return td


class _StrictOptimStepWorkerGroup:
    def __init__(self):
        self.optim_step_params = None

    def optimizer_step(self, optim_step_params=None):
        self.optim_step_params = optim_step_params
        return [{"grad_norm": 1.0}]


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_generate_delegates_to_server_manager(self):
        """generate() delegates to LLMServerClient.generate()."""
        backend = _make_backend(_make_config())
        backend._lifecycle.mark_awake(ModelRole.ROLLOUT)

        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock(return_value="result_0")
        backend._server_manager = mock_manager

        result = await backend.generate("req-0", [1, 2], {"temp": 0.7}, image_data=["img"])

        assert result == "result_0"
        mock_manager.generate.assert_awaited_once_with(
            "req-0",
            prompt_ids=[1, 2],
            sampling_params={"temp": 0.7},
            image_data=["img"],
            video_data=None,
        )

    @pytest.mark.asyncio
    async def test_generate_raises_when_no_manager(self):
        """generate() raises RuntimeError when server_manager is None."""
        backend = _make_backend(_make_config())
        backend._server_manager = None

        with pytest.raises(RuntimeError, match="No rollout replicas"):
            await backend.generate("req-0", [1], {})

    @pytest.mark.asyncio
    async def test_generate_strips_max_tokens_for_vexact(self):
        """vexact's VeXactServer.generate asserts ``max_tokens`` is absent;
        the backend must scrub it from the tinker SDK's sampling_params
        when the rollout backend is vexact (verified live on pod
        c69629e66d0808f9 — first call dispatched the unscrubbed dict)."""
        cfg = _make_config()
        cfg.actor_rollout_ref.rollout.name = "vexact"
        backend = _make_backend(cfg)
        backend._lifecycle.mark_awake(ModelRole.ROLLOUT)
        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock(return_value="tok")
        backend._server_manager = mock_manager

        user_params = {
            "max_tokens": 128,
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "include_stop_str_in_output": True,
        }
        await backend.generate("rid", [1, 2, 3], user_params)

        sent = mock_manager.generate.await_args.kwargs["sampling_params"]
        assert "max_tokens" not in sent
        assert "max_new_tokens" not in sent
        assert "include_stop_str_in_output" not in sent
        assert sent == {"temperature": 0.7, "top_p": 0.95}
        # Caller's dict not mutated (SDK reuses it across calls).
        assert "max_tokens" in user_params and "max_new_tokens" in user_params
        assert "include_stop_str_in_output" in user_params

    @pytest.mark.asyncio
    async def test_generate_keeps_max_tokens_for_vllm(self):
        """vllm honours the user's ``max_tokens`` — must NOT be stripped."""
        backend = _make_backend(_make_config())  # name="vllm" by default
        backend._lifecycle.mark_awake(ModelRole.ROLLOUT)
        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock(return_value="tok")
        backend._server_manager = mock_manager

        await backend.generate(
            "rid",
            [1],
            {"max_tokens": 64, "temperature": 0.5, "include_stop_str_in_output": True},
        )
        sent = mock_manager.generate.await_args.kwargs["sampling_params"]
        assert sent == {"max_tokens": 64, "temperature": 0.5, "include_stop_str_in_output": True}

    @pytest.mark.asyncio
    async def test_generate_wakes_slept_replicas_before_sampling(self):
        """A direct /asample after training must not dispatch into slept vLLM."""
        backend = _make_backend(_make_config(param_offload=True))
        call_order = []
        backend.actor_rollout_wg.to_actor.side_effect = lambda **_: call_order.append("offload")
        backend.checkpoint_manager.wake_up_replicas = AsyncMock()
        backend.checkpoint_manager.wake_up_replicas.side_effect = lambda: call_order.append("wake")
        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock(side_effect=lambda *args, **kwargs: call_order.append("generate") or "tok")
        backend._server_manager = mock_manager

        result = await backend.generate("rid", [1], {"temperature": 0.5})

        assert result == "tok"
        backend.actor_rollout_wg.to_actor.assert_called_once_with(
            device="cpu",
            model=True,
            optimizer=True,
            grad=True,
        )
        backend.checkpoint_manager.wake_up_replicas.assert_awaited_once()
        assert backend._lifecycle.awake_roles == {ModelRole.ROLLOUT}
        mock_manager.generate.assert_awaited_once()
        assert call_order == ["offload", "wake", "generate"]

    @pytest.mark.asyncio
    async def test_concurrent_generate_calls_share_one_wake_transition(self):
        """Concurrent samples serialize only the slept -> awake transition."""
        backend = _make_backend(_make_config())
        wake_started = asyncio.Event()
        release_wake = asyncio.Event()

        async def wake_once():
            wake_started.set()
            await release_wake.wait()

        backend.checkpoint_manager.wake_up_replicas = AsyncMock(side_effect=wake_once)
        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock(side_effect=lambda *args, **kwargs: args[0])
        backend._server_manager = mock_manager

        tasks = [asyncio.create_task(backend.generate(f"rid-{i}", [i], {"temperature": 0.5})) for i in range(3)]
        await wake_started.wait()
        assert backend.checkpoint_manager.wake_up_replicas.await_count == 1

        release_wake.set()
        results = await asyncio.gather(*tasks)

        assert results == ["rid-0", "rid-1", "rid-2"]
        assert backend.checkpoint_manager.wake_up_replicas.await_count == 1
        assert mock_manager.generate.await_count == 3


class TestWorkerSpawn:
    def test_spawn_worker_groups_passes_profile_steps_for_torch(self):
        config = _make_config()
        config.global_profiler = {"tool": "torch", "steps": [1, 3]}
        backend = object.__new__(ColocatedBackend)
        backend.config = config

        wg = MagicMock()
        wg.spawn.return_value = {"actor": "wg"}

        with (
            patch(f"{_BACKEND_MODULE}.RayResourcePool") as mock_pool,
            patch(f"{_BACKEND_MODULE}.create_colocated_worker_cls", return_value="worker_dict_cls"),
            patch(f"{_BACKEND_MODULE}.RayWorkerGroup", return_value=wg) as mock_wg_cls,
        ):
            result = backend._spawn_worker_groups({"actor": object()})

        assert result == {"actor": "wg"}
        mock_pool.assert_called_once()
        mock_wg_cls.assert_called_once_with(
            resource_pool=mock_pool.return_value,
            ray_cls_with_init="worker_dict_cls",
            profile_steps=[1, 3],
        )
        wg.spawn.assert_called_once_with(prefix_set=dict.fromkeys(["actor"]).keys())

    def test_spawn_worker_groups_requires_nsys_worker_options(self):
        config = _make_config()
        config.global_profiler = {"tool": "nsys", "steps": [1], "global_tool_config": {"nsys": {}}}
        backend = object.__new__(ColocatedBackend)
        backend.config = config

        with (
            patch(f"{_BACKEND_MODULE}.RayResourcePool"),
            patch(f"{_BACKEND_MODULE}.create_colocated_worker_cls", return_value="worker_dict_cls"),
        ):
            with pytest.raises(ValueError, match="worker_nsight_options"):
                backend._spawn_worker_groups({"actor": object()})


class TestReplicaLifecycle:
    """Test the server-side sleep/wake lifecycle management."""

    def _make_lifecycle_backend(self):
        backend = _make_backend(_make_config())
        backend.checkpoint_manager = MagicMock()
        backend._lifecycle.mark_awake(ModelRole.ROLLOUT)
        backend.ref_policy_wg = backend.actor_rollout_wg
        return backend

    def test_lifecycle_noops_when_server_offload_disabled(self):
        backend = self._make_lifecycle_backend()
        backend._enable_offload = False
        backend._lifecycle.enable_offload = False

        backend._prepare_model_roles({ModelRole.ROLLOUT}, reason="test")

        backend.actor_rollout_wg.to_actor.assert_not_called()
        backend.checkpoint_manager.sleep_replicas.assert_not_called()
        backend.checkpoint_manager.wake_up_replicas.assert_not_called()

    def test_lifecycle_moves_actor_to_cpu_when_rollout_is_required(self):
        backend = self._make_lifecycle_backend()

        backend._prepare_model_roles({ModelRole.ROLLOUT}, reason="test")

        backend.actor_rollout_wg.to_actor.assert_called_once_with(
            device="cpu",
            model=True,
            optimizer=True,
            grad=True,
        )
        assert backend._lifecycle.awake_roles == {ModelRole.ROLLOUT}

    def test_lifecycle_wakes_actor_after_rollout(self):
        backend = self._make_lifecycle_backend()
        backend._prepare_model_roles({ModelRole.ROLLOUT}, reason="rollout")

        backend._prepare_model_roles({ModelRole.ACTOR}, reason="actor")

        backend.checkpoint_manager.sleep_replicas.assert_called_once()
        backend.actor_rollout_wg.to_actor.assert_any_call(device="cpu", model=True, optimizer=True, grad=True)
        backend.actor_rollout_wg.to_actor.assert_any_call(device="device", model=True, optimizer=True, grad=True)
        assert backend._lifecycle.awake_roles == {ModelRole.ACTOR}

    def test_update_weights_releases_optimizer_before_sync_then_marks_rollout_awake(self):
        """Offload mode releases training-only memory before waking rollout weights."""
        backend = self._make_lifecycle_backend()

        call_order = []
        backend.checkpoint_manager.sleep_replicas.side_effect = lambda: call_order.append("sleep")
        backend.actor_rollout_wg.prepare_actor_for_weight_sync.side_effect = lambda: call_order.append(
            "release_optimizer"
        )
        backend.checkpoint_manager.update_weights.side_effect = lambda: call_order.append("update")

        backend.update_weights()

        assert call_order == ["sleep", "release_optimizer", "update"]
        assert backend._lifecycle.awake_roles == {ModelRole.ROLLOUT}

    def test_update_weights_keeps_optimizer_resident_when_server_offload_disabled(self):
        backend = self._make_lifecycle_backend()
        backend._enable_offload = False
        backend._lifecycle.enable_offload = False

        backend.update_weights()

        backend.actor_rollout_wg.prepare_actor_for_weight_sync.assert_not_called()
        backend.actor_rollout_wg.to_actor.assert_not_called()
        backend.checkpoint_manager.update_weights.assert_called_once_with()

    def test_compute_log_prob_sleeps_first(self):
        """compute_log_prob() sleeps replicas before running FSDP forward."""
        backend = self._make_lifecycle_backend()
        assert ModelRole.ROLLOUT in backend._lifecycle.awake_roles

        backend.compute_log_prob("data")

        backend.checkpoint_manager.sleep_replicas.assert_called_once()
        backend.actor_rollout_wg.compute_log_prob.assert_called_once_with("data")
        assert backend._lifecycle.awake_roles == {ModelRole.ACTOR}

    def test_compute_ref_log_prob_sleeps_first(self):
        """compute_ref_log_prob() sleeps replicas before running FSDP forward."""
        backend = self._make_lifecycle_backend()

        backend.compute_ref_log_prob("data")

        backend.checkpoint_manager.sleep_replicas.assert_called_once()

    def test_lora_reference_log_prob_uses_actor_without_adapter(self):
        """LoRA base log-probs reuse actor inference with its adapter disabled."""
        backend = self._make_lifecycle_backend()
        backend._ref_in_actor = True
        backend.ref_policy_wg = None
        data = TensorDict({}, batch_size=[])

        backend.compute_ref_log_prob(data)

        backend.actor_rollout_wg.compute_log_prob.assert_called_once_with(data)
        assert tu.get(data, "no_lora_adapter") is True
        assert backend._lifecycle.awake_roles == {ModelRole.ACTOR}

    def test_full_finetune_reference_log_prob_uses_reference_worker(self):
        """Full-model training retains the separate reference-worker path."""
        backend = self._make_lifecycle_backend()
        backend._ref_in_actor = False

        backend.compute_ref_log_prob("data")

        backend.ref_policy_wg.compute_ref_log_prob.assert_called_once_with("data")
        backend.actor_rollout_wg.compute_log_prob.assert_not_called()

    def test_forward_backward_sleeps_first(self):
        """forward_backward() sleeps replicas before running FSDP backward."""
        backend = self._make_lifecycle_backend()

        backend.forward_backward("data")

        backend.checkpoint_manager.sleep_replicas.assert_called_once()
        backend.actor_rollout_wg.forward_backward.assert_called_once_with("data")

    def test_forward_backward_waits_for_nonblocking_result(self):
        """forward_backward() materializes VeRL's non-blocking DataProtoFuture."""
        backend = self._make_lifecycle_backend()
        future = DataProtoFuture(collect_fn=lambda output: output, futures=[])
        future.get = MagicMock(return_value="result_td")
        backend.actor_rollout_wg.forward_backward.return_value = future

        result = backend.forward_backward("data")

        assert result == "result_td"
        future.get.assert_called_once_with()

    def test_forward_backward_profiles_configured_steps(self):
        """forward_backward() starts/stops verl profiling on configured Tinker training steps."""
        backend = self._make_lifecycle_backend()
        backend.config.global_profiler = {"steps": [1], "tool": "torch"}
        backend.actor_rollout_wg.forward_backward.return_value = "result_td"

        result = backend.forward_backward("data")

        assert result == "result_td"
        backend.actor_rollout_wg.start_profile.assert_called_once_with(role="actor")
        backend.actor_rollout_wg.stop_profile.assert_called_once_with()

    def test_forward_backward_logs_profiler_decision(self, caplog):
        """Profiler decision logs expose the active global profiler settings."""
        backend = self._make_lifecycle_backend()
        backend.config.global_profiler = {"steps": [1], "tool": "torch", "save_path": "outputs/profile"}
        backend.actor_rollout_wg.forward_backward.return_value = "result_td"

        with caplog.at_level(logging.INFO, logger="ray"):
            result = backend.forward_backward("data")

        assert result == "result_td"
        assert "[profiler] forward_backward step=1 should_profile=True" in caplog.text
        assert "'tool': 'torch'" in caplog.text
        assert "'save_path': 'outputs/profile'" in caplog.text
        assert "[profiler] starting actor profiler step=1" in caplog.text
        assert "[profiler] stopped actor profiler step=1" in caplog.text

    def test_forward_backward_skips_unconfigured_profile_steps(self):
        """Only steps listed in global_profiler.steps are profiled."""
        backend = self._make_lifecycle_backend()
        backend.config.global_profiler = {"steps": [2], "tool": "torch"}
        backend.actor_rollout_wg.forward_backward.return_value = "result_td"

        result = backend.forward_backward("data")

        assert result == "result_td"
        backend.actor_rollout_wg.start_profile.assert_not_called()
        backend.actor_rollout_wg.stop_profile.assert_not_called()

    def test_forward_backward_stops_profile_after_worker_error(self):
        """A failed worker call must not leave torch profiler running."""
        backend = self._make_lifecycle_backend()
        backend.config.global_profiler = {"steps": [1], "tool": "torch"}
        backend.actor_rollout_wg.forward_backward.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            backend.forward_backward("data")

        backend.actor_rollout_wg.start_profile.assert_called_once_with(role="actor")
        backend.actor_rollout_wg.stop_profile.assert_called_once_with()

    def test_optim_step_sleeps_first(self):
        """optim_step() sleeps replicas before stepping the optimizer."""
        backend = self._make_lifecycle_backend()

        backend.optim_step()

        backend.checkpoint_manager.sleep_replicas.assert_called_once()
        backend.actor_rollout_wg.optimizer_step.assert_called_once_with()

    def test_optim_step_passes_optim_step_params_keyword(self):
        """optim_step() forwards the per-step params using VeRL's current keyword."""
        backend = self._make_lifecycle_backend()
        backend.actor_rollout_wg = _StrictOptimStepWorkerGroup()
        optim_step_params = {"lr": 2e-5, "betas": (0.8, 0.9), "eps": 1e-7, "weight_decay": 0.01}

        result = backend.optim_step(optim_step_params)

        assert result == [{"grad_norm": 1.0}]
        assert backend.actor_rollout_wg.optim_step_params == optim_step_params

    def test_load_checkpoint_can_zero_optimizer_grad_after_load(self):
        """optimizer=false loads the checkpoint then resets the restored optimizer state."""
        backend = self._make_lifecycle_backend()

        backend.load_checkpoint("/tmp/checkpoint", zero_optimizer_grad=True)

        backend.actor_rollout_wg.load_checkpoint.assert_called_once_with("/tmp/checkpoint")
        backend.actor_rollout_wg.reset_optimizer_state.assert_called_once_with()

    def test_load_checkpoint_keeps_loaded_optimizer_grad_by_default(self):
        """Default load_checkpoint preserves the loaded optimizer state as-is."""
        backend = self._make_lifecycle_backend()

        backend.load_checkpoint("/tmp/checkpoint")

        backend.actor_rollout_wg.load_checkpoint.assert_called_once_with("/tmp/checkpoint")
        backend.actor_rollout_wg.optimizer_zero_grad.assert_not_called()
        backend.actor_rollout_wg.reset_optimizer_state.assert_not_called()

    def test_sleep_is_idempotent(self):
        """Multiple training ops don't re-sleep already-slept replicas."""
        backend = self._make_lifecycle_backend()

        backend.compute_log_prob("data1")
        backend.compute_ref_log_prob("data2")
        backend.forward_backward("data3")
        backend.optim_step()

        # Only one sleep call despite multiple training ops
        backend.checkpoint_manager.sleep_replicas.assert_called_once()

    def test_full_training_loop_lifecycle(self):
        """Simulate a full training step: generate → train → update_weights."""
        backend = self._make_lifecycle_backend()

        call_order = []
        backend.checkpoint_manager.sleep_replicas.side_effect = lambda: call_order.append("sleep")
        backend.checkpoint_manager.update_weights.side_effect = lambda: call_order.append("update_weights")
        backend.actor_rollout_wg.compute_log_prob.side_effect = lambda d: call_order.append("log_prob")
        backend.actor_rollout_wg.forward_backward.side_effect = lambda d: call_order.append("forward_backward")
        backend.actor_rollout_wg.optimizer_step.side_effect = lambda: call_order.append("optim_step")

        # generate() is async, doesn't sleep — skipped here

        # Training phase: first op triggers sleep
        backend.compute_log_prob("d1")
        backend.forward_backward("d2")
        backend.optim_step()

        # Sync weights: already slept, just updates and wakes
        backend.update_weights()

        assert call_order == ["sleep", "log_prob", "forward_backward", "optim_step", "update_weights"]
        assert backend._lifecycle.awake_roles == {ModelRole.ROLLOUT}

        # Next training step: replicas are awake again, first op triggers sleep
        backend.compute_log_prob("d3")
        assert call_order == [
            "sleep",
            "log_prob",
            "forward_backward",
            "optim_step",
            "update_weights",
            "sleep",
            "log_prob",
        ]


def test_forward_backward_does_not_compute_server_side_kl_reference_log_probs():
    cfg = _make_config()
    cfg.algorithm.use_kl_in_reward = True
    backend = _make_backend(cfg)
    backend.ref_policy_wg = MagicMock(name="ref_policy_wg")
    data = _make_update_actor_td()

    backend.forward_backward(data)

    backend.ref_policy_wg.compute_ref_log_prob.assert_not_called()
    backend.actor_rollout_wg.forward_backward.assert_called_once_with(data)


class TestSynchronousEngineLock:
    """Sync operations serialize internally now that ColocatedBackend is the engine."""

    def test_compute_log_prob_delegates_and_releases_lock(self):
        backend = _make_backend(_make_config())
        backend.actor_rollout_wg.compute_log_prob.return_value = "lp"

        result = backend.compute_log_prob("data")

        backend.actor_rollout_wg.compute_log_prob.assert_called_once_with("data")
        assert result == "lp"
        assert backend._engine_lock.acquire(blocking=False)
        backend._engine_lock.release()

    def test_save_checkpoint_delegates_with_kwargs(self):
        backend = _make_backend(_make_config())

        backend.save_checkpoint("/tmp/ckpt", global_step=5, max_ckpt_to_keep=3)

        backend.actor_rollout_wg.save_checkpoint.assert_called_once_with("/tmp/ckpt", global_step=5, max_ckpt_to_keep=3)


class TestBackendOffloadConfig:
    """Tests for backend-owned model lifecycle config resolution."""

    @pytest.mark.parametrize("enable_offload", [True, False])
    def test_init_uses_server_enable_offload(self, enable_offload):
        config = _make_config(
            param_offload=True,
            optimizer_offload=False,
        )
        config.server.enable_offload = enable_offload

        with (
            patch.object(ColocatedBackend, "_build_role_cls", return_value=({}, "actor")),
            patch.object(ColocatedBackend, "_spawn_worker_groups", return_value={}),
            patch.object(ColocatedBackend, "_init_worker_groups"),
            patch.object(ColocatedBackend, "_prepare_model_roles"),
            patch.object(ColocatedBackend, "_init_rollout_replicas"),
            patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=False),
            patch(f"{_BACKEND_MODULE}.is_ref_in_actor", return_value=False),
        ):
            backend = ColocatedBackend(config)

        assert backend._enable_offload is enable_offload
        assert backend._lifecycle.enable_offload is enable_offload


class TestNoRolloutDeployment:
    """Test no_rollout_deployment mode: NoRolloutWorker role transformation and backend behavior."""

    def test_build_rollout_is_noop(self):
        """_build_rollout() is a no-op so vLLM initialization is skipped."""
        with patch.object(TinkerActorRolloutRefWorker, "__init__", return_value=None):
            worker = NoRolloutWorker(MagicMock(), "actor")
        # Should do nothing and return None
        assert worker._build_rollout(trust_remote_code=False) is None

    @pytest.mark.parametrize("op", ["compute_log_prob", "compute_ref_log_prob", "forward_backward"])
    def test_training_ops_work(self, op):
        """Training ops run without sleeping replicas (checkpoint_manager is None)."""
        backend = _make_backend(_make_config(no_rollout_deployment=True))
        backend.ref_policy_wg = backend.actor_rollout_wg
        getattr(backend, op)("data")

    def test_no_rollout_optim_step_works(self):
        """optim_step runs without sleeping replicas (checkpoint_manager is None)."""
        backend = _make_backend(_make_config(no_rollout_deployment=True))
        backend.optim_step()

    def test_update_weights_raises(self):
        """update_weights raises RuntimeError (no rollout replicas to sync)."""
        backend = _make_backend(_make_config(no_rollout_deployment=True))
        with pytest.raises(RuntimeError, match="no_rollout_deployment"):
            backend.update_weights()

    @pytest.mark.asyncio
    async def test_generate_raises(self):
        """generate() raises RuntimeError (no server_manager)."""
        backend = _make_backend(_make_config(no_rollout_deployment=True))
        with pytest.raises(RuntimeError, match="No rollout replicas"):
            await backend.generate("req-0", [1], {})

    def test_build_role_cls_uses_no_rollout_worker_and_actor_role(self):
        """_build_role_cls selects NoRolloutWorker and an actor-only role."""
        backend = _make_backend(_make_config(no_rollout_deployment=True))
        with (
            patch(f"{_BACKEND_MODULE}.ray") as mock_ray,
            patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=False),
        ):
            mock_ray.remote = MagicMock(side_effect=lambda cls: cls)
            role_cls, actor_role = backend._build_role_cls()
        assert str(actor_role) == "actor"
        assert list(role_cls.values())[0].cls is NoRolloutWorker

    def test_build_role_cls_uses_actor_rollout_ref_worker(self):
        """_build_role_cls selects the Tinker actor/rollout worker."""
        backend = _make_backend(_make_config())
        with (
            patch(f"{_BACKEND_MODULE}.ray") as mock_ray,
            patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=False),
        ):
            mock_ray.remote = MagicMock(side_effect=lambda cls: cls)
            role_cls, _ = backend._build_role_cls()
        assert list(role_cls.values())[0].cls is TinkerServerActorRolloutRefWorker

    def test_build_role_cls_strips_server_only_ref_enable_before_verl_worker(self):
        config = _make_config()
        config.actor_rollout_ref.ref = {"enable": True}
        backend = _make_backend(config)

        with (
            patch(f"{_BACKEND_MODULE}.ray") as mock_ray,
            patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=True),
        ):
            mock_ray.remote = MagicMock(side_effect=lambda cls: cls)
            role_cls, actor_role = backend._build_role_cls()

        worker_config = role_cls[str(actor_role)].kwargs["config"]
        assert "enable" not in worker_config.ref
        assert config.actor_rollout_ref.ref.enable is True

    def test_tinker_worker_exposes_optimizer_zero_grad(self):
        """optimizer=false load_weights depends on this registered worker method."""
        assert hasattr(TinkerActorRolloutRefWorker, "optimizer_zero_grad")

    def test_tinker_worker_forwards_logical_device_to_actor_and_ref(self):
        worker = object.__new__(TinkerServerActorRolloutRefWorker)
        worker.actor = MagicMock()
        worker.ref = MagicMock()

        worker.to_actor("device", model=True, optimizer=True, grad=True)
        worker.to_ref("device", model=True, optimizer=False, grad=False)

        worker.actor.to.assert_called_once_with(device="device", model=True, optimizer=True, grad=True)
        worker.ref.to.assert_called_once_with(device="device", model=True, optimizer=False, grad=False)

    def test_tinker_worker_prepares_actor_for_weight_sync(self):
        worker = object.__new__(TinkerServerActorRolloutRefWorker)
        worker.actor = MagicMock()

        with patch(f"{_BACKEND_MODULE}.aggressive_empty_cache") as mock_empty_cache:
            worker.prepare_actor_for_weight_sync()

        worker.actor.optimizer_zero_grad.assert_not_called()
        worker.actor.to.assert_called_once_with(device="cpu", model=False, optimizer=True, grad=False)
        mock_empty_cache.assert_called_once_with(force_sync=True)


# ==================== EnvLoop: per-step grad_clip_norm ====================


class _CpuFsdpLikeEngine:
    """Just enough of VeRL's FSDPEngine to run its real ``optimizer_step`` on CPU."""

    def __init__(self, clip_grad=1.0, lr=0.1):
        from verl.workers.config.optimizer import FSDPOptimizerConfig

        torch.manual_seed(0)
        self.module = torch.nn.Linear(4, 3)
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=lr)
        self.optimizer_config = FSDPOptimizerConfig(lr=lr, clip_grad=clip_grad)
        self._qat_enabled = False

    def set_grads(self, scale):
        for param in self.module.parameters():
            param.grad = torch.full_like(param, scale)

    def grad_norm(self):
        return torch.sqrt(sum((p.grad**2).sum() for p in self.module.parameters())).item()

    def params(self):
        return [p.detach().clone() for p in self.module.parameters()]

    def train_mode(self, zero_grad_on_exit=False, **kwargs):
        engine = self

        class _Ctx:
            def __enter__(self):
                return engine

            def __exit__(self, *exc):
                if zero_grad_on_exit:
                    engine.optimizer.zero_grad()

        return _Ctx()

    def optimizer_step(self):
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

        return FSDPEngine.optimizer_step(self)

    def is_mp_src_rank_with_outputs(self):
        return True


def _grad_clip_worker(engine):
    from verl.workers.engine_workers_tinker import TinkerTrainingWorker

    actor = object.__new__(TinkerTrainingWorker)
    actor.engine = engine
    worker = object.__new__(TinkerServerActorRolloutRefWorker)
    worker.role = "actor"
    worker.actor = actor
    return worker


@pytest.mark.parametrize(
    ("grad_clip_norm", "expected"),
    [(None, None), (0.0, float("inf")), (-1.0, float("inf")), (0.5, 0.5), (3, 3.0)],
)
def test_effective_clip_grad_maps_tinker_semantics(grad_clip_norm, expected):
    from verl_tinker.backends.colocated import effective_clip_grad

    assert effective_clip_grad(grad_clip_norm) == expected


@pytest.mark.parametrize(
    ("grad_clip_norm", "server_clip_grad", "expected_norm_after_clip"),
    [
        (0.5, 1.0, 0.5),  # Tinker clip below the server default
        (10.0, 1.0, None),  # threshold above the norm: untouched
        (0.0, 1.0, None),  # Tinker default 0.0: no clipping although the server config would clip at 1.0
        (None, 1.0, 1.0),  # no AdamParams sent: server-start clip_grad applies
    ],
)
def test_optimizer_step_with_grad_clip_clips_global_norm_and_reports_pre_clip_norm(
    grad_clip_norm, server_clip_grad, expected_norm_after_clip
):
    engine = _CpuFsdpLikeEngine(clip_grad=server_clip_grad, lr=0.1)
    engine.set_grads(1.0)  # global norm = sqrt(15) ~ 3.87
    pre_clip = engine.grad_norm()
    before = engine.params()
    worker = _grad_clip_worker(engine)

    metrics = worker.optimizer_step_with_grad_clip(optim_step_params=None, grad_clip_norm=grad_clip_norm)

    assert metrics["grad_norm"] == pytest.approx(pre_clip)
    applied_scale = 1.0 if expected_norm_after_clip is None else expected_norm_after_clip / pre_clip
    for old, new in zip(before, engine.params(), strict=True):
        torch.testing.assert_close(old - new, torch.full_like(old, 0.1 * applied_scale), rtol=1e-4, atol=1e-6)
    # The per-step override never leaks into later steps.
    assert engine.optimizer_config.clip_grad == server_clip_grad
    # zero_grad_on_exit: gradients are consumed by the step.
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in engine.module.parameters())


def test_optimizer_step_with_grad_clip_restores_config_on_failure():
    engine = _CpuFsdpLikeEngine(clip_grad=1.0)
    engine.set_grads(1.0)
    worker = _grad_clip_worker(engine)
    engine.optimizer_step = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        worker.optimizer_step_with_grad_clip(grad_clip_norm=0.25)

    assert engine.optimizer_config.clip_grad == 1.0


def test_backend_optim_step_routes_grad_clip_norm_to_worker_override():
    backend = _make_backend(_make_config())
    backend.actor_rollout_wg.optimizer_step_with_grad_clip.return_value = [{"grad_norm": 2.0}]

    result = backend.optim_step({"lr": 1e-5}, grad_clip_norm=0.0)

    assert result == [{"grad_norm": 2.0}]
    backend.actor_rollout_wg.optimizer_step_with_grad_clip.assert_called_once_with(
        optim_step_params={"lr": 1e-5}, grad_clip_norm=0.0
    )
    backend.actor_rollout_wg.optimizer_step.assert_not_called()

    backend.optim_step({"lr": 1e-5})
    backend.actor_rollout_wg.optimizer_step.assert_called_once_with(optim_step_params={"lr": 1e-5})


def test_reset_optimizer_state_makes_the_next_adam_step_a_first_step():
    """Weights-only load: restored Adam moments are discarded, parameters are kept."""
    from types import SimpleNamespace

    def make():
        torch.manual_seed(0)
        module = torch.nn.Linear(3, 2)
        return module, torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.0)

    def step(module, optimizer, scale):
        optimizer.zero_grad()
        (module(torch.ones(1, 3)).sum() * scale).backward()
        optimizer.step()

    # Restored path: one step of history, then a second step.
    module, optimizer = make()
    step(module, optimizer, 1.0)
    assert optimizer.state
    snapshot = [p.detach().clone() for p in module.parameters()]
    worker = _grad_clip_worker(SimpleNamespace(optimizer=optimizer))
    worker.reset_optimizer_state()
    assert not optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(snapshot, module.parameters(), strict=True))
    step(module, optimizer, 5.0)
    after_reset = [p.detach().clone() for p in module.parameters()]

    # Reference: a brand-new optimizer from the same weights takes the same step.
    fresh_module, _ = make()
    with torch.no_grad():
        for target, source in zip(fresh_module.parameters(), snapshot, strict=True):
            target.copy_(source)
    fresh_optimizer = torch.optim.AdamW(fresh_module.parameters(), lr=0.1, weight_decay=0.0)
    step(fresh_module, fresh_optimizer, 5.0)
    for a, b in zip(after_reset, fresh_module.parameters(), strict=True):
        torch.testing.assert_close(a, b.detach())


def test_oversized_cpu_weight_is_sent_from_the_device(monkeypatch):
    """FSDP1 gathers CPU tensors; oversized ones must not be CUDA-IPC shared from host memory."""
    from types import SimpleNamespace

    from verl_tinker.backends import colocated

    import verl.utils.device as device_utils

    monkeypatch.setattr(device_utils, "get_device_name", lambda: "cuda")
    monkeypatch.setattr(device_utils, "get_device_id", lambda: 3)
    moves = []

    class FakeTensor:
        def __init__(self, device_type):
            self.device = SimpleNamespace(type=device_type)

        def to(self, device):
            moves.append(device)
            return FakeTensor("cuda")

    on_host = colocated._on_sender_device(FakeTensor("cpu"))
    assert on_host.device.type == "cuda" and moves == ["cuda:3"]
    on_device = FakeTensor("cuda")
    assert colocated._on_sender_device(on_device) is on_device and moves == ["cuda:3"]


def test_bucketed_sender_direct_send_is_patched():
    """With vLLM installed (GPU hosts), VeRL's sender routes oversized weights through the device move."""
    pytest.importorskip("vllm")
    import verl_tinker.backends.colocated  # noqa: F401 - applies the patch at import

    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    assert getattr(BucketedWeightSender._direct_send_large_weight, "_envloop_device_patch", False)


# ==================== Lazy rollout start (EnvLoop) ====================


def _patched_init(stack, call_order):
    for patcher in (
        patch.object(ColocatedBackend, "_build_role_cls", return_value=({}, "actor")),
        patch.object(ColocatedBackend, "_spawn_worker_groups", return_value={}),
        patch.object(ColocatedBackend, "_init_worker_groups"),
        patch.object(
            ColocatedBackend, "_prepare_model_roles", side_effect=lambda *_a, **_k: call_order.append("offload")
        ),
        patch.object(ColocatedBackend, "_init_rollout_replicas", side_effect=lambda: call_order.append("init_rollout")),
        patch(f"{_BACKEND_MODULE}.needs_reference_policy", return_value=False),
        patch(f"{_BACKEND_MODULE}.is_ref_in_actor", return_value=True),
    ):
        stack.enter_context(patcher)


def _lazy_backend(load_format="dummy"):
    config = _make_config()
    config.server.rollout_start = "lazy"
    config.actor_rollout_ref.rollout.load_format = load_format
    backend = _make_backend(config)
    backend._lazy_rollout = True
    backend._rollout_state = RolloutState.NOT_STARTED
    return backend


@pytest.mark.parametrize(("rollout_start", "expected"), [("lazy", []), ("eager", ["offload", "init_rollout"])])
def test_lazy_server_init_does_not_start_the_rollout(rollout_start, expected):
    config = _make_config()
    config.server.rollout_start = rollout_start
    call_order = []

    with ExitStack() as stack:
        _patched_init(stack, call_order)
        backend = ColocatedBackend(config)

    assert call_order == expected
    assert backend.rollout_state is (RolloutState.NOT_STARTED if rollout_start == "lazy" else RolloutState.READY)


def test_start_rollout_runs_the_eager_init_once():
    backend = _lazy_backend()
    call_order = []

    with (
        patch.object(
            backend, "_prepare_model_roles", side_effect=lambda roles, **_: call_order.append(("roles", roles))
        ),
        patch.object(backend, "_init_rollout_replicas", side_effect=lambda: call_order.append("init_rollout")),
    ):
        assert backend.start_rollout(sync_weights=False, reason="test") is True
        assert backend.start_rollout(sync_weights=False, reason="test") is False

    # Same order as an eager server: offload every role, then launch, warm and sleep vLLM.
    assert call_order == [("roles", set()), "init_rollout"]
    assert backend.rollout_state is RolloutState.READY
    backend.checkpoint_manager.update_weights.assert_not_called()


def test_start_rollout_with_sync_pushes_the_actor_weights_after_init():
    backend = _lazy_backend()
    call_order = []
    backend.checkpoint_manager.update_weights.side_effect = lambda: call_order.append("sync")

    with (
        patch.object(backend, "_prepare_model_roles"),
        patch.object(backend, "_init_rollout_replicas", side_effect=lambda: call_order.append("init_rollout")),
    ):
        backend.start_rollout(sync_weights=True, reason="test")

    assert call_order == ["init_rollout", "sync"]
    backend.actor_rollout_wg.prepare_actor_for_weight_sync.assert_called_once()
    assert backend._lifecycle.awake_roles == {ModelRole.ROLLOUT}


def test_update_weights_starts_a_lazy_rollout_first():
    backend = _lazy_backend()
    call_order = []
    backend.checkpoint_manager.update_weights.side_effect = lambda: call_order.append("sync")

    with (
        patch.object(backend, "_prepare_model_roles"),
        patch.object(backend, "_init_rollout_replicas", side_effect=lambda: call_order.append("init_rollout")),
    ):
        backend.update_weights()
        backend.update_weights()

    assert call_order == ["init_rollout", "sync", "sync"]


@pytest.mark.asyncio
async def test_failed_rollout_start_is_typed_and_never_retried():
    backend = _lazy_backend()
    init = MagicMock(side_effect=RuntimeError("EngineCore failed to start."))

    with patch.object(backend, "_prepare_model_roles"), patch.object(backend, "_init_rollout_replicas", init):
        with pytest.raises(RolloutStartError, match=ROLLOUT_ENGINE_START_FAILED) as first:
            backend.start_rollout(sync_weights=True, reason="test")
        with pytest.raises(RolloutStartError, match="EngineCore failed to start"):
            backend.update_weights()
        with pytest.raises(RolloutStartError, match=ROLLOUT_ENGINE_START_FAILED):
            await backend.generate("rid", [1, 2], {})

    assert first.value.code == ROLLOUT_ENGINE_START_FAILED
    assert init.call_count == 1
    assert backend.rollout_state is RolloutState.FAILED
    assert "EngineCore failed to start" in backend.rollout_start_error
    backend.checkpoint_manager.update_weights.assert_not_called()


def test_training_never_starts_a_lazy_rollout():
    backend = _lazy_backend()
    backend.actor_rollout_wg.forward_backward.return_value = "fb"

    with patch.object(backend, "_init_rollout_replicas") as init:
        backend.forward_backward(MagicMock())
        backend.optim_step()
        backend.save_checkpoint("/tmp/ckpt")

    init.assert_not_called()
    assert backend.rollout_state is RolloutState.NOT_STARTED


@pytest.mark.parametrize(("load_format", "expected"), [("dummy", False), ("safetensors", True), ("auto", True)])
def test_rollout_loads_base_weights_unless_dummy(load_format, expected):
    assert _lazy_backend(load_format).rollout_loads_base_weights is expected
