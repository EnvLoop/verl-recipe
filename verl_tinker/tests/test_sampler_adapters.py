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

"""Sampler checkpoints served as extra LoRA adapters beside the training weights (EnvLoop)."""

import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf
from verl_tinker.backends.colocated import ColocatedBackend, RolloutState
from verl_tinker.model_resource_manager import ModelResourceManager, SamplingResource
from verl_tinker.sampler_adapters import (
    REQUEST_LORA,
    LoRAOverrideEngine,
    SamplerAdapterError,
    SamplerAdapterIds,
    SamplerAdapterIncompatible,
    adapter_dir,
    adapter_tag,
    check_adapter_compatible,
    multi_lora_engine_args,
)
from verl_tinker.tinker_ops import sample as tinker_sample
from verl_tinker.tinker_router import FifoReadWriteGate, RequestStatus, ServerStatus, TinkerServer


def _router_class():
    for cls in TinkerServer.func_or_class.__mro__:
        if cls.__module__ == "verl_tinker.tinker_router" and cls.__name__ == "TinkerServer":
            return cls
    raise AssertionError("Original TinkerServer class not found under Ray Serve wrapper")


def _init_future_tracking(server):
    server._futures = {}
    server._pending = {}
    server._errors = {}
    server._request_status = {}
    server._retrieved_request_status_archive = {}


def _write_adapter(directory, *, rank=8, modules=("q_proj", "v_proj")):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": 32, "target_modules": list(modules)})
    )
    (directory / "adapter_model.safetensors").write_bytes(b"tensors")
    return directory


# ==================== Compatibility and paths ====================


def test_compatibility_rule():
    config = {"peft_type": "LORA", "r": 24, "target_modules": ["q_proj", "o_proj"]}
    check_adapter_compatible(config, max_lora_rank=32, unembed_enabled=False)
    check_adapter_compatible({**config, "target_modules": ["gate_proj"]}, max_lora_rank=32, unembed_enabled=False)
    with pytest.raises(SamplerAdapterIncompatible, match="rank 64"):
        check_adapter_compatible({**config, "r": 64}, max_lora_rank=32, unembed_enabled=False)
    with pytest.raises(SamplerAdapterIncompatible, match="lm_head"):
        check_adapter_compatible(
            {**config, "target_modules": ["q_proj", "lm_head"]}, max_lora_rank=32, unembed_enabled=False
        )
    check_adapter_compatible(
        {**config, "target_modules": ["q_proj", "lm_head"]}, max_lora_rank=32, unembed_enabled=True
    )
    with pytest.raises(SamplerAdapterIncompatible):
        check_adapter_compatible({**config, "target_modules": "all-linear"}, max_lora_rank=32, unembed_enabled=False)
    split = {**config, "target_modules": ["in_proj_q", "in_proj_k", "in_proj_v", "out_proj"]}
    with pytest.raises(SamplerAdapterIncompatible, match="q/k/v"):
        check_adapter_compatible(split, max_lora_rank=32, unembed_enabled=False)
    check_adapter_compatible(split, max_lora_rank=32, unembed_enabled=False, split_qkv_enabled=True)


def test_adapter_paths_stay_inside_the_checkpoint_root(tmp_path):
    assert adapter_tag("tinker://verl-tinker/weights/x") is None
    assert adapter_tag("sampler-adapter://adapter-abc") == "adapter-abc"
    for bad in (
        "sampler-adapter://../x",
        "sampler-adapter://a/b",
        "sampler-adapter://..",
        "sampler-adapter://",
    ):
        with pytest.raises(SamplerAdapterError):
            adapter_tag(bad)
    assert adapter_dir(tmp_path, "adapter-abc") == (tmp_path / "adapter-abc" / "lora_adapter").resolve()


def test_engine_args_keep_the_configured_slots_and_rank():
    args = Namespace(max_loras=1, max_cpu_loras=None, max_lora_rank=16, enable_lora=True)
    multi_lora_engine_args(args, {"max_loras": 3, "max_cpu_loras": 3, "max_lora_rank": 40})
    assert (args.max_loras, args.max_cpu_loras, args.max_lora_rank) == (3, 3, 64)
    args = Namespace(max_loras=1, max_cpu_loras=2, max_lora_rank=64, enable_lora=True)
    multi_lora_engine_args(args, {"max_loras": 4, "max_lora_rank": 8})
    assert (args.max_loras, args.max_cpu_loras, args.max_lora_rank) == (4, 4, 64)


@pytest.mark.asyncio
async def test_engine_override_applies_only_to_the_requests_task():
    engine = MagicMock()
    proxy = LoRAOverrideEngine(engine)
    current = object()

    async def generate(adapter):
        token = REQUEST_LORA.set(adapter) if adapter is not None else None
        try:
            await asyncio.sleep(0)
            proxy.generate(prompt="p", lora_request=current)
            return engine.generate.call_args.kwargs["lora_request"]
        finally:
            if token is not None:
                REQUEST_LORA.reset(token)

    foreign = object()
    assert await asyncio.create_task(generate(foreign)) is foreign
    assert await asyncio.create_task(generate(None)) is current
    proxy.list_loras()
    engine.list_loras.assert_called_once()


# ==================== Adapter naming (vLLM owns the LRU) ====================


def test_adapter_ids_are_stable_per_directory_and_never_the_current_weights():
    ids = SamplerAdapterIds()
    a, b = ids.spec("/r/a"), ids.spec("/r/b")
    assert ids.spec("/r/a") == a and a["lora_int_id"] != b["lora_int_id"]
    assert min(a["lora_int_id"], b["lora_int_id"]) >= 1000  # VeRL's current weights are 123
    assert a == {"lora_name": f"sampler-{a['lora_int_id']}", "lora_int_id": a["lora_int_id"], "lora_path": "/r/a"}


# ==================== Rollout sampling with an adapter ====================


@pytest.mark.asyncio
async def test_sample_decodes_and_scores_with_the_adapter_never_the_actor():
    calls = []

    class Engine:
        config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"max_model_len": 64}}})

        async def generate(self, request_id, prompt_ids, sampling_params, lora=None):
            calls.append({"lora": lora, "params": dict(sampling_params)})
            k = sampling_params.get("prompt_logprobs")
            prompt_lp = None
            extras = {}
            if k == 0 and "prompt_logprobs" in sampling_params:  # scalar pass: the actual token only
                extras = {"prompt_logprobs": [[-0.1 * t] for t in prompt_ids[1:]]}
            elif k:
                extras = {
                    "prompt_ids": [[t, 99] for t in prompt_ids[1:]],
                    "prompt_logprobs": [[-0.1 * t, -5.0] for t in prompt_ids[1:]],
                }
            return SimpleNamespace(
                token_ids=[5],
                log_probs=[-0.25],
                stop_reason="length",
                extra_fields={**extras, "prompt_logprobs": extras.get("prompt_logprobs", prompt_lp)},
            )

        def compute_log_prob(self, data):
            raise AssertionError("an adapter sample never reads the actor")

    lora = {"lora_name": "sampler-1000", "lora_int_id": 1000, "lora_path": "/r/a"}
    req = SimpleNamespace(
        prompt=SimpleNamespace(to_ints=lambda: [1, 2, 3]),
        sampling_params=SimpleNamespace(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, stop=None, seed=None),
        num_samples=1,
        prompt_logprobs=True,
        topk_prompt_logprobs=2,
        topk_sample_logprobs=1,
    )
    result = await tinker_sample(Engine(), req, lora=lora)
    # decode, top-k sample scoring and the scalar prompt-logprob pass all use the adapter
    assert len(calls) == 3 and all(call["lora"] == lora for call in calls)
    assert result["prompt_logprobs"] == [None, pytest.approx(-0.2), pytest.approx(-0.3)]
    assert result["sequences"][0]["tokens"] == [5]


# ==================== Router binding and late resolution ====================


def _server(tmp_path, *, slots=2, max_rank=32):
    server = object.__new__(_router_class())
    server._shutdown_started = False
    server._status = ServerStatus.INITIALIZED
    server._teacher_backend = None
    server._op_gate = FifoReadWriteGate()
    server._checkpoint_root = str(tmp_path)
    server._saved_state_paths = {}
    _init_future_tracking(server)
    server._model_resource_manager = ModelResourceManager(actor_model_identifiers=("actor",))
    engine = MagicMock()
    engine.sampler_max_lora_rank = max_rank
    engine.sampler_unembed_enabled = False
    engine.sampler_split_qkv_enabled = False
    server._engine = engine
    server._sampler_adapters = SamplerAdapterIds() if slots else None
    return server


def _sample_req(sampler_id):
    return SimpleNamespace(
        sampling_session_id=sampler_id,
        base_model=None,
        model_path=None,
        prompt=SimpleNamespace(to_ints=lambda: [1, 2]),
        sampling_params=SimpleNamespace(max_tokens=4, temperature=1, top_p=1, top_k=-1, stop=None, seed=None),
        num_samples=1,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
        topk_sample_logprobs=0,
        target_prompt_logprobs=None,
    )


async def _run(server, response):
    await asyncio.gather(*server._pending.values())
    request_id = response["request_id"]
    return server._request_status[request_id], server._futures.get(request_id), server._errors.get(request_id)


@pytest.mark.asyncio
async def test_staged_adapter_is_bound_and_sampled_without_touching_training_weights(tmp_path, monkeypatch):
    server = _server(tmp_path)
    _write_adapter(tmp_path / "adapter-abc" / "lora_adapter")
    manager = server._model_resource_manager
    manager.actor_updated()
    manager.rollout_synchronized()
    before = (manager.actor_id, manager.rollout_id)
    session_id = manager.create_session()
    created = await server.create_sampling_session(
        SimpleNamespace(
            session_id=session_id,
            sampling_session_seq_id=0,
            base_model=None,
            model_path="sampler-adapter://adapter-abc",
        )
    )
    binding = manager.get_sampler(created.sampling_session_id)
    assert (
        binding.is_adapter
        and manager.resolve_resource(binding, scalar_prompt_logprobs=True) is SamplingResource.ADAPTER
    )
    assert created.sampling_session_id in manager.valid_sampler_ids_for_session(session_id)

    sampled = []

    async def fake_sample(engine, req, *, lora=None):
        sampled.append(lora)
        return {"type": "sample", "sequences": []}

    monkeypatch.setitem(server.asample.__globals__, "tinker_sample", fake_sample)
    for _ in range(2):
        status, _result, error = await _run(server, await server.asample(_sample_req(created.sampling_session_id)))
        assert status is RequestStatus.COMPLETED, error
    assert sampled[0] == sampled[1] and sampled[0]["lora_path"] == str(tmp_path / "adapter-abc" / "lora_adapter")
    # Training weights and the rollout's current-weights version are untouched.
    assert (manager.actor_id, manager.rollout_id) == before
    server._engine.update_weights.assert_not_called()
    server._engine.load_checkpoint.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files,code",
    [
        ({}, "SAMPLER_ADAPTER_NOT_FOUND"),
        ({"rank": 64}, "SAMPLER_ADAPTER_INCOMPATIBLE"),
        ({"modules": ("q_proj", "lm_head")}, "SAMPLER_ADAPTER_INCOMPATIBLE"),
    ],
)
async def test_unservable_adapter_is_rejected_at_binding(tmp_path, files, code):
    server = _server(tmp_path)
    if files:
        _write_adapter(tmp_path / "adapter-abc" / "lora_adapter", **files)
    session_id = server._model_resource_manager.create_session()
    with pytest.raises(HTTPException) as rejected:
        await server.create_sampling_session(
            SimpleNamespace(
                session_id=session_id,
                sampling_session_seq_id=0,
                base_model=None,
                model_path="sampler-adapter://adapter-abc",
            )
        )
    assert rejected.value.detail["code"] == code
    assert server._model_resource_manager.sampler_ids() == []


@pytest.mark.asyncio
async def test_server_without_spare_slots_rejects_adapters(tmp_path):
    server = _server(tmp_path, slots=0)
    _write_adapter(tmp_path / "adapter-abc" / "lora_adapter")
    session_id = server._model_resource_manager.create_session()
    with pytest.raises(HTTPException) as rejected:
        await server.create_sampling_session(
            SimpleNamespace(
                session_id=session_id,
                sampling_session_seq_id=0,
                base_model=None,
                model_path="sampler-adapter://adapter-abc",
            )
        )
    assert rejected.value.status_code == 422


@pytest.mark.asyncio
async def test_older_sampler_of_this_session_falls_back_to_its_saved_adapter(tmp_path, monkeypatch):
    server = _server(tmp_path)
    manager = server._model_resource_manager
    # Named sampler save: training state then rollout sync at the same actor version.
    manager.actor_updated()
    manager.state_saved("tinker://verl-tinker/state/sampler-a-1")
    manager.rollout_synchronized()
    manager.sampler_path_saved("tinker://verl-tinker/weights/step-a")
    session_id = manager.create_session()
    created = await server.create_sampling_session(
        SimpleNamespace(
            session_id=session_id,
            sampling_session_seq_id=0,
            base_model=None,
            model_path="tinker://verl-tinker/weights/step-a",
        )
    )
    # Training continues; the rollout now holds newer weights.
    manager.actor_updated()
    manager.rollout_synchronized()

    sampled = []

    async def fake_sample(engine, req, *, lora=None):
        sampled.append(lora)
        return {"type": "sample", "sequences": []}

    monkeypatch.setitem(server.asample.__globals__, "tinker_sample", fake_sample)
    status, _result, error = await _run(server, await server.asample(_sample_req(created.sampling_session_id)))
    assert status is RequestStatus.ERROR and "409" in error  # no PEFT export yet: unchanged behaviour

    _write_adapter(tmp_path / "sampler-a-1" / "lora_adapter")  # the guest exports it before publishing
    status, _result, error = await _run(server, await server.asample(_sample_req(created.sampling_session_id)))
    assert status is RequestStatus.COMPLETED, error
    assert sampled == [
        {"lora_name": "sampler-1000", "lora_int_id": 1000, "lora_path": str(tmp_path / "sampler-a-1" / "lora_adapter")}
    ]


# ==================== Backend configuration ====================


def _backend(**rollout):
    backend = object.__new__(ColocatedBackend)
    backend._no_rollout_deployment = False
    backend._rollout_state = RolloutState.READY
    backend.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora_rank": 24, "target_modules": "all-linear"},
                "rollout": {"name": "vllm", **rollout},
            }
        }
    )
    return backend


def test_backend_reports_spare_slots_from_vllm_engine_kwargs(monkeypatch):
    assert _backend().sampler_adapter_slots == 0
    backend = _backend(engine_kwargs={"vllm": {"max_loras": 3}})
    assert (backend.sampler_adapter_slots, backend.sampler_max_lora_rank, backend.sampler_unembed_enabled) == (
        2,
        32,
        False,
    )
    assert _backend(engine_kwargs={"vllm": {"max_loras": 3, "max_lora_rank": 64}}).sampler_max_lora_rank == 64
    full = _backend(engine_kwargs={"vllm": {"max_loras": 3}})
    full.config.actor_rollout_ref.model.lora_rank = 0
    assert full.sampler_adapter_slots == 0
    # lm_head adapters follow the recipe's reported capability, not the session's target modules.
    unembed = _backend(engine_kwargs={"vllm": {"max_loras": 3}})
    unembed.config.actor_rollout_ref.model.target_modules = "(?:.*\\.)?(?:q_proj|lm_head)"
    import sys
    import types

    monkeypatch.setitem(sys.modules, "verl_tinker.capabilities", types.ModuleType("verl_tinker.capabilities"))
    assert unembed.sampler_unembed_enabled is False  # a recipe without the capability report
    module = types.ModuleType("verl_tinker.capabilities")
    module.runtime_capabilities = lambda: {"lm_head_lora_prompt_logprobs": True}
    monkeypatch.setitem(sys.modules, "verl_tinker.capabilities", module)
    assert unembed.sampler_unembed_enabled is True
    module.runtime_capabilities = lambda: {"lm_head_lora_prompt_logprobs": False}
    assert unembed.sampler_unembed_enabled is False
    assert unembed.sampler_split_qkv_enabled is False
    module.runtime_capabilities = lambda: {"lora_split_qkv": True}
    assert unembed.sampler_split_qkv_enabled is True


@pytest.mark.asyncio
async def test_backend_generate_names_the_adapter_only_when_asked():
    backend = _backend()
    backend._enable_offload = False
    backend._server_manager = MagicMock()
    backend._server_manager.generate = MagicMock(side_effect=lambda *a, **k: asyncio.sleep(0, result=k))
    backend.config.actor_rollout_ref.rollout.name = "vllm"
    plain = await backend.generate("r1", prompt_ids=[1], sampling_params={})
    assert "envloop_lora" not in plain
    lora = {"lora_name": "sampler-1000", "lora_int_id": 1000, "lora_path": "/r/a"}
    assert (await backend.generate("r2", prompt_ids=[1], sampling_params={}, lora=lora))["envloop_lora"] == lora


def test_backend_pins_the_current_weights_only_with_spare_slots(monkeypatch):
    import verl_tinker.backends.colocated as colocated

    pinned = []
    replica = SimpleNamespace(
        _server_handle=SimpleNamespace(pin_current_weights=SimpleNamespace(remote=lambda: pinned.append(1) or "ref"))
    )
    monkeypatch.setattr(colocated.ray, "get", lambda refs: refs)
    backend = _backend(engine_kwargs={"vllm": {"max_loras": 3}})
    backend.rollout_replicas = [replica]
    backend._pin_current_weights()
    assert pinned == [1]
    plain = _backend()
    plain.rollout_replicas = [replica]
    plain._pin_current_weights()
    assert pinned == [1]


def test_tensor_parallel_rollout_keeps_the_sampler_adapter_slots():
    backend = _backend(tensor_model_parallel_size=2, engine_kwargs={"vllm": {"max_loras": 3, "max_cpu_loras": 3}})
    vllm_kwargs = backend.config.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert (vllm_kwargs.max_loras, vllm_kwargs.max_cpu_loras, backend.sampler_adapter_slots) == (3, 3, 2)


@pytest.mark.asyncio
async def test_adapter_sample_on_a_lazy_head_starts_the_rollout_first(tmp_path, monkeypatch):
    server = _server(tmp_path)
    server._rollout_start_lock = asyncio.Lock()
    _write_adapter(tmp_path / "adapter-abc" / "lora_adapter")
    manager = server._model_resource_manager
    manager.configure_resources(
        rollout_enabled=True, reference_enabled=False, rollout_started=False, rollout_loads_base_weights=False
    )
    order = []
    server._engine.start_rollout = lambda **kwargs: order.append(("start", kwargs["sync_weights"]))
    server._engine.rollout_state = RolloutState.READY
    session_id = manager.create_session()
    created = await server.create_sampling_session(
        SimpleNamespace(
            session_id=session_id,
            sampling_session_seq_id=0,
            base_model=None,
            model_path="sampler-adapter://adapter-abc",
        )
    )

    async def fake_sample(engine, req, *, lora=None):
        order.append(("sample", lora["lora_path"]))
        return {"type": "sample", "sequences": []}

    monkeypatch.setitem(server.asample.__globals__, "tinker_sample", fake_sample)
    status, _result, error = await _run(server, await server.asample(_sample_req(created.sampling_session_id)))
    assert status is RequestStatus.COMPLETED, error
    # dummy-loaded LoRA rollout: started by pushing the current weights, then the adapter decodes
    assert order == [("start", True), ("sample", str(tmp_path / "adapter-abc" / "lora_adapter"))]
    assert not manager.rollout_pending
