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

"""EnvLoop Tinker-alignment tests for the HTTP router (CPU only).

Synchronous request validation (4xx before any GPU work), LoRA metadata and
sampling-field handling. Imports the Ray Serve router, which (like upstream)
requires the server's pinned ``tinker`` types (0.19).
"""

import asyncio
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from test_envloop_tinker_alignment import _sample_req, _wire_rl_datum
from verl_tinker.lora_info import configured_lora_info, train_flags_for_target_modules
from verl_tinker.model_resource_manager import ModelResourceManager
from verl_tinker.schemas import TinkerForwardBackwardRequest, TinkerForwardRequest
from verl_tinker.tinker_router import FifoReadWriteGate, ServerStatus, TinkerServer


def _router_class():
    for cls in TinkerServer.func_or_class.__mro__:
        if cls.__module__ == "verl_tinker.tinker_router" and cls.__name__ == "TinkerServer":
            return cls
    raise AssertionError("Original TinkerServer class not found under Ray Serve wrapper")


def _server(config=None, engine=None):
    server = object.__new__(_router_class())
    server._status = ServerStatus.INITIALIZED
    server._shutdown_started = False
    server._futures, server._pending, server._errors = {}, {}, {}
    server._request_status, server._retrieved_request_status_archive = {}, {}
    server._op_gate = FifoReadWriteGate()
    server._model_to_base_model, server._model_metadata = {}, {}
    server._teacher_backend = None
    server._model_resource_manager = ModelResourceManager(actor_model_identifiers=("actor",))
    if config is not None:
        server.config = OmegaConf.create(config)
    server._engine = engine if engine is not None else MagicMock()
    return server


def _lora_config(rank=8, target_modules="all-linear"):
    return {
        "server": {"model_name": "Qwen/Qwen3-8B"},
        "actor_rollout_ref": {
            "model": {"path": "/m", "lora_rank": rank, "lora_alpha": 16, "target_modules": target_modules}
        },
    }


@pytest.mark.asyncio
async def test_router_rejects_invalid_loss_requests_synchronously_with_422():
    server = _server()
    schedule = MagicMock()
    server._schedule = schedule
    bad = {
        "forward_backward_input": {
            "data": [
                {
                    **_wire_rl_datum(),
                    "loss_fn_inputs": {
                        **_wire_rl_datum()["loss_fn_inputs"],
                        "weights": {"data": [1.0] * 4, "dtype": "float32", "shape": [4]},
                    },
                }
            ],
            "loss_fn": "ppo",
        },
        "model_id": "m",
    }
    with pytest.raises(HTTPException) as exc_info:
        await server.forward_backward(TypeAdapter(TinkerForwardBackwardRequest).validate_python(bad))
    assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "weights" in exc_info.value.detail
    schedule.assert_not_called()

    good = {
        "forward_backward_input": {"data": [_wire_rl_datum(with_ratio_mask=True)], "loss_fn": "gspo"},
        "model_id": "m",
    }
    response = await server.forward_backward(TypeAdapter(TinkerForwardBackwardRequest).validate_python(good))
    assert schedule.call_count == 1
    assert response["model_id"] == "m"
    schedule.call_args.args[1].close()


@pytest.mark.asyncio
async def test_router_forward_passes_loss_fn_and_rejects_multi_target(monkeypatch):
    server = _server()
    captured = {}

    async def fake_forward(engine, data, loss_fn, loss_fn_config):
        captured.update(loss_fn=loss_fn, loss_fn_config=loss_fn_config, n=len(data))
        return {"type": "forward_backward"}

    monkeypatch.setitem(server.forward.__globals__, "tinker_forward", fake_forward)
    body = {"forward_input": {"data": [_wire_rl_datum()], "loss_fn": "importance_sampling"}, "model_id": "m"}
    await server.forward(TypeAdapter(TinkerForwardRequest).validate_python(body))
    await asyncio.gather(*server._pending.values())
    assert captured == {"loss_fn": "importance_sampling", "loss_fn_config": None, "n": 1}

    if True:
        multi = {
            "forward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                        "loss_fn_inputs": {"target_tokens": {"data": [2, 3, 4, 5], "dtype": "int64", "shape": [2, 2]}},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            "model_id": "m",
        }
        with pytest.raises(HTTPException) as exc_info:
            await server.forward(TypeAdapter(TinkerForwardRequest).validate_python(multi))
        assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_create_model_rejects_rank_mismatch_with_409_and_records_real_lora_metadata():
    server = _server(_lora_config(rank=8))
    session_id = server._model_resource_manager.create_session()

    with pytest.raises(HTTPException) as exc_info:
        await server.create_model(
            SimpleNamespace(
                session_id=session_id, model_seq_id=0, base_model="Qwen/Qwen3-8B", lora_config=SimpleNamespace(rank=32)
            )
        )
    assert exc_info.value.status_code == HTTPStatus.CONFLICT
    assert "rank 32" in exc_info.value.detail and "rank 8" in exc_info.value.detail
    assert server._model_resource_manager.model_ids_for_session(session_id) == []

    response = await server.create_model(
        SimpleNamespace(
            session_id=session_id,
            model_seq_id=0,
            base_model="Qwen/Qwen3-8B",
            lora_config=SimpleNamespace(rank=8, train_unembed=True, train_mlp=True, train_attn=True),
        )
    )
    assert server._model_metadata[response["model_id"]] == {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": True,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "train_unembed": False,
        "train_mlp": True,
        "train_attn": True,
    }


@pytest.mark.asyncio
async def test_create_model_rejects_dimuon_optimizer():
    server = _server(_lora_config(rank=8))
    session_id = server._model_resource_manager.create_session()
    with pytest.raises(HTTPException) as exc_info:
        await server.create_model(
            SimpleNamespace(
                session_id=session_id,
                model_seq_id=0,
                base_model="m",
                lora_config=None,
                optimizer_config={"type": "dimuon", "version": 1},
            )
        )
    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.detail["code"] == "OPTIMIZER_UNSUPPORTED"


@pytest.mark.asyncio
async def test_get_info_and_weights_info_report_configured_lora(tmp_path):
    config = _lora_config(rank=16, target_modules=["q_proj", "v_proj", "lm_head"])
    server = _server(config, engine=SimpleNamespace(config=OmegaConf.create(config)))

    info = await server.get_info(SimpleNamespace(model_id="session:train:0"))
    assert info["is_lora"] is True
    assert info["lora_rank"] == 16
    assert info["lora_alpha"] == 16.0
    assert (info["train_attn"], info["train_mlp"], info["train_unembed"]) == (True, False, True)
    assert info["model_name"] == "Qwen/Qwen3-8B"

    # Legacy checkpoint metadata (written by upstream) gets the server's adapter.
    legacy = server._weights_info_metadata({"base_model": "Qwen/Qwen3-8B", "is_lora": False, "lora_rank": None})
    assert legacy["is_lora"] is True and legacy["lora_rank"] == 16 and legacy["train_unembed"] is True
    current = server._weights_info_metadata({"base_model": "b", "is_lora": True, "lora_rank": 16, "train_mlp": False})
    assert current == {"base_model": "b", "is_lora": True, "lora_rank": 16, "train_mlp": False}


def test_configured_lora_info_variants():
    envloop_yaml = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "/m",
                    "lora_rank": "${oc.decode:${oc.env:TEST_ENVLOOP_LORA_RANK,8}}",
                    "lora_alpha": 16,
                    "target_modules": "all-linear",
                }
            }
        }
    )
    info = configured_lora_info(envloop_yaml)
    assert (info.rank, info.alpha, info.train_attn, info.train_mlp, info.train_unembed) == (8, 16.0, True, True, False)
    assert configured_lora_info({"actor_rollout_ref": {"model": {"lora": {"rank": 4, "alpha": 8}}}}).rank == 4
    assert configured_lora_info({"actor_rollout_ref": {"model": {"lora_rank": 0}}}).is_lora is False
    assert configured_lora_info(None).is_lora is False
    assert configured_lora_info(MagicMock()).is_lora is False
    assert train_flags_for_target_modules(["gate_proj", "up_proj", "down_proj"]) == (False, True, False)
    assert train_flags_for_target_modules(["model.layers.0.self_attn.q_proj"]) == (True, False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "status", "code"),
    [
        (
            {"target_prompt_logprobs": {"data": [1], "dtype": "int64", "shape": [1, 1]}},
            HTTPStatus.BAD_REQUEST,
            "TARGET_PROMPT_LOGPROBS_UNSUPPORTED",
        ),
        ({"topk_sample_logprobs": 21}, HTTPStatus.BAD_REQUEST, "TOPK_SAMPLE_LOGPROBS_TOO_LARGE"),
        ({"topk_sample_logprobs": -1}, HTTPStatus.UNPROCESSABLE_ENTITY, "TOPK_SAMPLE_LOGPROBS_INVALID"),
    ],
)
async def test_asample_rejects_unsupported_sampling_fields_with_typed_errors(overrides, status, code):
    server = _server()
    req = SimpleNamespace(sampling_session_id=None, base_model="actor", model_path=None, **overrides)
    with pytest.raises(HTTPException) as exc_info:
        await server.asample(req)
    assert exc_info.value.status_code == status
    assert exc_info.value.detail["code"] == code
    assert server._pending == {}


@pytest.mark.asyncio
async def test_asample_with_topk_sample_logprobs_uses_rollout_not_actor_prompt_logprob_path(monkeypatch):
    server = _server()
    server._model_resource_manager.configure_resources(rollout_enabled=True, reference_enabled=False)
    server._model_resource_manager.register_actor_sampler("s", base_model="actor", actor_id=0)
    rollout = AsyncMock(return_value={"type": "sample", "sequences": []})
    actor = AsyncMock()
    monkeypatch.setitem(server.asample.__globals__, "tinker_sample", rollout)
    monkeypatch.setitem(server.asample.__globals__, "tinker_sample_prompt_logprobs", actor)
    req = _sample_req([1, 2], k=2, max_tokens=1)
    req.sampling_session_id = "s"
    req.prompt_logprobs = True

    await server.asample(req)
    await asyncio.gather(*server._pending.values())

    rollout.assert_awaited_once()
    actor.assert_not_awaited()


def test_routes_are_bound_to_the_intended_handlers_and_request_schemas():
    from verl_tinker.tinker_router import app

    routes = {route.path: route for route in app.routes if hasattr(route, "endpoint")}
    expected = {
        "/api/v1/asample": ("asample", "TinkerSampleRequest"),
        "/api/v1/create_model": ("create_model", "TinkerCreateModelRequest"),
        "/api/v1/forward": ("forward", "TinkerForwardRequest"),
        "/api/v1/forward_backward": ("forward_backward", "TinkerForwardBackwardRequest"),
        "/api/v1/optim_step": ("optim_step", "OptimStepRequest"),
        "/api/v1/get_info": ("get_info", "GetInfoRequest"),
    }
    for path, (handler, schema) in expected.items():
        endpoint = routes[path].endpoint
        assert endpoint.__name__ == handler, path
        assert schema in str(endpoint.__annotations__.get("req")), path
    assert not any(route.endswith("validate_sample_request") for route in routes)
