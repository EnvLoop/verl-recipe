"""Exercise exact router methods/FIFO with the heavyweight engine boundary replaced.

Ray/VeRL are not installed by this CPU test. Methods are compiled directly from
tinker_router.py (not reimplemented); full Ray Serve/GPU validation is separate.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import sys
import threading
import types
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from omegaconf import DictConfig
from verl_tinker.external_weights import retained_external_result, submit_external_weights


def router_class():
    source = Path(__file__).parents[1] / "src/verl_tinker/tinker_router.py"
    tree = ast.parse(source.read_text())
    names = {"ServerStatus", "ScheduledOpKind", "RequestStatus", "FifoReadWriteGate"}
    methods = {
        "_get_engine",
        "_require_ready",
        "_close_unawaited",
        "_server_option",
        "_optim_params",
        "_run_optim_step",
        "_future_envelope",
        "_schedule",
        "_remember_retrieved",
        "_compact_request_status_if_needed",
        "optim_step",
        "save_weights_external",
        "retrieve_future",
    }
    nodes = ast.parse("from __future__ import annotations").body
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "TinkerServer":
            node.decorator_list = []
            node.body = [
                method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in methods
            ]
            for method in node.body:
                method.decorator_list = [
                    decorator
                    for decorator in method.decorator_list
                    if isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                ]
            nodes.append(node)
    namespace = dict(globals())
    namespace.update(
        logger=logging.getLogger(__name__),
        MAX_REQUEST_STATUS_ENTRIES=100000,
        MAX_RECENTLY_RETRIEVED_RESULTS=1024,
        tinker_optim_step=optimizer_boundary,
        AsyncIterator=AsyncIterator,
        Mapping=Mapping,
        DictConfig=DictConfig,
        deque=deque,
        Enum=Enum,
        asynccontextmanager=asynccontextmanager,
        uuid=uuid,
        retained_external_result=retained_external_result,
        submit_external_weights=submit_external_weights,
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


async def optimizer_boundary(engine, params):
    await asyncio.to_thread(engine.optim_step)
    return {"type": "optim_step", "metrics": {"step": engine.weight}}


class Engine:
    def __init__(self):
        self.weight = 0
        self.events = []

    def optim_step(self):
        self.weight += 1
        self.events.append(("optim", self.weight))

    def save_checkpoint(self):
        self.events.append(("save", self.weight))
        return self.weight


@pytest.fixture
def server(monkeypatch):
    namespace = router_class()
    result = object.__new__(namespace["TinkerServer"])
    result._status = namespace["ServerStatus"].INITIALIZED
    result._engine = Engine()
    result.config = {"server": {"external_weights_handler": "external_handler_fixture.export"}}
    result._checkpoint_root = "/unused-fixture"
    result._model_to_base_model = {"model-1": "qwen"}
    result._lora_info = lambda: SimpleNamespace(is_lora=True)
    result._futures, result._pending, result._errors = {}, {}, {}
    result._request_status, result._retrieved_request_status_archive = {}, {}
    result._op_gate = namespace["FifoReadWriteGate"]()
    result._model_resource_manager = SimpleNamespace(actor_updated=lambda: None)
    handler = types.ModuleType("external_handler_fixture")
    handler.export = lambda engine, config, root, body: {
        "type": "save_weights_external",
        "path": "tinker://verl-tinker/external_weights/" + body["path"],
        "frozen_weight": engine.save_checkpoint(),
    }
    monkeypatch.setitem(sys.modules, "external_handler_fixture", handler)
    return result


@pytest.mark.asyncio
async def test_external_export_is_between_optimizer_steps_in_existing_fifo(server, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    frozen = []

    def export(engine, config, root, body):
        frozen.append(engine.save_checkpoint())
        entered.set()
        assert release.wait(5)
        return {"type": "save_weights_external", "path": "tinker://verl-tinker/external_weights/one"}

    monkeypatch.setattr(sys.modules["external_handler_fixture"], "export", export)
    first = await server.optim_step(SimpleNamespace(model_id="model-1", adam_params={}))
    saved = await server.save_weights_external({"model_id": "model-1", "path": "one", "seq_id": 2})
    last = await server.optim_step(SimpleNamespace(model_id="model-1", adam_params={}))
    assert await asyncio.to_thread(entered.wait, 2)
    assert server._engine.events == [("optim", 1), ("save", 1)]
    release.set()
    await asyncio.gather(*list(server._pending.values()))
    assert server._engine.events == [("optim", 1), ("save", 1), ("optim", 2)]
    assert frozen == [1]
    assert (await server.retrieve_future(SimpleNamespace(request_id=saved["request_id"])))["path"].endswith("/one")
    assert server._futures[first["request_id"]]["metrics"]["step"] == 1
    assert server._futures[last["request_id"]]["metrics"]["step"] == 2


@pytest.mark.asyncio
async def test_same_external_sequence_has_one_future_and_one_snapshot(server):
    body = {"model_id": "model-1", "path": "same", "seq_id": 5, "ttl_seconds": 3600}
    one = await server.save_weights_external(body)
    two = await server.save_weights_external(body)
    assert one == two
    await asyncio.gather(*list(server._pending.values()))
    result = await server.retrieve_future(SimpleNamespace(request_id=one["request_id"]))
    assert result["frozen_weight"] == 0
    assert server._engine.events == [("save", 0)]
    server._futures.clear()
    server.__dict__.setdefault("_recently_retrieved", OrderedDict()).clear()
    assert await server.retrieve_future(SimpleNamespace(request_id=one["request_id"])) == result
    with pytest.raises(HTTPException) as conflict:
        await server.save_weights_external(body | {"path": "different"})
    assert conflict.value.status_code == 409


@pytest.mark.asyncio
async def test_missing_handler_and_foreign_model_fail_before_queue(server):
    server.config = {"server": {}}
    with pytest.raises(HTTPException) as missing:
        await server.save_weights_external({"model_id": "model-1", "path": "x", "seq_id": 1})
    assert missing.value.status_code == 501
    with pytest.raises(HTTPException) as foreign:
        await server.save_weights_external({"model_id": "foreign", "path": "x", "seq_id": 1})
    assert foreign.value.status_code == 404
    assert server._pending == {} and server._engine.events == []


@pytest.mark.asyncio
async def test_failed_export_is_cached_without_saving_again(server, monkeypatch):
    def failing(engine, *_args):
        engine.save_checkpoint()
        raise RuntimeError("private provider message must not be returned")

    monkeypatch.setattr(sys.modules["external_handler_fixture"], "export", failing)
    body = {"model_id": "model-1", "path": "bad", "seq_id": 1}
    future = await server.save_weights_external(body)
    await asyncio.gather(*list(server._pending.values()))
    result = await server.retrieve_future(SimpleNamespace(request_id=future["request_id"]))
    assert result["error"] == "External weights export failed"
    assert "private" not in str(result)
    assert await server.save_weights_external(body) == future
    assert server._engine.events == [("save", 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra", [{"ttl_seconds": 3}, {"seq_id": -1}, {"age_encryption_recipients": []}, {"path": "../escape"}]
)
async def test_invalid_export_request_has_no_gpu_side_effect(server, extra):
    with pytest.raises(HTTPException):
        await server.save_weights_external({"model_id": "model-1", "path": "x", "seq_id": 1} | extra)
    assert server._engine.events == [] and server._pending == {}
