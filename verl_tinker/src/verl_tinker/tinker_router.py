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

"""Ray Serve deployment exposing the Tinker-compatible server API."""

import asyncio
import logging
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from omegaconf import DictConfig
from ray import serve
from tinker.types import (
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    FutureRetrieveRequest,
    GetInfoRequest,
    OptimStepRequest,
    SaveWeightsForSamplerRequest,
)

from .backends.colocated import ColocatedBackend, RolloutState
from .backends.teacher import TeacherInferenceBackend
from .capabilities import runtime_capabilities
from .external_weights import retained_external_result, submit_external_weights
from .lora_info import LoraInfo, configured_lora_info
from .model_resource_manager import (
    LazyRolloutPlan,
    ModelResourceManager,
    ModelResourceManagerError,
    SamplingResource,
    StaleSamplerError,
    UnknownSamplerError,
)
from .sampler_adapters import (
    ADAPTER_DIR,
    SamplerAdapterError,
    SamplerAdapterIds,
    SamplerAdapterNotFound,
    adapter_dir,
    adapter_tag,
    check_adapter_compatible,
    read_adapter_config,
)
from .schemas import (
    StatusResponse,
    TinkerCreateModelRequest,
    TinkerForwardBackwardRequest,
    TinkerForwardRequest,
    TinkerSampleRequest,
)
from .tinker_losses import LossContractError
from .tinker_ops import (
    forward as tinker_forward,
)
from .tinker_ops import (
    forward_backward as tinker_forward_backward,
)
from .tinker_ops import (
    get_configured_model_name,
    get_supported_models,
    load_state,
    load_state_metadata,
    save_state,
    state_path_to_local,
    validate_tinker_loss_request,
)
from .tinker_ops import (
    optim_step as tinker_optim_step,
)
from .tinker_ops import (
    sample as tinker_sample,
)
from .tinker_ops import (
    sample_prompt_logprobs as tinker_sample_prompt_logprobs,
)
from .tinker_ops import (
    save_weights_for_sampler as tinker_save_weights_for_sampler,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="VeRL Tinker Server", version="0.1.0")

# Only used for full-parameter servers (lora_rank == 0): the Tinker SDK's
# create_training_client_from_state asserts LoRA metadata before load_state.
TINKER_COOKBOOK_COMPAT_LORA_RANK = 1
MAX_REQUEST_STATUS_ENTRIES = 100_000
# Retrieved results kept for a repeated retrieve_future (the client retries when a response is lost).
MAX_RECENTLY_RETRIEVED_RESULTS = 1024
# vLLM's default ``max_logprobs``; override with ``server.max_topk_logprobs``.
DEFAULT_MAX_TOPK_LOGPROBS = 20


def _typed_error(status: HTTPStatus, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


class ServerStatus(str, Enum):
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    ERROR = "error"


class ScheduledOpKind(str, Enum):
    SAMPLE = "sample"
    EXCLUSIVE = "exclusive"


class RequestStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    RETRIEVED = "retrieved"
    ERROR = "error"


class FifoReadWriteGate:
    """
    FIFO gate where consecutive SAMPLE operations may run concurrently.

    Every non-SAMPLE operation is exclusive and acts as a barrier:

        SAMPLE, SAMPLE, EXCLUSIVE, SAMPLE
        └── run together ──┘          └─ waits for EXCLUSIVE
    """

    def __init__(self, max_readers: int = 16):
        max_readers = int(max_readers)
        if max_readers <= 0:
            raise ValueError(f"max_readers must be positive, got {max_readers}")

        self._max_readers = max_readers

        # Protects the queue and active-operation counters.
        self._lock = asyncio.Lock()

        # Each queued request gets an Event that is set when it may enter.
        self._queue: deque[tuple[ScheduledOpKind, asyncio.Event]] = deque()

        self._active_samples = 0
        self._exclusive_active = False

    def _grant_ready_locked(self) -> None:
        """Grant the FIFO prefix that can currently run.

        The caller must hold self._lock.
        """
        if self._exclusive_active:
            return

        while self._queue:
            kind, ready = self._queue[0]

            if kind is not ScheduledOpKind.SAMPLE:
                # A non-sample request is an exclusive barrier. It cannot start
                # until all earlier samples finish, and later requests cannot
                # pass it.
                if self._active_samples > 0:
                    return

                self._queue.popleft()
                self._exclusive_active = True
                ready.set()
                return

            # Samples may run concurrently only while they form the continuous
            # prefix at the front of the queue.
            if self._active_samples >= self._max_readers:
                return

            self._queue.popleft()
            self._active_samples += 1
            ready.set()

    @asynccontextmanager
    async def hold(
        self,
        kind: ScheduledOpKind,
    ) -> AsyncIterator[None]:
        ready = asyncio.Event()
        waiter = (kind, ready)

        # The order in which requests are appended here defines FIFO order.
        async with self._lock:
            self._queue.append(waiter)
            self._grant_ready_locked()

        try:
            await ready.wait()
            yield
        finally:
            async with self._lock:
                if ready.is_set():
                    # The request had already been granted, so release its slot.
                    if kind is ScheduledOpKind.SAMPLE:
                        self._active_samples -= 1
                    else:
                        self._exclusive_active = False
                else:
                    # The caller was cancelled while it was still waiting.
                    self._queue.remove(waiter)

                self._grant_ready_locked()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=1024,
    graceful_shutdown_wait_loop_s=5.0,
    graceful_shutdown_timeout_s=5.0,
)
@serve.ingress(app)
class TinkerServer:
    """Single Ray Serve/FastAPI route owner for the Tinker server."""

    def __init__(self, config):
        self._status = ServerStatus.INITIALIZING
        self._error: Optional[str] = None
        self._engine: Optional[ColocatedBackend] = None
        self._teacher_backend: Optional[TeacherInferenceBackend] = None
        self._sampler_adapters: Optional[SamplerAdapterIds] = None
        self._gpu_executor = ThreadPoolExecutor(max_workers=1)
        self._op_gate = FifoReadWriteGate(
            max_readers=config["server"].get("max_concurrent_samples", 16),
        )
        # Serializes a lazy rollout start and its bookkeeping across concurrent samples.
        self._rollout_start_lock = asyncio.Lock()

        self._futures: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._errors: dict[str, str] = {}
        self._request_status: dict[str, RequestStatus] = {}
        self._retrieved_request_status_archive: dict[str, RequestStatus] = {}
        self._shutdown_started = False
        self._model_to_base_model: dict[str, str] = {}
        self._saved_state_paths: dict[str, str] = {}
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._saved_state_metadata: dict[str, dict[str, Any]] = {}
        self._step_counter = 0
        self._checkpoint_root = config["server"].get("checkpoint_dir", "/tmp/tinker-checkpoints")

        actor_base_model = str(config["server"]["model_name"])
        actor_model_path = str(config["actor_rollout_ref"]["model"]["path"])
        self._model_resource_manager = ModelResourceManager(
            actor_model_identifiers=(actor_base_model, actor_model_path),
        )

        self.config = config

        loop = asyncio.get_running_loop()
        self._init_future: asyncio.Future = loop.run_in_executor(self._gpu_executor, self._init_engine)

    async def _exec(self, op_name: str, fn, *args, **kwargs):
        """Run a blocking engine operation in the serialized GPU executor."""
        self._require_ready()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._gpu_executor, partial(fn, *args, **kwargs))
        except Exception as e:
            logger.exception(f"{op_name} failed: {e}")
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, str(e)) from e

    def _get_engine(self) -> ColocatedBackend:
        self._require_ready()
        if self._engine is None:
            raise RuntimeError("Server is initialized without an actor engine")
        return self._engine

    def _require_ready(self) -> None:
        if getattr(self, "_status", None) != ServerStatus.INITIALIZED:
            status = getattr(self, "_status", ServerStatus.ERROR)
            status_value = getattr(status, "value", str(status))
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Server not ready: status={status_value}",
            )

    def _close_unawaited(self, awaitable) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close rejected scheduled coroutine")

    def _lora_info(self) -> LoraInfo:
        """LoRA configuration this server was started with (rank 0 = full-parameter)."""
        info = getattr(self, "_configured_lora", None)
        if info is None:
            config = getattr(self, "config", None)
            if config is None and getattr(self, "_engine", None) is not None:
                config = getattr(self._engine, "config", None)
            info = configured_lora_info(config)
            self._configured_lora = info
        return info

    def _server_option(self, key: str, default: Any) -> Any:
        config = getattr(self, "config", None)
        if config is None and getattr(self, "_engine", None) is not None:
            config = getattr(self._engine, "config", None)
        if not isinstance(config, (DictConfig, Mapping)):
            return default
        server = config.get("server")
        if not isinstance(server, (DictConfig, Mapping)):
            return default
        value = server.get(key, default)
        return default if value is None else value

    def _metadata_from_create_model(self, req: TinkerCreateModelRequest) -> dict[str, Any]:
        return {"base_model": req.base_model, **self._lora_info().metadata()}

    def _check_create_model_lora(self, req: TinkerCreateModelRequest) -> None:
        """The adapter is fixed at server start; a different rank (or layer selection) is a 409."""
        optimizer_config = getattr(req, "optimizer_config", None)
        optimizer_type = (
            optimizer_config.get("type")
            if isinstance(optimizer_config, dict)
            else getattr(optimizer_config, "type", None)
        )
        if optimizer_type not in (None, "adamw"):
            raise _typed_error(
                HTTPStatus.BAD_REQUEST,
                "OPTIMIZER_UNSUPPORTED",
                f"optimizer_config.type={optimizer_type!r} is not supported; this server only runs AdamW",
            )

        info = self._lora_info()
        lora_config = getattr(req, "lora_config", None)
        if lora_config is None or not info.is_lora:
            # Full-parameter servers train every weight and ignore lora_config.
            return
        requested_rank = int(getattr(lora_config, "rank", info.rank))
        if requested_rank != info.rank:
            raise HTTPException(
                HTTPStatus.CONFLICT,
                f"create_model requested LoRA rank {requested_rank}, but this server was started with "
                f"LoRA rank {info.rank} (actor_rollout_ref.model.lora_rank); the adapter rank cannot change "
                f"at runtime. Request rank={info.rank} or start a server with the desired rank.",
            )
        mismatched = {
            flag: bool(getattr(lora_config, flag))
            for flag in ("train_attn", "train_mlp", "train_unembed")
            if getattr(lora_config, flag, None) is not None and bool(getattr(lora_config, flag)) != getattr(info, flag)
        }
        if mismatched and info.explicit_targets:
            # The adapter was built from Tinker's flags (EnvLoop sets target_modules per
            # create_model); a different layer selection would silently train other layers.
            raise HTTPException(
                HTTPStatus.CONFLICT,
                f"create_model requested {mismatched}, but this server's LoRA adapter trains "
                f"train_attn={info.train_attn}, train_mlp={info.train_mlp}, train_unembed={info.train_unembed} "
                f"(target_modules={info.target_modules!r}); the adapter layers cannot change at runtime.",
            )
        for flag, requested in mismatched.items():
            logger.warning(
                "create_model requested %s=%s but the server adapter (target_modules=%r) has %s=%s; "
                "training uses the server configuration",
                flag,
                requested,
                info.target_modules,
                flag,
                getattr(info, flag),
            )

    def _current_model_metadata(self, model_id: str) -> dict[str, Any]:
        metadata = self._model_metadata.get(model_id)
        if metadata is None:
            raise HTTPException(
                HTTPStatus.CONFLICT,
                f"No metadata for model {model_id!r}; create_model must run first",
            )
        return metadata

    def _weights_info_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        info = self._lora_info()
        if info.is_lora:
            if metadata.get("is_lora") and metadata.get("lora_rank") is not None:
                return dict(metadata)
            # Checkpoint metadata written before the LoRA fields were recorded:
            # report the adapter this server trains.
            return {**metadata, **info.metadata()}
        # Full-parameter server. The Tinker SDK's create_training_client_from_state
        # asserts LoRA metadata before load_state, so keep the cookbook-compatible
        # shape here (create_model on a full-parameter server accepts any rank).
        return {
            **metadata,
            "is_lora": True,
            "lora_rank": TINKER_COOKBOOK_COMPAT_LORA_RANK,
            "train_unembed": True,
            "train_mlp": True,
            "train_attn": True,
        }

    def _validate_loss_request(self, loss_input, *, forward_only: bool) -> None:
        try:
            validate_tinker_loss_request(
                loss_input.loss_fn,
                loss_input.loss_fn_config,
                loss_input.data,
                forward_only=forward_only,
            )
        except LossContractError as exc:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc

    @staticmethod
    def _optim_params(req: OptimStepRequest):
        """AdamParams from either the 0.19 (``adam_params``) or 0.30 (``optim_params``) request type."""
        params = getattr(req, "optim_params", None)
        if params is None:
            params = getattr(req, "adam_params", None)
        if getattr(params, "type", None) == "dimuon":
            raise _typed_error(
                HTTPStatus.BAD_REQUEST,
                "OPTIMIZER_UNSUPPORTED",
                "Dimuon optimizer params are not supported; this server only runs AdamW",
            )
        return params

    async def _run_optim_step(self, req: OptimStepRequest) -> dict:
        try:
            result = await tinker_optim_step(self._get_engine(), self._optim_params(req))
        except BaseException:
            # An optimizer failure may have partially changed weights. Give that
            # state a fresh identity so it can never compare equal to rollout.
            self._model_resource_manager.actor_updated()
            raise
        self._model_resource_manager.actor_updated()
        return result

    async def _run_save_state(self, req: dict) -> dict:
        result = await save_state(
            self._get_engine(),
            self._checkpoint_root,
            self._saved_state_paths,
            self._saved_state_metadata,
            self._current_model_metadata(req.get("model_id", "")),
            self._step_counter,
            req.get("path"),
        )
        self._model_resource_manager.state_saved(result["path"])
        return result

    def _resolve_load_path(self, uri: str) -> str:
        if not uri or not uri.startswith("tinker://"):
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"Invalid checkpoint path: {uri!r}")
        local_dir = self._saved_state_paths.get(uri) or state_path_to_local(self._checkpoint_root, uri)
        if not Path(local_dir).exists():
            raise HTTPException(HTTPStatus.NOT_FOUND, f"Unknown checkpoint path: {uri}")
        return local_dir

    async def _run_load_state(self, req: dict) -> dict:
        uri = req.get("path", "")
        self._resolve_load_path(uri)
        if self._model_resource_manager.should_skip_state_load(uri):
            logger.info(
                "[model_resource_manager] skipping load path=%s actor_id=%s reason=already_loaded",
                uri,
                self._model_resource_manager.actor_id,
            )
            return {"type": "load_weights", "path": uri}

        try:
            result = await load_state(
                self._get_engine(),
                self._checkpoint_root,
                self._saved_state_paths,
                uri,
                load_optimizer=req.get("optimizer", True),
            )
        except BaseException:
            self._model_resource_manager.state_load_failed()
            raise

        self._model_resource_manager.state_loaded(uri)
        return result

    async def _run_save_weights_for_sampler(self, req: SaveWeightsForSamplerRequest) -> dict:
        named = req.path is not None
        session_id: str | None = None
        if not named and req.sampling_session_seq_id is None:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                "Unnamed save_weights_for_sampler requires sampling_session_seq_id",
            )
        if not named:
            session_id = self._model_resource_manager.session_id_for_model(str(req.model_id))

        engine = self._get_engine()
        try:
            # A lazy rollout starts here (``ColocatedBackend.update_weights``) inside this request's future.
            result = await tinker_save_weights_for_sampler(engine, named=named)
        except BaseException:
            self._model_resource_manager.rollout_synchronization_failed(
                rollout_started=engine.rollout_state is RolloutState.READY,
            )
            raise

        rollout_id = self._model_resource_manager.rollout_synchronized()
        if named:
            self._model_resource_manager.sampler_path_saved(result["path"])
            return result

        assert session_id is not None
        sampler_id = self._model_resource_manager.sampler_id(session_id, req.sampling_session_seq_id)
        self._model_resource_manager.register_actor_sampler(
            sampler_id,
            base_model=self._model_resource_manager.actor_base_model,
            actor_id=rollout_id,
        )
        result["sampling_session_id"] = sampler_id
        return result

    async def _start_lazy_rollout(self, binding) -> None:
        """Start a lazy rollout so it holds ``binding``'s weights (``None``: any weights).

        Runs inside the request's scheduled future, so the client polls with
        ``try_again`` while vLLM starts. Concurrent samples wait on the start lock and
        then re-resolve against the started rollout.
        """
        async with self._rollout_start_lock:
            manager = self._model_resource_manager
            plan = manager.lazy_rollout_plan(binding)
            if plan is None:
                return
            engine = self._get_engine()
            try:
                await asyncio.to_thread(
                    engine.start_rollout,
                    sync_weights=plan is LazyRolloutPlan.SYNC,
                    reason=f"sampler {getattr(binding, 'sampler_id', None) or 'request'} ({plan.value})",
                )
            except BaseException:
                manager.rollout_synchronization_failed(rollout_started=engine.rollout_state is RolloutState.READY)
                raise
            manager.rollout_started(plan)

    # ==================== Sampler adapters (EnvLoop) ====================

    def _sampler_adapter_error(self, exc: SamplerAdapterError) -> HTTPException:
        if isinstance(exc, SamplerAdapterNotFound):
            return _typed_error(HTTPStatus.NOT_FOUND, exc.code, str(exc))
        return _typed_error(HTTPStatus.UNPROCESSABLE_ENTITY, exc.code, str(exc))

    def _checked_adapter_dir(self, path: Path) -> Path:
        """Raise unless this head can serve the PEFT adapter at ``path`` beside its current weights."""
        engine = self._get_engine()
        if getattr(self, "_sampler_adapters", None) is None:
            raise SamplerAdapterError("this server has no spare LoRA slots for sampler checkpoints")
        check_adapter_compatible(
            read_adapter_config(path),
            max_lora_rank=engine.sampler_max_lora_rank,
            unembed_enabled=engine.sampler_unembed_enabled,
            split_qkv_enabled=engine.sampler_split_qkv_enabled,
        )
        return path

    def _binding_adapter_dir(self, binding) -> Path | None:
        """The PEFT export that holds a sampler binding's weights, if it has one on this head."""
        if binding.is_adapter:
            return adapter_dir(self._checkpoint_root, adapter_tag(binding.adapter_path))
        for state_path in self._model_resource_manager.adapter_state_paths(binding.actor_id):
            local = self._saved_state_paths.get(state_path) or state_path_to_local(self._checkpoint_root, state_path)
            candidate = Path(local) / ADAPTER_DIR
            if (candidate / "adapter_config.json").is_file():
                return candidate
        return None

    async def _sample_with_adapter(self, path: Path, req):
        try:
            self._checked_adapter_dir(path)
        except SamplerAdapterError as exc:
            raise self._sampler_adapter_error(exc) from exc
        if self._model_resource_manager.rollout_pending:
            # A lazy rollout starts with any weights: the adapter replaces them for this request.
            await self._start_lazy_rollout(None)
        # vLLM loads the adapter from its path on first use and LRU-evicts idle ones.
        return await tinker_sample(self._get_engine(), req, lora=self._sampler_adapters.spec(path))

    def _init_engine(self):
        try:
            self._engine = ColocatedBackend(self.config)
            self._model_resource_manager.configure_resources(
                rollout_enabled=not self._engine.no_rollout_deployment,
                reference_enabled=self._engine.capabilities.use_reference_policy,
                rollout_started=self._engine.rollout_state is RolloutState.READY,
                rollout_loads_base_weights=self._engine.rollout_loads_base_weights,
            )
            slots = self._engine.sampler_adapter_slots
            if slots:
                self._sampler_adapters = SamplerAdapterIds()
                logger.info(
                    "[sampler adapters] %s extra LoRA slot(s), max_lora_rank=%s",
                    slots,
                    self._engine.sampler_max_lora_rank,
                )
            if TeacherInferenceBackend.is_enabled(self.config):
                self._teacher_backend = TeacherInferenceBackend(self.config)
                self._model_resource_manager.configure_teacher_models(self._teacher_backend.sampling_targets)
            if self._shutdown_started:
                logger.info("Tinker server initialization complete after shutdown was requested")
            else:
                self._status = ServerStatus.INITIALIZED
                logger.info("Tinker server initialization complete")
        except Exception as e:
            logger.exception(f"Initialization failed: {e}")
            if not self._shutdown_started:
                self._status = ServerStatus.ERROR
                self._error = str(e)

    def _stash(self, payload: dict) -> str:
        request_id = uuid.uuid4().hex
        self._request_status[request_id] = RequestStatus.COMPLETED
        self._futures[request_id] = payload
        return request_id

    def _future_envelope(self, request_id: str, model_id: Optional[str] = None) -> dict:
        return {"request_id": request_id, "model_id": model_id}

    def _schedule(self, request_id: str, coro, kind: ScheduledOpKind = ScheduledOpKind.EXCLUSIVE):
        try:
            self._require_ready()
        except HTTPException:
            self._close_unawaited(coro)
            raise

        self._request_status[request_id] = RequestStatus.PENDING
        logger.info(f"[tinker_router] scheduling task rid={request_id}")

        self._compact_request_status_if_needed()

        async def _run():
            try:
                async with self._op_gate.hold(kind):
                    try:
                        self._require_ready()
                    except HTTPException:
                        self._close_unawaited(coro)
                        raise
                    result = await coro
                logger.info(f"[tinker_router] task rid={request_id} succeeded")
                self._futures[request_id] = result
                self._request_status[request_id] = RequestStatus.COMPLETED
            except BaseException as e:
                logger.exception(f"[tinker_router] task rid={request_id} failed: {e!r}")
                self._errors[request_id] = repr(e)
                self._request_status[request_id] = RequestStatus.ERROR
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                self._pending.pop(request_id, None)

        self._pending[request_id] = asyncio.create_task(_run())

    def _remember_retrieved(self, request_id: str, response: dict[str, Any]) -> dict[str, Any]:
        recent = self.__dict__.setdefault("_recently_retrieved", OrderedDict())
        recent[request_id] = response
        while len(recent) > MAX_RECENTLY_RETRIEVED_RESULTS:
            recent.popitem(last=False)
        return response

    def _compact_request_status_if_needed(self) -> None:
        if len(self._request_status) < MAX_REQUEST_STATUS_ENTRIES:
            return

        retrieved_status = {
            request_id: status
            for request_id, status in self._request_status.items()
            if status is RequestStatus.RETRIEVED
        }
        if not retrieved_status:
            return

        self._retrieved_request_status_archive = retrieved_status
        for request_id in retrieved_status:
            del self._request_status[request_id]

    @app.get("/api/v1/healthz")
    async def healthz(self) -> dict[str, str]:
        """Compatibility health endpoint for existing launch helpers."""
        if self._status == ServerStatus.INITIALIZED:
            engine = getattr(self, "_engine", None)
            if engine is None:
                return {"status": "ready"}
            # A lazy rollout can start (or fail to start) long after the server is ready.
            health = {"status": "ready", "rollout": engine.rollout_state.value}
            if engine.rollout_start_error:
                health["rollout_error"] = engine.rollout_start_error
            return health
        if self._status == ServerStatus.ERROR:
            return {"status": self._status.value, "message": self._error or ""}
        return {"status": self._status.value}

    @app.post("/api/v1/shutdown")
    async def shutdown(self, request: Request) -> StatusResponse:
        """Compatibility shutdown endpoint for existing launch helpers."""
        self._shutdown_started = True
        self._status = ServerStatus.SHUTDOWN_COMPLETE
        self._error = None
        return StatusResponse(status="accepted")

    @app.api_route("/api/v1/get_server_capabilities", methods=["GET", "POST"])
    async def get_server_capabilities(self):
        return {
            **get_supported_models(self._get_engine(), getattr(self, "_teacher_backend", None)),
            "capabilities": runtime_capabilities(),
        }

    @app.post("/api/v1/client/config")
    async def client_config(self):
        self._require_ready()
        return {
            "pjwt_auth_enabled": False,
            "credential_default_source": "api_key",
            "sample_dispatch_bytes_semaphore_size": 0,
            "inflight_response_bytes_semaphore_size": 0,
            "parallel_fwdbwd_chunks": False,
        }

    @app.post("/api/v1/get_info")
    async def get_info(self, req: GetInfoRequest):
        # Always describe the model/tokenizer that this server actually loaded.
        # A client-provided base_model from create_model is request metadata and
        # must not override the configured model identity returned by get_info.
        model_name = get_configured_model_name(self._get_engine())
        lora = self._lora_info().metadata()
        return {
            "type": "get_info",
            "model_data": {"arch": None, "model_name": model_name, "tokenizer_id": model_name},
            "model_id": req.model_id,
            "optimizer_config": {"type": "adamw"},
            "capabilities": runtime_capabilities(),
            **lora,
            "model_name": model_name,
        }

    @app.post("/api/v1/create_session")
    async def create_session(self, _: dict = None):
        self._require_ready()
        return {
            "type": "create_session",
            "session_id": self._model_resource_manager.create_session(),
            "info_message": None,
            "warning_message": None,
            "error_message": None,
        }

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(self, session_id: str):
        self._require_ready()
        try:
            return {
                "training_run_ids": self._model_resource_manager.model_ids_for_session(session_id),
                "sampler_ids": self._model_resource_manager.valid_sampler_ids_for_session(session_id),
            }
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc

    @app.post("/api/v1/sessions")
    async def list_sessions(self):
        self._require_ready()
        return {"sessions": self._model_resource_manager.session_ids()}

    @app.post("/api/v1/create_model")
    async def create_model(self, req: TinkerCreateModelRequest):
        self._require_ready()
        self._check_create_model_lora(req)
        try:
            model_id = self._model_resource_manager.register_model(req.session_id, req.model_seq_id)
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if req.base_model:
            self._model_to_base_model[model_id] = req.base_model
        self._model_metadata[model_id] = self._metadata_from_create_model(req)
        request_id = self._stash({"type": "create_model", "model_id": model_id})
        return self._future_envelope(request_id, model_id=model_id)

    @app.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
    async def create_sampling_session(self, req: CreateSamplingSessionRequest):
        self._require_ready()
        try:
            tag = adapter_tag(req.model_path)
            if tag is not None:
                self._checked_adapter_dir(adapter_dir(self._checkpoint_root, tag))
        except SamplerAdapterError as exc:
            raise self._sampler_adapter_error(exc) from exc
        try:
            sampler_id = self._model_resource_manager.sampler_id(req.session_id, req.sampling_session_seq_id)
            self._model_resource_manager.resolve_sampler_target(
                sampler_id=sampler_id,
                base_model=req.base_model,
                model_path=req.model_path,
            )
        except ModelResourceManagerError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        return CreateSamplingSessionResponse(
            type="create_sampling_session",
            sampling_session_id=sampler_id,
        )

    @app.get("/api/v1/samplers/{sampler_id}")
    async def get_sampler(self, sampler_id: str):
        self._require_ready()
        try:
            binding = self._model_resource_manager.get_sampler(sampler_id)
        except UnknownSamplerError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
        return {
            "sampler_id": sampler_id,
            "base_model": binding.base_model,
            "model_path": binding.model_path,
        }

    @app.post("/api/v1/weights_info")
    async def weights_info(self, req: dict):
        self._require_ready()
        tinker_path = req.get("tinker_path") or req.get("path")
        if not tinker_path:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "weights_info requires tinker_path")

        metadata = load_state_metadata(self._checkpoint_root, self._saved_state_metadata, tinker_path)
        if metadata is None:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"Unknown checkpoint path: {tinker_path}")
        return self._weights_info_metadata(metadata)

    @app.post("/api/v1/session_heartbeat")
    async def session_heartbeat(self, body: dict = None):
        self._require_ready()
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/telemetry")
    async def telemetry(self, body: dict = None):
        self._require_ready()
        return {"status": "accepted"}

    @app.post("/api/v1/forward")
    async def forward(self, req: TinkerForwardRequest):
        """Tinker ``forward(data, loss_fn, loss_fn_config)``: logprobs + loss, no gradients."""
        self._require_ready()
        self._validate_loss_request(req.forward_input, forward_only=True)
        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            tinker_forward(
                self._get_engine(),
                req.forward_input.data,
                req.forward_input.loss_fn,
                req.forward_input.loss_fn_config,
            ),
        )
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/forward_backward")
    async def forward_backward(self, req: TinkerForwardBackwardRequest):
        self._require_ready()
        self._validate_loss_request(req.forward_backward_input, forward_only=False)
        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            tinker_forward_backward(
                self._get_engine(),
                req.forward_backward_input.data,
                req.forward_backward_input.loss_fn,
                req.forward_backward_input.loss_fn_config,
            ),
        )
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/optim_step")
    async def optim_step(self, req: OptimStepRequest):
        self._require_ready()
        self._optim_params(req)
        request_id = uuid.uuid4().hex
        self._schedule(request_id, self._run_optim_step(req))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(self, req: SaveWeightsForSamplerRequest):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, self._run_save_weights_for_sampler(req))
        return self._future_envelope(request_id, model_id=req.model_id)

    @app.post("/api/v1/save_weights")
    async def save_weights(self, req: dict):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._step_counter += 1
        self._schedule(request_id, self._run_save_state(req))
        return self._future_envelope(request_id, model_id=req.get("model_id"))

    @app.post("/api/v1/save_weights_external")
    async def save_weights_external(self, req: dict):
        return await submit_external_weights(self, req)

    @app.post("/api/v1/load_weights")
    async def load_weights(self, req: dict):
        self._require_ready()
        request_id = uuid.uuid4().hex
        self._schedule(request_id, self._run_load_state(req))
        return self._future_envelope(request_id, model_id=req.get("model_id"))

    def _validate_sample_request(self, req) -> None:
        if getattr(req, "target_prompt_logprobs", None) is not None:
            raise _typed_error(
                HTTPStatus.BAD_REQUEST,
                "TARGET_PROMPT_LOGPROBS_UNSUPPORTED",
                "target_prompt_logprobs is not supported by this server: the vLLM rollout only reports the "
                "actual prompt token and the top-k candidates at each prompt position and cannot score "
                "arbitrary token ids. Use topk_prompt_logprobs, or forward() for next-token logprobs.",
            )
        k = int(getattr(req, "topk_sample_logprobs", 0) or 0)
        max_k = int(self._server_option("max_topk_logprobs", DEFAULT_MAX_TOPK_LOGPROBS))
        if k < 0:
            raise _typed_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "TOPK_SAMPLE_LOGPROBS_INVALID",
                f"topk_sample_logprobs must be >= 0, got {k}",
            )
        if k > max_k:
            raise _typed_error(
                HTTPStatus.BAD_REQUEST,
                "TOPK_SAMPLE_LOGPROBS_TOO_LARGE",
                f"topk_sample_logprobs={k} exceeds this server's limit of {max_k} (vLLM max_logprobs)",
            )

    @app.post("/api/v1/asample")
    async def asample(self, req: TinkerSampleRequest):
        self._require_ready()
        self._validate_sample_request(req)
        try:
            binding = self._model_resource_manager.resolve_sampling_request(
                sampling_session_id=req.sampling_session_id,
                base_model=getattr(req, "base_model", None),
                model_path=getattr(req, "model_path", None),
            )
        except UnknownSamplerError as exc:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except (ModelResourceManagerError, SamplerAdapterError) as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(exc)) from exc

        scalar_prompt_logprobs = (
            int(getattr(req, "num_samples", 1)) == 1
            and getattr(getattr(req, "sampling_params", None), "max_tokens", None) == 1
            and getattr(req, "prompt_logprobs", None) is True
            and int(getattr(req, "topk_prompt_logprobs", 0)) == 0
            and int(getattr(req, "topk_sample_logprobs", 0) or 0) == 0
        )

        def _resolve():
            try:
                return self._model_resource_manager.resolve_resource(
                    binding,
                    scalar_prompt_logprobs=scalar_prompt_logprobs,
                )
            except StaleSamplerError as exc:
                # The bound weights left the rollout: serve the checkpoint saved at that
                # version as an extra LoRA adapter when this head kept its PEFT export.
                if getattr(self, "_sampler_adapters", None) and self._binding_adapter_dir(binding) is not None:
                    return SamplingResource.ADAPTER
                raise HTTPException(HTTPStatus.CONFLICT, str(exc)) from exc

        async def _sample_with_late_binding():
            resource = _resolve()
            if resource is SamplingResource.ROLLOUT and self._model_resource_manager.rollout_pending:
                await self._start_lazy_rollout(binding)
                resource = _resolve()

            if resource is SamplingResource.ADAPTER:
                return await self._sample_with_adapter(self._binding_adapter_dir(binding), req)
            if resource is SamplingResource.TEACHER:
                if self._teacher_backend is None or binding.teacher_model_path is None:
                    raise RuntimeError("Teacher sampler binding has no initialized teacher backend")
                return await tinker_sample(self._teacher_backend.get_client(binding.teacher_model_path), req)
            if resource is SamplingResource.ROLLOUT:
                return await tinker_sample(self._get_engine(), req)
            if resource is SamplingResource.ACTOR:
                return await tinker_sample_prompt_logprobs(self._get_engine(), req, reference=False)
            if resource is SamplingResource.REFERENCE:
                return await tinker_sample_prompt_logprobs(self._get_engine(), req, reference=True)
            raise RuntimeError(f"Unsupported sampling resource: {resource!r}")

        request_id = uuid.uuid4().hex
        self._schedule(
            request_id,
            _sample_with_late_binding(),
            ScheduledOpKind.SAMPLE,
        )
        return self._future_envelope(request_id, model_id=None)

    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(self, req: FutureRetrieveRequest):
        self._require_ready()
        request_id = req.request_id
        external = retained_external_result(self, request_id)
        if external is not None:
            self._futures.pop(request_id, None)
            self._errors.pop(request_id, None)
            self._request_status[request_id] = RequestStatus.RETRIEVED
            return external
        task = self._pending.get(request_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30.0)
            except TimeoutError:
                return {"type": "try_again", "message": "future not yet completed"}

            # we are seeing that sometimes we double request for the same id, leading to error
            return {"type": "try_again", "message": "future completed; retry to retrieve result"}

        if request_id in self._futures:
            payload = self._futures.pop(request_id)
            self._request_status[request_id] = RequestStatus.RETRIEVED
            return self._remember_retrieved(request_id, payload)

        if request_id in self._errors:
            error = self._errors.pop(request_id)
            self._request_status[request_id] = RequestStatus.RETRIEVED
            return self._remember_retrieved(request_id, {"error": error, "category": "Application"})

        recent = self.__dict__.get("_recently_retrieved")
        if recent is not None and request_id in recent:
            # The client retries a retrieve whose response it never received (e.g. a dropped
            # connection through the gateway); answer it again instead of failing the future.
            recent.move_to_end(request_id)
            return recent[request_id]

        if request_id in self._request_status or request_id in self._retrieved_request_status_archive:
            logger.warning(
                f"[tinker_router] retrieve_future for id={request_id}; but future likely already retrieved"
                f"pending={len(self._pending)} futures={len(self._futures)} errors={len(self._errors)}"
            )
            return {
                "error": "future id result already retrieved",
                "category": "Application",
                "request_id": request_id,
                "seen_before": True,
                "seen_status": "seen_before",
                "pending_count": len(self._pending),
                "ready_count": len(self._futures),
                "error_count": len(self._errors),
            }
        else:
            # we have never seen this request id so we will have to have raise
            logger.warning(
                f"[tinker_router] retrieve_future for unknown rid={request_id}; "
                f"pending={len(self._pending)} "
                f"futures={len(self._futures)} errors={len(self._errors)}"
            )
            return {
                "error": "future id was never seen by this server",
                "category": "Application",
                "request_id": request_id,
                "seen_before": False,
                "seen_status": "never_seen",
                "pending_count": len(self._pending),
                "ready_count": len(self._futures),
                "error_count": len(self._errors),
            }
