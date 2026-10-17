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

"""VeRL's vLLM rollout server with extra, per-request LoRA adapters (sampler checkpoints).

VeRL's ``vLLMHttpServer`` starts vLLM with ``max_loras=1`` and decodes every request with the
current-weights adapter (``VLLM_LORA_INT_ID``). This subclass

* keeps ``max_loras`` / ``max_cpu_loras`` / ``max_lora_rank`` from
  ``rollout.engine_kwargs.vllm`` (VeRL overwrites the first and last after merging them);
* pins the current-weights adapter after each sync, so vLLM's LoRA LRU only evicts sampler
  adapters (which it reloads from their path on demand);
* takes ``envloop_lora={lora_name, lora_int_id, lora_path}`` on ``generate`` and decodes that
  request with that adapter instead of the current weights, as SkyRL's vLLM server actor resolves
  a LoRA per request (Apache-2.0, NovaSky-AI/SkyRL
  ``skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py``).

The class lives in ``verl_tinker`` so Ray's server actor imports it from the recipe venv.
"""

from __future__ import annotations

import logging
from typing import Any

import ray
from vllm.lora.request import LoRARequest

from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from ..sampler_adapters import REQUEST_LORA, LoRAOverrideEngine, multi_lora_engine_args

logger = logging.getLogger(__name__)


class EnvLoopVLLMHttpServer(vLLMHttpServer):
    def _envloop_engine_kwargs(self) -> dict:
        engine_kwargs = self.config.get("engine_kwargs", {}) or {}
        return dict(engine_kwargs.get(self._get_engine_kwargs_key(), {}) or {})

    async def run_server(self, args):
        multi_lora_engine_args(args, self._envloop_engine_kwargs())
        logger.info(
            "[sampler adapters] vLLM max_loras=%s max_cpu_loras=%s max_lora_rank=%s",
            getattr(args, "max_loras", None),
            getattr(args, "max_cpu_loras", None),
            getattr(args, "max_lora_rank", None),
        )
        result = await super().run_server(args)
        self.engine = LoRAOverrideEngine(self.engine)
        return result

    async def run_headless(self, args):
        multi_lora_engine_args(args, self._envloop_engine_kwargs())
        return await super().run_headless(args)

    async def pin_current_weights(self) -> bool:
        """Keep VeRL's tensor-pushed current-weights adapter out of vLLM's LRU eviction."""
        if VLLM_LORA_INT_ID not in await self.engine.list_loras():
            return False
        return bool(await self.engine.pin_lora(VLLM_LORA_INT_ID))

    async def generate(self, *args, envloop_lora: dict | None = None, **kwargs) -> Any:
        if envloop_lora is None:
            return await super().generate(*args, **kwargs)
        token = REQUEST_LORA.set(
            LoRARequest(
                lora_name=envloop_lora["lora_name"],
                lora_int_id=int(envloop_lora["lora_int_id"]),
                lora_path=envloop_lora["lora_path"],
            )
        )
        try:
            return await super().generate(*args, **kwargs)
        finally:
            REQUEST_LORA.reset(token)


class EnvLoopVLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(EnvLoopVLLMHttpServer)
