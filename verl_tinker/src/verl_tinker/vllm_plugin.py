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

"""vLLM general plugin: LoRA on the Qwen3.5 output layer for vLLM releases that lack it.

Tinker's ``train_unembed`` trains a LoRA on ``lm_head``. vLLM only wraps a model's
output layer with ``LogitsProcessorWithLoRA`` when the model class lists ``lm_head`` in
``embedding_modules`` (``vllm.lora.utils.get_supported_lora_modules``). vLLM 0.31 declares
it on ``Qwen3_5ForCausalLMBase``; 0.26 (this recipe's lock) does not, and VeRL's in-memory
adapter sync (``TensorLoRARequest``) does not check expected modules, so the rollout
silently dropped the ``lm_head`` LoRA and sampled from different weights than the actor
trained. This backports vLLM 0.31's declaration (the same one 0.26 already carries for
Qwen3, whose LoRA machinery handles the tied output layer of Qwen3.5-4B the same way).

Registered under the ``vllm.general_plugins`` entry point, which vLLM loads in every
process (API server, engine core, workers) before the model is built. A vLLM that already
declares ``lm_head`` is left untouched.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# vllm 0.31 vllm/model_executor/models/qwen3_5.py: Qwen3_5ForCausalLMBase.embedding_modules
QWEN3_5_EMBEDDING_MODULES = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}


def register() -> None:
    try:
        from vllm.model_executor.models import qwen3_5
    except Exception:  # noqa: BLE001 - a vLLM without Qwen3.5 has nothing to patch
        return
    base = getattr(qwen3_5, "Qwen3_5ForCausalLMBase", None)
    if base is None or "lm_head" in (getattr(base, "embedding_modules", None) or {}):
        return
    base.embedding_modules = dict(QWEN3_5_EMBEDDING_MODULES)
    logger.info("verl_tinker: enabled LoRA on the Qwen3.5 lm_head (vLLM embedding_modules backport)")
