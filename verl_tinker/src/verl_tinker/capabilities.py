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

"""Capabilities this recipe declares to EnvLoop (static) and reports from a running head (verified).

``LM_HEAD_LORA_PROMPT_LOGPROBS``: the rollout's vLLM applies an ``lm_head`` LoRA (vLLM #48850) and maps
it per token for prompt logprobs and mixed-adapter batches (vLLM #62049); both via ``vllm_backports``,
applied at ``verl_tinker.start``. The running value is only True when the installed vLLM actually
contains both fixes, so a head whose backports failed (or were never applied) reports False.

``LORA_SPLIT_QKV``: the rollout's vLLM serves adapters with separate ``in_proj_{q,k,v}`` LoRAs (the layout
``verl_tinker.lora_split`` trains) through the ``vllm_split_qkv`` general plugin. True when vLLM will load the
plugin (entry point installed, not excluded by ``VLLM_PLUGINS``, hooks present in the installed vLLM).
"""

from __future__ import annotations

from functools import lru_cache

from . import vllm_backports, vllm_split_qkv

# This recipe version ships the backports.
LM_HEAD_LORA_PROMPT_LOGPROBS = True
LORA_SPLIT_QKV = True


@lru_cache(maxsize=1)
def runtime_capabilities() -> dict[str, bool]:
    """What this head can do, as reported in ``get_info`` / ``get_server_capabilities``."""
    return {
        "lm_head_lora_prompt_logprobs": LM_HEAD_LORA_PROMPT_LOGPROBS and vllm_backports.present_installed(),
        "lora_split_qkv": LORA_SPLIT_QKV and vllm_split_qkv.enabled(vllm_backports.installed_root()),
    }


def assert_startup_capabilities() -> None:
    """Called after ``vllm_backports.apply_installed``: a declared capability must hold."""
    runtime_capabilities.cache_clear()
    if (
        LM_HEAD_LORA_PROMPT_LOGPROBS
        and vllm_backports.find_installed()
        and not runtime_capabilities()["lm_head_lora_prompt_logprobs"]
    ):
        raise vllm_backports.BackportError("vLLM backports for lm_head LoRA are not in the installed vLLM")
    if LORA_SPLIT_QKV and vllm_backports.find_installed() and not runtime_capabilities()["lora_split_qkv"]:
        raise vllm_backports.BackportError("vLLM will not load the verl_tinker split q/k/v LoRA plugin")
