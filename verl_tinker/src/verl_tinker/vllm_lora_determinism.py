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

"""Run-to-run deterministic LoRA in the vLLM rollout, through vLLM's own tuned-kernel-config knob.

vLLM's default Punica LoRA ``shrink`` splits K 64 (or 8) ways and reduces with float ``atomic_add``, so the same
prompt scored twice gives slightly different LoRA outputs (vllm-project/vllm#50059; fix #50278 is still open).
Through a rank-r all-layer adapter on Qwen3.8-27B the drift reaches 0.49 nats on single prompt logprobs between
two identical requests (2x H200, TP 2); with ``split_k=1`` repeated requests, and requests on either side of a
rollout sleep/wake, are bit-identical.

``VLLM_TUNED_CONFIG_FOLDER`` points vLLM at ``vllm_lora_configs``: per GPU name, a ``shrink`` config with
``split_k=1`` for every adapter count, slice count and shape (vLLM picks the nearest key), using the default block
sizes otherwise. ``expand`` has no file and keeps vLLM's default (it does not split K). A folder the operator set
is kept, and ``VLLM_BATCH_INVARIANT=1`` (which ignores tuned configs and is deterministic itself) is left alone.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "vllm_lora_configs"


def enable(environ=os.environ) -> None:
    """Before any vLLM process starts; vLLM workers inherit the environment."""
    if environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"):
        return
    environ.setdefault("VLLM_TUNED_CONFIG_FOLDER", str(CONFIG_DIR))
