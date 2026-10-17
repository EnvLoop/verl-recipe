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

vLLM finds the file as ``torch.cuda.get_device_name()`` with spaces and dashes replaced by underscores, plus
``_SHRINK.json`` (``load_lora_op_config`` in vllm/lora/ops/triton_ops/utils.py, 0.26 and 0.31). On a GPU without a
file vLLM silently uses the nondeterministic default, so the head reports ``lora_deterministic`` and warns.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "vllm_lora_configs"

# ``torch.cuda.get_device_name()`` / NVML names of the GPUs EnvLoop's catalog can hand out. Sources: vLLM's own
# tuned-config file names (vllm/model_executor/layers/fused_moe/configs: H100 80GB HBM3, H100 PCIe, H200, B200,
# A100-SXM4-40/80GB, RTX PRO 6000 Blackwell Server/Workstation Edition), and vLLM startup logs for the rest
# (H100 NVL, H200 NVL, A100 80GB PCIe, A100-PCIE-40GB).
DEVICE_NAMES = (
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 NVL",
    "NVIDIA H200",
    "NVIDIA H200 NVL",
    "NVIDIA B200",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-PCIE-40GB",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
)


def shrink_config_name(device_name: str) -> str:
    """vLLM's file name for a GPU's tuned LoRA shrink config."""
    return device_name.replace(" ", "_").replace("-", "_") + "_SHRINK.json"


def enable(environ=os.environ) -> None:
    """Before any vLLM process starts; vLLM workers inherit the environment."""
    if _batch_invariant(environ):
        return
    environ.setdefault("VLLM_TUNED_CONFIG_FOLDER", str(CONFIG_DIR))


def deterministic(device_name: str | None, environ=os.environ) -> bool:
    """vLLM's LoRA shrink on this GPU reduces without atomics: batch-invariant mode, or a split_k=1 config file."""
    if _batch_invariant(environ):
        return True
    folder = environ.get("VLLM_TUNED_CONFIG_FOLDER")
    if not device_name or not folder:
        return False
    path = Path(folder) / shrink_config_name(device_name)
    try:
        leaves = list(_leaves(json.loads(path.read_text())))
    except (OSError, ValueError):
        return False
    return bool(leaves) and all(leaf.get("split_k") == 1 for leaf in leaves)


def current_device_name() -> str | None:
    """The rollout GPU's name as vLLM sees it (NVML; does not create a CUDA context). None without a GPU/vLLM."""
    try:
        from vllm.platforms import current_platform

        return current_platform.get_device_name(0)
    except Exception:  # no vLLM, no GPU, or NVML unavailable
        return None


def check(environ=os.environ) -> bool:
    """Startup check: warn when the rollout's LoRA kernels on this GPU are not run-to-run deterministic."""
    name = current_device_name()
    ok = deterministic(name, environ)
    if not ok:
        logger.warning(
            "vLLM LoRA shrink is not deterministic on GPU %r (no split_k=1 config %s in VLLM_TUNED_CONFIG_FOLDER=%s): "
            "repeated prompt logprobs can differ; capabilities.lora_deterministic=false",
            name,
            shrink_config_name(name) if name else "?",
            environ.get("VLLM_TUNED_CONFIG_FOLDER"),
        )
    return ok


def write_configs(folder: Path = CONFIG_DIR, names=DEVICE_NAMES) -> None:
    """(Re)generate the shipped files: vLLM's default shrink blocks with split_k=1."""
    small = {"block_m": 32, "block_n": 16, "block_k": 256, "split_k": 1, "num_warps": 4, "num_ctas": 1,
             "group_size_m": 8, "num_stages": 2, "max_nreg": None}  # fmt: skip
    by_m = {"1": {"1": {"1": small}}, "128": {"1": {"1": dict(small, block_k=32)}}}  # vLLM's batch < 128 split
    config = {"1": {str(slices): by_m for slices in range(1, 9)}}  # max_loras -> num_slices -> m -> k -> n
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / shrink_config_name(name)).write_text(json.dumps(config, indent=1) + "\n")


def _batch_invariant(environ) -> bool:
    return environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0")


def _leaves(node):
    if isinstance(node, dict) and "split_k" in node:
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)


if __name__ == "__main__":
    write_configs()
