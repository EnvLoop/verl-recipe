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

"""GPU validation of the EnvLoop Tinker-alignment changes against a real server.

Talks to the verl_tinker head over the same JSON wire shapes the EnvLoop
gateway forwards (no SDK needed). Skipped automatically without CUDA.

Run on a GPU host from the ``verl_tinker`` directory, inside the recipe venv
(``/tts/verl_tinker/.venv`` on EnvLoop guests)::

    # a) let the test start a small LoRA server itself (default Qwen/Qwen3-0.6B, rank 8, all local GPUs)
    TINKER_ALIGN_MODEL=Qwen/Qwen3-0.6B TINKER_ALIGN_LORA_RANK=8 \\
        python -m pytest -q -s tests/gpu/test_envloop_alignment_gpu.py

    # b) start it from an explicit config (e.g. EnvLoop's trainingmaster-actor-rollout.yaml + env)
    TINKER_ALIGN_CONFIG=/path/config.yaml python -m pytest -q -s tests/gpu/test_envloop_alignment_gpu.py

    # c) use an already running head
    TINKER_ALIGN_SERVER_URL=http://127.0.0.1:8000 TINKER_ALIGN_LORA_RANK=8 \\
        python -m pytest -q -s tests/gpu/test_envloop_alignment_gpu.py

Checks: real LoRA rank in get_info + 409 on mismatch; forward == forward_backward
logprobs and forward leaves accumulated gradients untouched; server loss:sum for
every Tinker RL loss (incl. gspo + its metrics) equals the Tinker formula
evaluated on the returned logprobs; grad_clip_norm scales the update as
expected and grad_norm is the pre-clip norm; topk_sample_logprobs shape and
consistency with the sampled logprobs; typed 400 for target_prompt_logprobs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="EnvLoop alignment GPU validation needs CUDA")

LORA_RANK = int(os.environ.get("TINKER_ALIGN_LORA_RANK", "8"))
MODEL = os.environ.get("TINKER_ALIGN_MODEL", "Qwen/Qwen3-0.6B")
PORT = int(os.environ.get("TINKER_ALIGN_PORT", "18765"))
READY_TIMEOUT_S = float(os.environ.get("TINKER_ALIGN_READY_TIMEOUT_S", "1800"))
TOL = float(os.environ.get("TINKER_ALIGN_REL_TOL", "0.02"))

_CONFIG_TEMPLATE = """
server:
  model_name: {model}
  host: 127.0.0.1
  port: {port}
  ray_address: local
  checkpoint_dir: {checkpoint_dir}
  enable_offload: true
  auto_merge_verl_default_config: true
actor_rollout_ref:
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 16
    fsdp_config:
      model_dtype: bfloat16
      param_offload: false
      optimizer_offload: false
      use_orig_params: true
    optim:
      lr_scheduler_type: constant
      lr: 1.0e-06
  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    gpu_memory_utilization: 0.5
    layered_summon: true
  ref:
    enable: false
  model:
    path: {model}
    lora_rank: {rank}
    lora_alpha: 16
    target_modules: all-linear
trainer:
  logger: [console]
  project_name: tts
  experiment_name: envloop-alignment-gpu
  nnodes: 1
  n_gpus_per_node: {n_gpus}
"""


# ==================== HTTP client ====================


class Head:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post(self, path: str, body: dict | None = None, *, expect_status: int = 200):
        data = json.dumps(body or {}).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                status, payload = response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            status, payload = exc.code, json.loads(exc.read() or b"{}")
        assert status == expect_status, f"{path}: HTTP {status} {payload}"
        return payload

    def get(self, path: str):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as response:
            return json.loads(response.read())

    def result(self, envelope: dict, timeout_s: float = 1200.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            payload = self.post("/api/v1/retrieve_future", {"request_id": envelope["request_id"]})
            if payload.get("type") == "try_again":
                continue
            assert "error" not in payload, payload
            return payload
        raise TimeoutError(f"future {envelope['request_id']} did not complete")

    def call(self, path: str, body: dict) -> dict:
        return self.result(self.post(path, body))


def _wait_ready(head: Head, proc: subprocess.Popen | None):
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"verl_tinker server exited early with code {proc.returncode}")
        try:
            status = head.get("/api/v1/healthz").get("status")
        except Exception:
            status = None
        if status == "ready":
            return
        if status == "error":
            raise RuntimeError(f"server failed to initialize: {head.get('/api/v1/healthz')}")
        time.sleep(10)
    raise TimeoutError("verl_tinker server did not become ready")


@pytest.fixture(scope="module")
def head():
    url = os.environ.get("TINKER_ALIGN_SERVER_URL")
    if url:
        h = Head(url)
        _wait_ready(h, None)
        yield h
        return

    workdir = Path(tempfile.mkdtemp(prefix="envloop-align-"))
    config_path = os.environ.get("TINKER_ALIGN_CONFIG")
    if not config_path:
        config_path = str(workdir / "config.yaml")
        Path(config_path).write_text(
            _CONFIG_TEMPLATE.format(
                model=MODEL,
                port=PORT,
                checkpoint_dir=workdir / "checkpoints",
                rank=LORA_RANK,
                n_gpus=max(1, torch.cuda.device_count()),
            )
        )
    log = open(workdir / "server.log", "w")  # noqa: SIM115 - closed in teardown
    proc = subprocess.Popen(
        [sys.executable, "-m", "verl_tinker.start", "--config", config_path], stdout=log, stderr=subprocess.STDOUT
    )
    h = Head(f"http://127.0.0.1:{PORT}")
    try:
        _wait_ready(h, proc)
        yield h
    finally:
        try:
            h.post("/api/v1/shutdown", {})
        except Exception:
            pass
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=60)
        log.close()
        print(f"server log: {workdir / 'server.log'}")


@pytest.fixture(scope="module")
def model(head):
    session_id = head.post("/api/v1/create_session", {})["session_id"]
    model_name = head.post("/api/v1/get_server_capabilities", {})["supported_models"][0]["model_name"]
    envelope = head.post(
        "/api/v1/create_model",
        {"session_id": session_id, "model_seq_id": 0, "base_model": model_name, "lora_config": {"rank": LORA_RANK}},
    )
    head.result(envelope)
    return {"session_id": session_id, "model_id": envelope["model_id"], "base_model": model_name}


# ==================== datum helpers (Tinker wire JSON) ====================


def _tensor(values, dtype="float32"):
    values = list(values)
    return {"data": values, "dtype": dtype, "shape": [len(values)]}


def _datum(tokens: list[int], **loss_fn_inputs) -> dict:
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": {"target_tokens": _tensor(tokens[1:], "int64"), **loss_fn_inputs},
    }


PROMPT = list(range(1000, 1012))
COMPLETION = [2000 + 7 * i for i in range(10)]
FULL = PROMPT + COMPLETION
N = len(FULL) - 1
N_PREFIX = len(PROMPT) - 1


def _ce(weights=None):
    return _datum(FULL, weights=_tensor(weights if weights is not None else [0.0] * N_PREFIX + [1.0] * len(COMPLETION)))


def _forward(head, model, datum, loss_fn="cross_entropy", config=None):
    return head.call(
        "/api/v1/forward",
        {
            "forward_input": {"data": [datum], "loss_fn": loss_fn, "loss_fn_config": config},
            "model_id": model["model_id"],
        },
    )


def _forward_backward(head, model, datum, loss_fn="cross_entropy", config=None):
    return head.call(
        "/api/v1/forward_backward",
        {
            "forward_backward_input": {"data": [datum], "loss_fn": loss_fn, "loss_fn_config": config},
            "model_id": model["model_id"],
        },
    )


def _optim(head, model, *, lr=0.0, eps=1e-12, beta1=0.9, grad_clip_norm=0.0):
    return head.call(
        "/api/v1/optim_step",
        {
            "model_id": model["model_id"],
            "adam_params": {
                "learning_rate": lr,
                "beta1": beta1,
                "beta2": 0.95,
                "eps": eps,
                "weight_decay": 0.0,
                "grad_clip_norm": grad_clip_norm,
            },
        },
    )


def _logprobs(result) -> torch.Tensor:
    return torch.tensor(result["loss_fn_outputs"][0]["logprobs"]["data"], dtype=torch.float64)


# ==================== checks ====================


def test_lora_metadata_and_rank_conflict(head, model):
    info = head.post("/api/v1/get_info", {"model_id": model["model_id"]})
    assert info["is_lora"] is True and info["lora_rank"] == LORA_RANK, info
    assert info["train_attn"] is True and info["train_mlp"] is True
    response = head.post(
        "/api/v1/create_model",
        {
            "session_id": model["session_id"],
            "model_seq_id": 1,
            "base_model": model["base_model"],
            "lora_config": {"rank": LORA_RANK * 2},
        },
        expect_status=409,
    )
    assert "rank" in json.dumps(response)


def test_forward_matches_forward_backward_and_has_no_grad_side_effects(head, model):
    datum = _ce()
    fwd = _forward(head, model, datum)
    fb = _forward_backward(head, model, datum)
    torch.testing.assert_close(_logprobs(fwd), _logprobs(fb), rtol=0, atol=5e-2)
    weights = torch.tensor(datum["loss_fn_inputs"]["weights"]["data"], dtype=torch.float64)
    assert fwd["metrics"]["loss:sum"] == pytest.approx(-(_logprobs(fwd) * weights).sum().item(), rel=1e-4)
    assert fb["metrics"]["loss:sum"] == pytest.approx(fwd["metrics"]["loss:sum"], rel=TOL)

    # Gradients accumulated by forward_backward survive forward() calls.
    _forward(head, model, datum)
    _forward(head, model, datum)
    with_forwards = _optim(head, model, lr=0.0)["metrics"]["grad_norm:mean"]
    _forward_backward(head, model, datum)
    control = _optim(head, model, lr=0.0)["metrics"]["grad_norm:mean"]
    assert with_forwards > 0
    assert with_forwards == pytest.approx(control, rel=TOL)
    # optim_step consumed the gradients: a forward alone does not create new ones.
    _forward(head, model, datum)
    assert _optim(head, model, lr=0.0)["metrics"].get("grad_norm:mean", 0.0) == pytest.approx(0.0, abs=1e-8)


def _rl_datum(sampling_logprobs, advantages, ratio_mask=None):
    inputs = {
        "logprobs": _tensor([0.0] * N_PREFIX + list(sampling_logprobs)),
        "advantages": _tensor([0.0] * N_PREFIX + list(advantages)),
    }
    if ratio_mask is not None:
        inputs["sequence_ratio_mask"] = _tensor([0.0] * N_PREFIX + list(ratio_mask))
    return _datum(FULL, **inputs)


@pytest.mark.parametrize(
    ("loss_fn", "config"),
    [
        ("importance_sampling", None),
        ("ppo", None),
        ("cispo", None),
        ("dro", {"beta": 0.1}),
        ("gspo", None),
        ("gspo", {"clip_low_threshold": 0.5, "clip_high_threshold": 2.0}),
    ],
)
def test_rl_loss_sum_matches_tinker_formula_on_returned_logprobs(head, model, loss_fn, config):
    from verl_tinker.tinker_losses import (
        GSPO_CLIPPED_FRACTION_METRIC,
        GSPO_LOG_RATIO_METRIC,
        loss_metrics_from_logprobs,
        normalize_loss_spec,
    )

    current = _logprobs(_forward(head, model, _ce()))[N_PREFIX:]
    g = torch.Generator().manual_seed(0)
    sampling = (current + 0.05 * torch.randn(len(COMPLETION), generator=g, dtype=torch.float64)).tolist()
    advantages = torch.randn(len(COMPLETION), generator=g).tolist()
    datum = _rl_datum(sampling, advantages, ratio_mask=[1.0] * len(COMPLETION) if loss_fn == "gspo" else None)

    result = _forward_backward(head, model, datum, loss_fn, config)
    _optim(head, model, lr=0.0)  # drop the gradients again

    spec = normalize_loss_spec(loss_fn, config)

    class _D:  # minimal Datum view for the CPU formula
        def __init__(self, wire):
            self.loss_fn_inputs = {
                key: type("T", (), {"data": value["data"], "dtype": value["dtype"], "shape": value["shape"]})()
                for key, value in wire["loss_fn_inputs"].items()
            }

    expected = loss_metrics_from_logprobs(spec, [_D(datum)], [_logprobs(result).float()])
    assert result["metrics"]["loss:sum"] == pytest.approx(expected["loss:sum"], rel=TOL, abs=1e-3)
    if loss_fn == "gspo":
        assert result["metrics"][GSPO_LOG_RATIO_METRIC] == pytest.approx(expected[GSPO_LOG_RATIO_METRIC], abs=5e-3)
        assert result["metrics"][GSPO_CLIPPED_FRACTION_METRIC] == expected[GSPO_CLIPPED_FRACTION_METRIC]


def test_grad_clip_norm_scales_the_update(head, model):
    """With beta1=0 and eps >> |g| an AdamW step is lr * g / eps, so clipping is visible in the update.

    (A normal Adam step is invariant to gradient scale, which would hide clipping.)
    """
    datum = _ce()
    probe = _ce([0.0] * N_PREFIX + [1.0] * len(COMPLETION))
    base = head.call("/api/v1/save_weights", {"model_id": model["model_id"], "path": "align-base"})["path"]
    before = _logprobs(_forward(head, model, probe))

    _forward_backward(head, model, datum)
    norm = _optim(head, model, lr=0.0)["metrics"]["grad_norm:mean"]
    assert norm > 0 and math.isfinite(norm)

    lr = 1e3 / norm  # ||update|| = lr * ||g|| / eps = 1e-3 without clipping: small, first-order regime

    def delta(grad_clip_norm: float) -> float:
        head.call("/api/v1/load_weights", {"model_id": model["model_id"], "path": base, "optimizer": True})
        _forward_backward(head, model, datum)
        step = _optim(head, model, lr=lr, eps=1e6, beta1=0.0, grad_clip_norm=grad_clip_norm)
        reported = step["metrics"]["grad_norm:mean"]
        assert reported == pytest.approx(norm, rel=TOL)  # pre-clip norm, whatever the clip
        return (_logprobs(_forward(head, model, probe)) - before).abs().sum().item()

    unclipped = delta(0.0)
    clipped = delta(0.1 * norm)
    head.call("/api/v1/load_weights", {"model_id": model["model_id"], "path": base, "optimizer": True})
    assert unclipped > 0
    assert 0.03 < clipped / unclipped < 0.3, (clipped, unclipped)


def test_topk_sample_logprobs_and_target_prompt_logprobs(head, model):
    saved = head.call(
        "/api/v1/save_weights_for_sampler",
        {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 0},
    )
    sampler = saved["sampling_session_id"]
    body = {
        "sampling_session_id": sampler,
        "seq_id": 0,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": PROMPT}]},
        "sampling_params": {"max_tokens": 8, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "num_samples": 2,
        "topk_sample_logprobs": 5,
    }
    result = head.call("/api/v1/asample", body)
    matched = 0
    for sequence in result["sequences"]:
        rows = sequence["topk_logprobs"]
        assert len(rows) == len(sequence["tokens"])
        for token, logprob, row in zip(sequence["tokens"], sequence["logprobs"], rows, strict=True):
            if row is None:
                continue
            assert 1 <= len(row) <= 5
            values = [lp for _, lp in row]
            assert values == sorted(values, reverse=True)
            for candidate, candidate_lp in row:
                if candidate == token:
                    matched += 1
                    assert candidate_lp == pytest.approx(logprob, abs=0.1)
    assert matched > 0

    rejected = head.post(
        "/api/v1/asample",
        {
            **body,
            "topk_sample_logprobs": 0,
            "target_prompt_logprobs": {
                "data": [1] * (len(PROMPT) - 1),
                "dtype": "int64",
                "shape": [len(PROMPT) - 1, 1],
            },
        },
        expect_status=400,
    )
    assert rejected["detail"]["code"] == "TARGET_PROMPT_LOGPROBS_UNSUPPORTED"
