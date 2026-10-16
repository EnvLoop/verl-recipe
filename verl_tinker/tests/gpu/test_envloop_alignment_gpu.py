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

    # a) let the test start a small LoRA server itself (default Qwen/Qwen3-0.6B, rank 8, all local GPUs,
    #    FSDP2 actor prepared like EnvLoop's guest; TINKER_ALIGN_STRATEGY=fsdp selects FSDP1)
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

Known pre-existing limitation (not caused by the EnvLoop changes; reproduced on
upstream ``242c73fc``): with the FSDP1 actor (``strategy: fsdp``) and a model
with tied input/output embeddings such as Qwen3-0.6B, a ``forward`` issued
after ``forward_backward`` and before ``optim_step`` fails inside FSDP1
(``as_params=True type(prim_param)=<class 'torch.Tensor'>``) and the actor is
unusable afterwards (``Changing shared parameters is not supported yet``).
FSDP2 (EnvLoop's Qwen3.5-4B profile) is not affected, so the self-started
server defaults to FSDP2.
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
STRATEGY = os.environ.get("TINKER_ALIGN_STRATEGY", "fsdp2")

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
    strategy: {strategy}
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
    layered_summon: {layered_summon}
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


def _prepare_fsdp2_config(path: Path) -> str:
    """FSDP2 actor config, prepared like EnvLoop's guest ``recipe_server.prepare_config``.

    The server's defaults merge only knows ``fsdp``; merge with it, then select
    the FSDP2 engine and disable the second merge at startup.
    """
    from omegaconf import OmegaConf
    from verl_tinker.config_utils import process_actor_rollout_ref_config

    config = OmegaConf.load(path)
    config = process_actor_rollout_ref_config(config)
    config.actor_rollout_ref.actor.strategy = "fsdp2"
    config.actor_rollout_ref.actor.fsdp_config.strategy = "fsdp2"
    config.server.auto_merge_verl_default_config = False
    OmegaConf.resolve(config)
    out = path.with_suffix(".fsdp2.yaml")
    OmegaConf.save(config, out)
    return str(out)


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
                strategy="fsdp" if STRATEGY == "fsdp2" else STRATEGY,
                # FSDP2 LoRA sync needs the base model preloaded in vLLM for layered summon;
                # EnvLoop runs it with TINKER_ROLLOUT_LAYERED_SUMMON=false.
                layered_summon="false" if STRATEGY == "fsdp2" else "true",
            )
        )
        if STRATEGY == "fsdp2":
            config_path = _prepare_fsdp2_config(Path(config_path))
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
    print(f"get_info: {info}")
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
    print(f"create_model rank {LORA_RANK * 2}: HTTP 409 {response}")


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
    with_forwards = _optim(head, model, lr=0.0)["metrics"]["unclipped_grad_l2:mean"]
    _forward_backward(head, model, datum)
    control = _optim(head, model, lr=0.0)["metrics"]["unclipped_grad_l2:mean"]
    print(
        f"forward: loss:sum fwd={fwd['metrics']['loss:sum']} fb={fb['metrics']['loss:sum']} "
        f"max|lp_fwd-lp_fb|={(_logprobs(fwd) - _logprobs(fb)).abs().max().item():.3g}; "
        f"grad_norm after fb+2 forwards={with_forwards} vs fb only={control}"
    )
    assert with_forwards > 0
    assert with_forwards == pytest.approx(control, rel=TOL)
    # optim_step consumed the gradients: a forward alone does not create new ones,
    # and a step without gradients leaves the weights bit-identical (Tinker e2e).
    before = _logprobs(_forward(head, model, datum))
    assert _optim(head, model, lr=1e-3)["metrics"].get("unclipped_grad_l2:mean", 0.0) == pytest.approx(0.0, abs=1e-8)
    assert torch.equal(_logprobs(_forward(head, model, datum)), before)


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
    print(f"{loss_fn} {config}: server {result['metrics']} expected {expected}")
    assert result["metrics"]["loss:sum"] == pytest.approx(expected["loss:sum"], rel=TOL, abs=1e-3)
    if loss_fn == "gspo":
        assert result["metrics"][GSPO_LOG_RATIO_METRIC] == pytest.approx(expected[GSPO_LOG_RATIO_METRIC], abs=5e-3)
        assert result["metrics"][GSPO_CLIPPED_FRACTION_METRIC] == expected[GSPO_CLIPPED_FRACTION_METRIC]


def test_grad_clip_norm_scales_the_update(head, model):
    """With beta1=0 and eps >> |g| an AdamW step is lr * g / eps, so clipping is visible in the update.

    (A normal Adam step is invariant to gradient scale, which would hide clipping.)
    Clipping the gradient to 0.1 * ||g|| must therefore give the same update as
    an unclipped step with lr / 10. The probe logprobs are compared against that
    reference rather than by magnitude ratio: the bf16 forward has a noise floor
    (~0.5 summed over the 10 probe tokens on Qwen3-0.6B / H100 once the zero-init
    LoRA B becomes nonzero), so the update must be well above it (||update|| = 1
    here; at 1e-3 a 10x smaller update changed the probe just as much).
    """
    datum = _ce()
    probe = _ce([0.0] * N_PREFIX + [1.0] * len(COMPLETION))
    base = head.call("/api/v1/save_weights", {"model_id": model["model_id"], "path": "align-base"})["path"]
    before = _logprobs(_forward(head, model, probe))

    _forward_backward(head, model, datum)
    norm = _optim(head, model, lr=0.0)["metrics"]["unclipped_grad_l2:mean"]
    assert norm > 0 and math.isfinite(norm)

    lr = 1e6 / norm  # ||update|| = lr * ||g|| / eps = 1 without clipping

    def delta(step_lr: float, grad_clip_norm: float) -> torch.Tensor:
        head.call("/api/v1/load_weights", {"model_id": model["model_id"], "path": base, "optimizer": True})
        assert torch.equal(_logprobs(_forward(head, model, probe)), before)  # exact restore
        _forward_backward(head, model, datum)
        step = _optim(head, model, lr=step_lr, eps=1e6, beta1=0.0, grad_clip_norm=grad_clip_norm)
        reported = step["metrics"]["unclipped_grad_l2:mean"]
        assert reported == pytest.approx(norm, rel=TOL)  # pre-clip norm, whatever the clip
        return _logprobs(_forward(head, model, probe)) - before

    unclipped = delta(lr, 0.0)
    clipped = delta(lr, 0.1 * norm)
    reference = delta(lr / 10, 0.0)  # what a 0.1 clip must produce
    inactive = delta(lr, 10 * norm)  # clip above the norm: no-op
    head.call("/api/v1/load_weights", {"model_id": model["model_id"], "path": base, "optimizer": True})
    summary = {
        "unclipped": unclipped.abs().sum().item(),
        "clipped": clipped.abs().sum().item(),
        "reference_lr_div_10": reference.abs().sum().item(),
        "clipped_vs_reference": (clipped - reference).abs().sum().item(),
        "unclipped_vs_reference": (unclipped - reference).abs().sum().item(),
        "inactive_vs_unclipped": (inactive - unclipped).abs().sum().item(),
    }
    print(f"grad_clip_norm: norm={norm} {summary}")
    assert torch.isfinite(unclipped).all() and summary["unclipped"] > 0
    assert summary["clipped_vs_reference"] < 0.25 * summary["unclipped_vs_reference"], summary
    assert summary["inactive_vs_unclipped"] < 0.25 * summary["unclipped_vs_reference"], summary
    assert summary["clipped"] < 0.5 * summary["unclipped"], summary


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
            assert len(row) == 5
            values = [lp for _, lp in row]
            assert values == sorted(values, reverse=True)
            for candidate, candidate_lp in row:
                if candidate == token:
                    matched += 1
                    assert candidate_lp == logprob  # T=1: pinned to the sampled logprob, as on Tinker
    print(
        "topk_sample_logprobs=5: "
        + "; ".join(
            f"seq{i}: {len(seq['tokens'])} tokens, {sum(r is not None for r in seq['topk_logprobs'])} rows, "
            f"row widths {sorted({len(r) for r in seq['topk_logprobs'] if r is not None})}"
            for i, seq in enumerate(result["sequences"])
        )
        + f"; sampled-token entries pinned: {matched}"
    )
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
    print(f"target_prompt_logprobs: HTTP 400 {rejected['detail']['code']}")


def test_sample_prompt_logprobs_together_with_topk_prompt_logprobs(head, model):
    """``include_prompt_logprobs`` + ``topk_prompt_logprobs`` returns both fields (Tinker behaviour).

    The scalar ``prompt_logprobs`` must agree with what the SDK's
    ``compute_logprobs`` request (one sample, ``max_tokens=1``, prompt logprobs
    only) returns, and every top-k row after the first position has k entries.
    """
    saved = head.call(
        "/api/v1/save_weights_for_sampler",
        {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 1},
    )
    body = {
        "sampling_session_id": saved["sampling_session_id"],
        "seq_id": 0,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": FULL}]},
        "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "num_samples": 1,
        "prompt_logprobs": True,
        "topk_prompt_logprobs": 5,
    }
    both = head.call("/api/v1/asample", body)
    compute_logprobs = head.call("/api/v1/asample", {**body, "topk_prompt_logprobs": 0})

    prompt_lp, topk = both["prompt_logprobs"], both["topk_prompt_logprobs"]
    reference = compute_logprobs["prompt_logprobs"]
    assert prompt_lp is not None and topk is not None, both
    assert len(prompt_lp) == len(topk) == len(reference) == len(FULL)
    assert prompt_lp[0] is None and reference[0] is None
    widths = sorted({len(row) for row in topk[1:]})
    diff = (torch.tensor(prompt_lp[1:], dtype=torch.float64) - torch.tensor(reference[1:], dtype=torch.float64)).abs()
    print(
        f"prompt_logprobs+topk_prompt_logprobs=5: {len(prompt_lp)} positions, topk widths {widths}, "
        f"max|prompt_logprobs - compute_logprobs|={diff.max().item():.3g}"
    )
    assert widths == [5]
    assert diff.max().item() < 1e-2
