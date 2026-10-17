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

"""GPU validation of ``server.rollout_start: lazy`` against a real server.

Same harness as ``test_envloop_alignment_gpu.py`` (skipped without CUDA)::

    # a) self-started small LoRA server with rollout_start=lazy
    TINKER_ALIGN_MODEL=Qwen/Qwen3-0.6B python -m pytest -q -s tests/gpu/test_envloop_lazy_rollout_gpu.py

    # b) EnvLoop's trainingmaster-actor-rollout.yaml + production env, TINKER_ROLLOUT_START=lazy
    TINKER_ALIGN_CONFIG=/path/config.yaml python -m pytest -q -s tests/gpu/test_envloop_lazy_rollout_gpu.py

``TINKER_LAZY_TRAIN_ONLY=1`` stops after the training stage (e.g. Qwen3.8-27B on 2x 80 GB, where the
FSDP1 rollout sync does not fit). ``TINKER_LAZY_REPORT=/path.json`` writes timings and per-GPU memory.

Checks: the server becomes ready without vLLM (healthz ``rollout: not_started``); scalar compute_logprobs
and every training call leave it unstarted; the first save_weights_for_sampler starts it inside its future
(``try_again`` while vLLM starts); the sampler then decodes, serves compute_logprobs identical to the
sample's prompt logprobs, agrees with the trainer, and keeps working across another train/sync cycle.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
from test_envloop_alignment_gpu import (
    _CONFIG_TEMPLATE,
    FULL,
    LORA_RANK,
    MODEL,
    PORT,
    PROMPT,
    STRATEGY,
    Head,
    _ce,
    _forward,
    _forward_backward,
    _logprobs,
    _optim,
    _prepare_fsdp2_config,
    _wait_ready,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="lazy rollout GPU validation needs CUDA")

TRAIN_ONLY = os.environ.get("TINKER_LAZY_TRAIN_ONLY", "") == "1"
REPORT_PATH = os.environ.get("TINKER_LAZY_REPORT")


def _gpu_memory_used_mib() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [int(line) for line in out.split()]


class _MemoryPeak:
    """Per-GPU peak of ``nvidia-smi`` memory.used while a stage runs."""

    def __init__(self):
        self.peak = _gpu_memory_used_mib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.wait(0.5):
            self.peak = [max(a, b) for a, b in zip(self.peak, _gpu_memory_used_mib(), strict=True)]

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


@pytest.fixture(scope="module")
def head():
    url = os.environ.get("TINKER_ALIGN_SERVER_URL")
    if url:
        h = Head(url)
        _wait_ready(h, None)
        yield h
        return

    workdir = Path(tempfile.mkdtemp(prefix="envloop-lazy-"))
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
                layered_summon="false" if STRATEGY == "fsdp2" else "true",
            ).replace("  enable_offload: true\n", "  enable_offload: true\n  rollout_start: lazy\n")
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


def _rollout_state(head) -> str:
    return head.get("/api/v1/healthz").get("rollout")


def _asample(head, sampler: str, tokens: list[int], **overrides) -> dict:
    body = {
        "sampling_session_id": sampler,
        "seq_id": 0,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "num_samples": 1,
        **overrides,
    }
    return head.call("/api/v1/asample", body)


def _compute_logprobs(head, sampler: str) -> torch.Tensor:
    result = _asample(head, sampler, FULL, prompt_logprobs=True)
    return torch.tensor(result["prompt_logprobs"][1:], dtype=torch.float64)


def _timed_future(head, path: str, body: dict) -> tuple[dict, float, int]:
    """Result, seconds, and how many ``try_again`` polls the client saw."""
    started = time.monotonic()
    envelope = head.post(path, body)
    polls = 0
    while True:
        payload = head.post("/api/v1/retrieve_future", {"request_id": envelope["request_id"]})
        if payload.get("type") != "try_again":
            assert "error" not in payload, payload
            return payload, time.monotonic() - started, polls
        polls += 1


def test_lazy_rollout_lifecycle(head):
    report: dict = {"model": MODEL, "train_only": TRAIN_ONLY, "idle_mib": _gpu_memory_used_mib()}
    assert _rollout_state(head) == "not_started", head.get("/api/v1/healthz")

    session_id = head.post("/api/v1/create_session", {})["session_id"]
    base_model = head.post("/api/v1/get_server_capabilities", {})["supported_models"][0]["model_name"]
    envelope = head.post(
        "/api/v1/create_model",
        {"session_id": session_id, "model_seq_id": 0, "base_model": base_model, "lora_config": {"rank": LORA_RANK}},
    )
    head.result(envelope)
    model = {"session_id": session_id, "model_id": envelope["model_id"]}

    # compute_logprobs on the base model before anything needs vLLM: served by the actor.
    base_sampler = head.post(
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": base_model, "model_path": None},
    )["sampling_session_id"]
    base_lp = _compute_logprobs(head, base_sampler)
    trainer_base_lp = _logprobs(_forward(head, model, _ce()))
    report["base_compute_logprobs_vs_trainer_max_abs"] = (base_lp - trainer_base_lp).abs().max().item()
    assert report["base_compute_logprobs_vs_trainer_max_abs"] < 1e-2
    assert _rollout_state(head) == "not_started"

    with _MemoryPeak() as train:
        for _ in range(2):
            _forward_backward(head, model, _ce())
            _optim(head, model, lr=1e-4)
        _forward(head, model, _ce())
        head.call("/api/v1/save_weights", {"model_id": model["model_id"], "path": "lazy-state"})
    report["train_peak_mib"] = train.peak
    assert _rollout_state(head) == "not_started", "training must never start the rollout"
    print(f"training only: peak per-GPU memory {train.peak} MiB, rollout still not started")

    if not TRAIN_ONLY:
        with _MemoryPeak() as start:
            saved, seconds, polls = _timed_future(
                head,
                "/api/v1/save_weights_for_sampler",
                {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 1},
            )
        report.update(first_sampler_save_s=seconds, first_sampler_save_try_again=polls, start_peak_mib=start.peak)
        assert _rollout_state(head) == "ready"
        sampler = saved["sampling_session_id"]
        print(f"first save_weights_for_sampler started vLLM in {seconds:.1f}s ({polls} try_again polls)")

        sampled = _asample(head, sampler, PROMPT, sampling_params={"max_tokens": 8, "temperature": 1.0}, num_samples=2)
        assert all(len(seq["tokens"]) > 0 for seq in sampled["sequences"]), sampled

        both = _asample(head, sampler, FULL, prompt_logprobs=True, topk_prompt_logprobs=5)
        sampler_lp = _compute_logprobs(head, sampler)
        prompt_lp = torch.tensor(both["prompt_logprobs"][1:], dtype=torch.float64)
        trainer_lp = _logprobs(_forward(head, model, _ce()))
        answer = slice(len(PROMPT) - 1, None)
        report["compute_vs_prompt_logprobs_max_abs"] = (sampler_lp - prompt_lp).abs().max().item()
        report["sampler_vs_trainer_mean_abs"] = (sampler_lp[answer] - trainer_lp[answer]).abs().mean().item()
        assert report["compute_vs_prompt_logprobs_max_abs"] < 1e-2
        assert report["sampler_vs_trainer_mean_abs"] < 0.05

        # Another train -> sync -> sample cycle over the started rollout (sleep/wake lifecycle).
        _forward_backward(head, model, _ce())
        _optim(head, model, lr=1e-4)
        saved, seconds, _ = _timed_future(
            head,
            "/api/v1/save_weights_for_sampler",
            {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 2},
        )
        report["second_sampler_save_s"] = seconds
        resampled = _asample(
            head, saved["sampling_session_id"], PROMPT, sampling_params={"max_tokens": 4, "temperature": 1.0}
        )
        assert len(resampled["sequences"][0]["tokens"]) > 0

    print(json.dumps(report, indent=2))
    if REPORT_PATH:
        Path(REPORT_PATH).write_text(json.dumps(report, indent=2))
