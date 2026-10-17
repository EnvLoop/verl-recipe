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

"""GPU validation of Tinker LoraConfig parity, started exactly like an EnvLoop guest.

The server runs EnvPlatform's guest overlay (``trainingmaster-actor-rollout.yaml``) through
its ``recipe_server.py`` with the ``TINKER_*`` environment EnvPlatform renders for one
``create_model`` (rank, alpha 32, ``TINKER_LORA_TARGET_MODULES`` regex, ``TINKER_LORA_SEED``).
One scenario per run; the caller exports the environment and describes the expectation::

    LORA_GPU_GUEST_YAML=/w/trainingmaster-actor-rollout.yaml \\
    LORA_GPU_RECIPE_SERVER=/w/recipe_server.py LORA_GPU_LORA_EXPORT=/w/lora_export.py \\
    LORA_GPU_EXPECT='{"rank": 32, "train_attn": true, "train_mlp": true, "train_unembed": true,
                      "target_modules": [...], "params_per_rank": 2155520}' \\
    TINKER_LORA_RANK=32 TINKER_LORA_ALPHA=32 TINKER_LORA_TARGET_MODULES='(?:.*\\.)?(?:...)' ... \\
        python -m pytest -q -s tests/gpu/test_envloop_lora_gpu.py

Checks: get_info reports the configured adapter and create_model with other layer flags is
a 409; the exported PEFT adapter (EnvPlatform ``lora_export``) has exactly the expected
``target_modules``, ``r``, ``lora_alpha`` and ``rank * params_per_rank`` parameters and no
vision-tower tensors; after training, the rollout's sampled-token logprobs agree with the
actor's logprobs of the same tokens (a dropped ``lm_head`` LoRA in vLLM breaks this);
``LORA_GPU_INIT_DIGEST`` prints a digest of the freshly initialised adapter (same seed ->
same digest across server runs). Peak GPU memory (nvidia-smi) is printed at the end,
including one forward_backward over ``LORA_GPU_LONG_TOKENS`` tokens.
"""

from __future__ import annotations

import hashlib
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
from test_envloop_alignment_gpu import COMPLETION, PROMPT, Head, _datum, _forward_backward, _optim, _tensor, _wait_ready

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="LoRA parity GPU validation needs CUDA")

EXPECT = json.loads(os.environ.get("LORA_GPU_EXPECT", "{}"))
PORT = int(os.environ.get("TINKER_SERVER_PORT", "18766"))
LONG_TOKENS = int(os.environ.get("LORA_GPU_LONG_TOKENS", "4000"))
FLAGS = ("train_attn", "train_mlp", "train_unembed")


class PeakMemory:
    """Per-GPU peak of nvidia-smi memory.used (MiB), sampled every 0.5 s."""

    def __init__(self):
        self.peak: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                for line in out.strip().splitlines():
                    index, used = (int(part) for part in line.split(","))
                    self.peak[index] = max(self.peak.get(index, 0), used)
            except Exception:
                pass
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return dict(self.peak)


@pytest.fixture(scope="module")
def server():
    workdir = Path(tempfile.mkdtemp(prefix="envloop-lora-"))
    env = dict(os.environ)
    env.setdefault("TINKER_CHECKPOINT_DIR", str(workdir / "checkpoints"))
    env["TINKER_SERVER_PORT"] = str(PORT)
    config = workdir / "trainingmaster-actor-rollout.yaml"
    config.write_text(Path(os.environ["LORA_GPU_GUEST_YAML"]).read_text())
    memory = PeakMemory()
    log = open(workdir / "server.log", "w")  # noqa: SIM115 - closed in teardown
    proc = subprocess.Popen(
        [sys.executable, os.environ["LORA_GPU_RECIPE_SERVER"], "--config", str(config)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    head = Head(f"http://127.0.0.1:{PORT}")
    started = time.monotonic()
    try:
        _wait_ready(head, proc)
        print(f"server ready after {time.monotonic() - started:.0f}s; log {workdir / 'server.log'}")
        yield {"head": head, "checkpoints": Path(env["TINKER_CHECKPOINT_DIR"]), "memory": memory}
    finally:
        try:
            head.post("/api/v1/shutdown", {})
        except Exception:
            pass
        try:
            proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=60)
        log.close()
        print(f"RESULT peak_memory_mib={json.dumps(memory.stop())}")


@pytest.fixture(scope="module")
def model(server):
    head = server["head"]
    session_id = head.post("/api/v1/create_session", {})["session_id"]
    base_model = head.post("/api/v1/get_server_capabilities", {})["supported_models"][0]["model_name"]
    lora_config = {"rank": EXPECT["rank"], **{flag: EXPECT[flag] for flag in FLAGS}}
    envelope = head.post(
        "/api/v1/create_model",
        {"session_id": session_id, "model_seq_id": 0, "base_model": base_model, "lora_config": lora_config},
    )
    head.result(envelope)
    return {"session_id": session_id, "model_id": envelope["model_id"], "base_model": base_model}


def _export(server, model, name: str) -> tuple[dict, dict[str, torch.Tensor]]:
    """save_weights + EnvPlatform's lora_export: (adapter_config, tensors)."""
    import importlib.util

    from safetensors.torch import load_file

    head = server["head"]
    head.call("/api/v1/save_weights", {"model_id": model["model_id"], "path": name})
    tags = [path.parent for path in server["checkpoints"].rglob("lora_train_meta.json") if name in str(path)]
    assert len(tags) == 1, tags
    spec = importlib.util.spec_from_file_location("lora_export", os.environ["LORA_GPU_LORA_EXPORT"])
    lora_export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lora_export)
    shards = lora_export._load_shards(
        tags[0], int(json.loads((tags[0] / "fsdp_config.json").read_text())["world_size"])
    )
    visual = [key for key in shards[0] if "lora_" in key and ".visual." in key]
    assert visual == [], visual[:3]
    out = lora_export.export(tags[0])
    return json.loads((out / "adapter_config.json").read_text()), load_file(str(out / "adapter_model.safetensors"))


def _exported_tensors(server, name: str) -> dict[str, torch.Tensor]:
    """The PEFT tensors of an existing checkpoint (exported with EnvPlatform's lora_export if needed)."""
    import importlib.util

    from safetensors.torch import load_file

    tags = [path.parent for path in server["checkpoints"].rglob("lora_train_meta.json") if name in str(path)]
    assert len(tags) == 1, tags
    if not (tags[0] / "lora_adapter").exists():
        spec = importlib.util.spec_from_file_location("lora_export", os.environ["LORA_GPU_LORA_EXPORT"])
        lora_export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lora_export)
        lora_export.export(tags[0])
    return load_file(str(tags[0] / "lora_adapter" / "adapter_model.safetensors"))


def test_init_digest_and_reported_adapter(server, model):
    head = server["head"]
    info = head.post("/api/v1/get_info", {"model_id": model["model_id"]})
    print(f"get_info: { {key: info.get(key) for key in ('lora_rank', 'lora_alpha', *FLAGS, 'target_modules')} }")
    assert info["lora_rank"] == EXPECT["rank"] and info["lora_alpha"] == 32.0
    assert {flag: info[flag] for flag in FLAGS} == {flag: EXPECT[flag] for flag in FLAGS}
    assert info["target_modules"] == EXPECT["target_modules"]

    other = {flag: not EXPECT[flag] if flag == "train_unembed" else EXPECT[flag] for flag in FLAGS}
    if any(other.values()):
        response = head.post(
            "/api/v1/create_model",
            {
                "session_id": model["session_id"],
                "model_seq_id": 1,
                "base_model": model["base_model"],
                "lora_config": {"rank": EXPECT["rank"], **other},
            },
            expect_status=409,
        )
        print(f"create_model with other layers: HTTP 409 {str(response)[:160]}")

    config, tensors = _export(server, model, "lora-init")
    digest = hashlib.sha256()
    for key in sorted(tensors):
        if ".lora_A." in key:
            digest.update(key.encode())
            digest.update(tensors[key].float().cpu().numpy().tobytes())
    print(f"RESULT init_digest={digest.hexdigest()} seed={os.environ.get('TINKER_LORA_SEED')}")
    assert all(not tensors[key].any() for key in tensors if ".lora_B." in key)


def test_exported_adapter_matches_tinker_layers_and_param_count(server, model):
    config, tensors = _export(server, model, "lora-shape")
    numel = sum(tensor.numel() for tensor in tensors.values())
    ranks = {tensor.shape[0] for key, tensor in tensors.items() if ".lora_A." in key}
    print(
        f"RESULT adapter target_modules={config['target_modules']} r={config['r']} alpha={config['lora_alpha']} "
        f"params={numel} expected={EXPECT['rank'] * EXPECT['params_per_rank']} tensors={len(tensors)}"
    )
    assert config["target_modules"] == EXPECT["target_modules"]
    assert (config["r"], config["lora_alpha"], ranks) == (EXPECT["rank"], 32, {EXPECT["rank"]})
    assert numel == EXPECT["rank"] * EXPECT["params_per_rank"]


def _actor_logprobs(head, model, tokens: list[int]) -> torch.Tensor:
    datum = _datum(tokens, weights=_tensor([1.0] * (len(tokens) - 1)))
    result = head.call(
        "/api/v1/forward",
        {
            "forward_input": {"data": [datum], "loss_fn": "cross_entropy", "loss_fn_config": None},
            "model_id": model["model_id"],
        },
    )
    return torch.tensor(result["loss_fn_outputs"][0]["logprobs"]["data"], dtype=torch.float64)


def _sampler_vs_actor(head, model, seq: int) -> dict:
    saved = head.call(
        "/api/v1/save_weights_for_sampler",
        {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": seq},
    )
    result = head.call(
        "/api/v1/asample",
        {
            "sampling_session_id": saved["sampling_session_id"],
            "seq_id": 0,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": PROMPT}]},
            "sampling_params": {"max_tokens": 16, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
            "num_samples": 4,
        },
    )
    diffs, targeted = [], []
    for sequence in result["sequences"]:
        tokens = PROMPT + sequence["tokens"]
        actor = _actor_logprobs(head, model, tokens)[len(PROMPT) - 1 :]
        sampled = torch.tensor(sequence["logprobs"], dtype=torch.float64)
        diffs.append((actor - sampled).abs())
        targeted.extend(sequence["tokens"])
    diff = torch.cat(diffs)
    return {
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "completion_token_share": sum(t in COMPLETION for t in targeted) / max(1, len(targeted)),
    }


def test_rollout_samples_from_the_trained_adapter(server, model):
    head = server["head"]
    before = _sampler_vs_actor(head, model, 10)
    # Teach the adapter COMPLETION after PROMPT (strong enough to move every adapted layer family).
    datum = _datum(PROMPT + COMPLETION, weights=_tensor([0.0] * (len(PROMPT) - 1) + [1.0] * len(COMPLETION)))
    losses = []
    for _ in range(int(os.environ.get("LORA_GPU_TRAIN_STEPS", "8"))):
        losses.append(_forward_backward(head, model, datum)["metrics"]["loss:sum"])
        _optim(head, model, lr=float(os.environ.get("LORA_GPU_LR", "1e-3")), eps=1e-8)
    after = _sampler_vs_actor(head, model, 11)
    print(f"RESULT losses={[round(x, 3) for x in losses]} sampler_vs_actor before={before} after={after}")
    assert losses[-1] < 0.5 * losses[0], losses
    dump = os.environ.get("LORA_GPU_DUMP")
    if dump:
        # The trained adapter as EnvPlatform publishes it, plus the actor's logprobs, so a
        # serving engine can be checked against the trainer (e.g. vLLM 0.31 with this adapter).
        _export(server, model, "lora-final")
        tags = [
            path.parent for path in server["checkpoints"].rglob("lora_train_meta.json") if "lora-final" in str(path)
        ]
        Path(dump).write_text(
            json.dumps(
                {
                    "adapter": str(tags[0] / "lora_adapter"),
                    "tokens": PROMPT + COMPLETION,
                    "actor_logprobs": _actor_logprobs(head, model, PROMPT + COMPLETION).tolist(),
                }
            )
        )
    # Same weights on both sides: only bf16 / kernel noise. A rollout that dropped part of the
    # adapter (e.g. the lm_head LoRA) samples from a different distribution.
    assert after["mean_abs"] < max(0.05, 3 * before["mean_abs"]), (before, after)


def test_prompt_logprobs_match_the_actor(server, model):
    """compute_logprobs-style request (prompt logprobs on the rollout) agrees with the actor.

    Runs after training, so the adapter (incl. an lm_head LoRA) is far from zero; vLLM without the
    #62049 backport scores prompt rows with the per-request mapping and misses the lm_head delta.
    """
    head = server["head"]
    saved = head.call(
        "/api/v1/save_weights_for_sampler",
        {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 12},
    )
    tokens = PROMPT + COMPLETION
    result = head.call(
        "/api/v1/asample",
        {
            "sampling_session_id": saved["sampling_session_id"],
            "seq_id": 0,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
            "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
            "num_samples": 1,
            "prompt_logprobs": True,
        },
    )
    sampler = torch.tensor(result["prompt_logprobs"][1:], dtype=torch.float64)
    actor = _actor_logprobs(head, model, tokens)
    diff = (sampler - actor).abs()
    print(f"RESULT prompt_logprobs_vs_actor mean_abs={diff.mean().item():.4f} max_abs={diff.max().item():.4f}")
    assert diff.mean().item() < 0.05


def test_long_forward_backward_memory(server, model):
    head = server["head"]
    tokens = [1000 + (i * 7919) % 50000 for i in range(LONG_TOKENS + 1)]
    datum = _datum(tokens, weights=_tensor([1.0] * LONG_TOKENS))
    started = time.monotonic()
    _forward_backward(head, model, datum)
    _optim(head, model, lr=0.0)
    print(
        f"RESULT long_fwd_bwd tokens={LONG_TOKENS} seconds={time.monotonic() - started:.1f} "
        f"peak_so_far_mib={json.dumps(server['memory'].peak)}"
    )


def test_legacy_fused_checkpoint_round_trip(server, model):
    """LORA_GPU_LEGACY=save on a server with ``lora.split_qkv: false`` writes a fused-q/k/v checkpoint and its
    logprobs; LORA_GPU_LEGACY=load on a split server (same checkpoint dir) resumes it and must reproduce them."""
    mode = os.environ.get("LORA_GPU_LEGACY")
    if not mode:
        pytest.skip("LORA_GPU_LEGACY not set")
    head = server["head"]
    record = Path(os.environ.get("LORA_GPU_LEGACY_RECORD", "/w/dump/legacy.json"))
    tokens = PROMPT + COMPLETION
    if mode == "save":
        datum = _datum(tokens, weights=_tensor([0.0] * (len(PROMPT) - 1) + [1.0] * len(COMPLETION)))
        for _ in range(4):
            _forward_backward(head, model, datum)
            _optim(head, model, lr=1e-3, eps=1e-8)
        path = head.call("/api/v1/save_weights", {"model_id": model["model_id"], "path": "legacy-ckpt"})["path"]
        logprobs = _actor_logprobs(head, model, tokens).tolist()
        _forward_backward(head, model, datum)  # the step a resume with optimizer state must reproduce
        _optim(head, model, lr=1e-3, eps=1e-8)
        _, after_step = _export(server, model, "legacy-next")
        torch.save(after_step, record.with_suffix(".next.pt"))
        record.write_text(json.dumps({"path": path, "logprobs": logprobs}))
        print(f"RESULT legacy_saved path={path}")
        return
    saved = json.loads(record.read_text())
    before = _actor_logprobs(head, model, tokens)
    head.call("/api/v1/load_weights", {"model_id": model["model_id"], "path": saved["path"], "optimizer": True})
    after = _actor_logprobs(head, model, tokens)
    expected = torch.tensor(saved["logprobs"], dtype=torch.float64)
    diff = (after - expected).abs()
    print(
        f"RESULT legacy_resume mean_abs={diff.mean().item():.5f} max_abs={diff.max().item():.5f} "
        f"untrained_vs_saved={(before - expected).abs().mean().item():.4f}"
    )
    assert diff.max().item() < 0.05
    # Resumed with the converted Adam state, the next step moves every tensor but the GDN A's exactly like
    # the fused layout did (A's get per-slice gradients by design).
    datum = _datum(tokens, weights=_tensor([0.0] * (len(PROMPT) - 1) + [1.0] * len(COMPLETION)))
    _forward_backward(head, model, datum)
    _optim(head, model, lr=1e-3, eps=1e-8)
    _, split_next = _export(server, model, "split-next")
    fused_next = torch.load(record.with_suffix(".next.pt"))
    fused_before = _exported_tensors(server, "legacy-ckpt")
    errors = []
    for name, fused in fused_next.items():
        if ".in_proj_qkv.lora_A." in name:  # A's get per-slice gradients by design
            continue
        if ".in_proj_qkv.lora_B." in name:
            prefix = name.split(".in_proj_qkv.")[0]
            split = torch.cat([split_next[f"{prefix}.in_proj_{s}.lora_B.weight"] for s in "qkv"])
        else:
            split = split_next[name]
        before = fused_before[name].float()
        fused_step, split_step = fused.float() - before, split.float() - before
        errors.append(((split_step - fused_step).norm() / fused_step.norm().clamp_min(1e-12)).item())
    errors.sort()
    median, worst = errors[len(errors) // 2], errors[-1]
    print(f"RESULT legacy_next_step tensors={len(errors)} update_rel_err median={median:.3f} max={worst:.3f}")
    # bf16 noise is ~1-3%; an Adam step without the restored moments differs by O(1).
    assert median < 0.03 and worst < 0.1
    # The rollout gets the converted adapter too.
    sampled = _sampler_vs_actor(head, model, 20)
    print(f"RESULT legacy_resume_sampler_vs_actor {sampled}")
    assert sampled["mean_abs"] < 0.05
