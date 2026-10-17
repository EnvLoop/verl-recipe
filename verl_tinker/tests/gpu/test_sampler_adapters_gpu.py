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

"""GPU validation: sampling saved sampler checkpoints as extra LoRA adapters inside a session.

Train a few steps, save named sampler A the way the EnvLoop gateway does (``save_weights`` then
``save_weights_for_sampler``), export its PEFT adapter like the guest sync loop (EnvPlatform
``guest/lora_export.py``, path in ``TINKER_SAMPLER_LORA_EXPORT``), keep training, then check:

* A sampled later (rollout has moved on -> adapter fallback) reproduces A's save-time rollout
  prompt logprobs and seeded samples;
* A staged under ``<checkpoint_dir>/<tag>/lora_adapter`` (the gateway's cross-session path) and bound
  as ``sampler-adapter://<tag>`` gives the same values;
* the current weights' sampler and the actor are unaffected (identical before/after), requests for
  different adapters running concurrently each get their own weights, LRU eviction reloads
  correctly, and training continues.

Run on a GPU host from ``verl_tinker`` inside the recipe venv, like ``test_envloop_alignment_gpu.py``
(same ``TINKER_ALIGN_*`` variables; the server config needs ``rollout.engine_kwargs.vllm.max_loras: 3``)::

    TINKER_SAMPLER_LORA_EXPORT=/path/lora_export.py TINKER_ALIGN_CONFIG=/path/config.yaml \\
        python -m pytest -q -s tests/gpu/test_sampler_adapters_gpu.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from test_envloop_alignment_gpu import (  # noqa: F401 - the head fixture is shared
    FULL,
    PROMPT,
    _ce,
    _forward,
    _forward_backward,
    _logprobs,
    _optim,
    head,
    model,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="sampler adapter GPU validation needs CUDA")

LR = float(os.environ.get("TINKER_SAMPLER_LR", "2e-3"))
# Same weights served twice still differ by bf16 / Qwen3.5 linear-attention kernel noise (measured
# below as the repeat noise floor); training moves these logprobs by ~10.
TOL_MAX = float(os.environ.get("TINKER_SAMPLER_TOL_MAX", "0.3"))
TOL_MEAN = float(os.environ.get("TINKER_SAMPLER_TOL_MEAN", "0.05"))


def _train(server, training, steps):
    for _ in range(steps):
        _forward_backward(server, training, _ce())
        _optim(server, training, lr=LR, eps=1e-8)


def _sampler(server, session_id, seq, model_path):
    return server.post(
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "sampling_session_seq_id": seq, "model_path": model_path},
    )["sampling_session_id"]


def _score(server, sampler):
    """Rollout prompt logprobs of FULL (max_tokens=2 keeps it on the rollout, never the actor)."""
    result = server.call(
        "/api/v1/asample",
        {
            "sampling_session_id": sampler,
            "seq_id": 0,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": FULL}]},
            "sampling_params": {"max_tokens": 2, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
            "num_samples": 1,
            "prompt_logprobs": True,
        },
    )
    return torch.tensor(result["prompt_logprobs"][1:], dtype=torch.float64)


def _seeded(server, sampler):
    result = server.call(
        "/api/v1/asample",
        {
            "sampling_session_id": sampler,
            "seq_id": 0,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": PROMPT}]},
            "sampling_params": {"max_tokens": 16, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 1234},
            "num_samples": 1,
        },
    )
    sequence = result["sequences"][0]
    return sequence["tokens"], torch.tensor(sequence["logprobs"], dtype=torch.float64)


def _diff(a, b):
    return (a - b).abs().max().item()


def _mean(a, b):
    return (a - b).abs().mean().item()


def _close(a, b):
    return _diff(a, b) < TOL_MAX and _mean(a, b) < TOL_MEAN


def _checkpoint_root(server):
    root = os.environ.get("TINKER_SAMPLER_CHECKPOINT_DIR")
    if root:
        return Path(root)
    roots = sorted(Path("/tmp").glob("envloop-align-*/checkpoints"), key=lambda p: p.stat().st_mtime)
    assert roots, "set TINKER_SAMPLER_CHECKPOINT_DIR to the server's checkpoint_dir"
    return roots[-1]


def _export(tag_dir: Path):
    script = os.environ.get("TINKER_SAMPLER_LORA_EXPORT")
    assert script, "TINKER_SAMPLER_LORA_EXPORT must name EnvPlatform's guest/lora_export.py"
    subprocess.run([sys.executable, script, str(tag_dir)], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert (tag_dir / "lora_adapter" / "adapter_config.json").is_file()


def test_saved_and_staged_samplers_are_served_beside_the_training_weights(head, model):  # noqa: F811
    root = _checkpoint_root(head)
    session_id = model["session_id"]
    _train(head, model, 2)

    # Named sampler A, as the gateway saves it: training state, then rollout sync.
    head.call("/api/v1/save_weights", {"model_id": model["model_id"], "path": "sampler-a"})
    saved = head.call("/api/v1/save_weights_for_sampler", {"model_id": model["model_id"], "path": "a"})
    _export(root / "sampler-a")
    sampler_a = _sampler(head, session_id, 100, saved["path"])
    ref_a = _score(head, sampler_a)
    ref_a_tokens, ref_a_lps = _seeded(head, sampler_a)
    actor_a = _logprobs(_forward(head, model, _ce()))

    _train(head, model, 3)
    current = head.call(
        "/api/v1/save_weights_for_sampler",
        {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 101},
    )["sampling_session_id"]
    cur_before = _score(head, current)
    repeat = _score(head, current)
    print(
        f"repeat noise floor (current twice): max {_diff(repeat, cur_before):.4g} mean {_mean(repeat, cur_before):.3g}"
    )
    actor_before = _logprobs(_forward(head, model, _ce()))
    print(
        f"training moved the weights: |current - A| max {_diff(cur_before, ref_a):.4f} "
        f"mean {_mean(cur_before, ref_a):.4f}"
    )
    assert _diff(cur_before, ref_a) > 10 * TOL_MAX

    # A, now outside the rollout, is served from its saved PEFT export.
    later_a = _score(head, sampler_a)
    tokens, lps = _seeded(head, sampler_a)
    print(
        f"A later vs at save: max|dlogprob| = {_diff(later_a, ref_a):.4g} mean {_mean(later_a, ref_a):.3g} "
        f"(actor at save vs rollout at save: {_diff(actor_a[-len(ref_a) :], ref_a):.4g}); "
        f"seeded tokens equal: {tokens == ref_a_tokens}, max|dlp| = "
        f"{_diff(lps, ref_a_lps) if tokens == ref_a_tokens else float('nan'):.4g}"
    )
    assert _close(later_a, ref_a)

    # The gateway's cross-session path: the adapter staged under its own tag.
    for tag in ("adapter-foreign", "adapter-third"):
        shutil.copytree(root / "sampler-a" / "lora_adapter", root / tag / "lora_adapter", dirs_exist_ok=True)
    foreign = _sampler(head, session_id, 102, "sampler-adapter://adapter-foreign")
    staged = _score(head, foreign)
    print(f"staged copy of A vs A at save: max|dlogprob| {_diff(staged, ref_a):.4g} mean {_mean(staged, ref_a):.3g}")
    print(f"staged copy vs A from the session's own export (both files): max {_diff(staged, later_a):.4g}")
    assert _close(staged, ref_a)

    # Concurrent requests for different adapters and the current weights each get their own weights.
    # vLLM is not batch-invariant, so thresholds come from this hardware's batching noise floor: the
    # same mixed batch run twice. A wrong adapter (or contamination) moves a request by ~|current - A|.
    def mixed_batch():
        with ThreadPoolExecutor(6) as pool:
            futures = [pool.submit(_score, head, s) for s in (sampler_a, current, foreign) * 2]
            return [f.result() for f in futures]

    first, second = mixed_batch(), mixed_batch()
    max_n = max(_diff(a, b) for a, b in zip(first, second, strict=True))
    mean_n = max(_mean(a, b) for a, b in zip(first, second, strict=True))
    max_tol, mean_tol = max(3 * max_n, 0.5), max(3 * mean_n, 0.1)
    expected = (later_a, cur_before, staged) * 2
    pairs = [(_diff(r, e), _mean(r, e)) for r, e in zip(first + second, expected * 2, strict=True)]
    wrong_mean = _mean(cur_before, ref_a)
    print(
        f"batching noise floor (same mixed batch twice): max {max_n:.3g} mean {mean_n:.3g} -> "
        f"tolerance max {max_tol:.3g} mean {mean_tol:.3g}; wrong-adapter mean {wrong_mean:.3g}"
    )
    print("mixed vs solo (max, mean) per request: " + ", ".join(f"({m:.3g}, {a:.3g})" for m, a in pairs))
    assert all(m <= max_tol and a <= mean_tol for m, a in pairs)
    assert max(a for _, a in pairs) < 0.05 * wrong_mean

    # Two spare slots: a third adapter evicts the least recently used one, which reloads on demand.
    third = _sampler(head, session_id, 103, "sampler-adapter://adapter-third")
    evicted = [_score(head, s) for s in (third, sampler_a, foreign)]
    print(f"LRU (2 spare slots, 3 adapters): max deviation {max(_diff(e, ref_a) for e in evicted):.4g}")
    assert all(_close(e, ref_a) for e in evicted)

    # The current weights' sampler and the actor are untouched, and training continues.
    cur_after = _score(head, current)
    actor_after = _logprobs(_forward(head, model, _ce()))
    print(
        f"current sampler before/after adapters: {_diff(cur_after, cur_before):.3g}; "
        f"actor before/after: {_diff(actor_after, actor_before):.3g}"
    )
    assert _close(cur_after, cur_before)
    assert _diff(actor_after, actor_before) == 0.0
    _train(head, model, 1)
    after_step = head.call(
        "/api/v1/save_weights_for_sampler",
        {"model_id": model["model_id"], "path": None, "sampling_session_seq_id": 104},
    )["sampling_session_id"]
    assert _diff(_score(head, after_step), cur_before) > 0
    assert _close(_score(head, sampler_a), ref_a)
