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

"""EnvLoop Tinker-alignment tests (CPU only).

Loss tests compare the full server path (wire Datum -> TensorDict translator
-> branching loss callback -> VeRL ``no_padding_2_padding`` / ``agg_loss``)
against literal transcriptions of the formulas in the Tinker loss docs, for
both the loss value and the gradient w.r.t. the model log-probs.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, TypeAdapter
from tensordict import TensorDict
from tinker.types import Datum
from verl_tinker.backends._loss import make_branching_loss
from verl_tinker.data.datum_processing import (
    _coerce_verl_metrics_to_floats,
    _datums_to_sft_td,
    _datums_to_update_actor_td,
)
from verl_tinker.schemas import (
    TinkerCreateModelRequest,
    TinkerForwardBackwardRequest,
    TinkerSampleRequest,
)
from verl_tinker.tinker_losses import (
    GSPO_CLIPPED_FRACTION_METRIC,
    GSPO_LOG_RATIO_METRIC,
    LossContractError,
    finalize_gspo_metrics,
    normalize_loss_spec,
    tensor_data_to_torch,
    validate_loss_fn_inputs,
)
from verl_tinker.tinker_ops import forward, forward_backward, sample

from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric

# ==================== Tinker reference formulas (docs, verbatim) ====================


def ref_importance_sampling(target_logprobs, sampling_logprobs, advantages):
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    return -(prob_ratio * advantages).sum()


def ref_ppo(target_logprobs, sampling_logprobs, advantages, clip_low_threshold, clip_high_threshold):
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
    unclipped_objective = prob_ratio * advantages
    clipped_objective = clipped_ratio * advantages
    ppo_objective = torch.min(unclipped_objective, clipped_objective)
    return -ppo_objective.sum()


def ref_cispo(target_logprobs, sampling_logprobs, advantages, clip_low_threshold, clip_high_threshold):
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
    cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
    return -cispo_objective.sum()


def ref_dro(target_logprobs, sampling_logprobs, advantages, beta):
    quadratic_term = (target_logprobs - sampling_logprobs) ** 2
    dro_objective = target_logprobs * advantages - 0.5 * beta * quadratic_term
    return -dro_objective.sum()


def ref_gspo(
    target_logprobs, sampling_logprobs, advantages, sequence_ratio_mask, clip_low_threshold, clip_high_threshold
):
    in_ratio = sequence_ratio_mask != 0
    log_ratio = (target_logprobs.detach() - sampling_logprobs)[in_ratio].mean()
    ratio = torch.exp(log_ratio + target_logprobs - target_logprobs.detach())
    clipped_ratio = torch.clamp(ratio, clip_low_threshold, clip_high_threshold)
    gspo_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    clipped = not (clip_low_threshold <= math.exp(log_ratio.item()) <= clip_high_threshold)
    return -gspo_objective.sum(), log_ratio.item(), clipped


# ==================== helpers ====================


def _td(values, dtype="float32", shape=None):
    values = list(values)
    return SimpleNamespace(data=values, dtype=dtype, shape=shape or [len(values)])


def _rl_case(seed: int, prompt_len: int, completion_len: int, *, near_on_policy: bool = False):
    """One Tinker RL datum laid out exactly as the Tinker docs build it."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randint(5, 1000, (prompt_len + completion_len,), generator=g).tolist()
    n_prefix = prompt_len - 1
    n = len(full) - 1
    sampled = (-torch.rand(completion_len, generator=g) * 3 - 0.05).tolist()
    advantages = torch.randn(completion_len, generator=g)
    advantages[1] = 0.0  # a genuine zero advantage inside the completion
    sampling_logprobs = [0.0] * n_prefix + sampled
    adv = [0.0] * n_prefix + advantages.tolist()
    ratio_mask = [0.0] * n_prefix + [1.0] * completion_len
    # Learner logprobs for every target position (prompt positions too).
    target_logprobs = -torch.rand(n, generator=g) * 3 - 0.05
    if near_on_policy:
        target_logprobs[n_prefix:] = torch.tensor(sampled) + 1e-5 * torch.randn(completion_len, generator=g)
    datum = SimpleNamespace(
        model_input=SimpleNamespace(to_ints=lambda full=full: full[:-1]),
        loss_fn_inputs={
            "target_tokens": _td(full[1:], "int64"),
            "logprobs": _td(sampling_logprobs),
            "advantages": _td(adv),
        },
    )
    return datum, target_logprobs, torch.tensor(sampling_logprobs), torch.tensor(adv), torch.tensor(ratio_mask)


def _branching_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "_target_": "verl.workers.config.actor.FSDPActorConfig",
                    "strategy": "fsdp",
                    "rollout_n": 1,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "loss_agg_mode": "token-mean",
                    "policy_loss": {
                        "_target_": "verl.workers.config.actor.PolicyLossConfig",
                        "loss_mode": "vanilla",
                    },
                }
            }
        }
    )


def _run_server_rl_loss(spec, datums, target_logprobs, *, pad_to_multiple=4, dp_size=1):
    """Translator + branching loss exactly as the GPU worker runs it; returns (loss, metrics, grads)."""
    td = _datums_to_update_actor_td(
        datums, mini_batch_size=len(datums), pad_to_multiple=pad_to_multiple, loss_name=spec["name"]
    )
    tu.assign_non_tensor_data(td, "__tinker_loss_spec__", spec)
    tu.assign_non_tensor_data(td, "dp_size", dp_size)
    rows = []
    for i in range(td.batch_size[0]):
        valid_len = td["input_ids"][i].numel()
        if i < len(datums):
            rows.append(torch.cat([target_logprobs[i], torch.tensor([-7.0])]))
        else:  # DP filler rows clone datum 0's tokens
            rows.append(torch.full((valid_len,), -0.3))
        assert rows[-1].numel() == valid_len
    flat = torch.cat(rows).clone().requires_grad_(True)
    nested = torch.nested.nested_tensor_from_jagged(flat, td["input_ids"].offsets())
    loss, metrics = make_branching_loss(_branching_config())(model_output={"log_probs": nested}, data=td)
    loss.backward()
    grads = []
    offset = 0
    for row in rows:
        grads.append(flat.grad[offset : offset + row.numel()])
        offset += row.numel()
    return loss, metrics, grads


# ==================== loss math vs Tinker reference ====================


RL_SPECS = [
    normalize_loss_spec("importance_sampling"),
    normalize_loss_spec("ppo"),
    normalize_loss_spec("ppo", {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}),
    normalize_loss_spec("cispo"),
    normalize_loss_spec("cispo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.5}),
    normalize_loss_spec("dro", {"beta": 0.05}),
]


@pytest.mark.parametrize("spec", RL_SPECS, ids=lambda spec: f"{spec['name']}-{sorted(spec.items())}")
def test_rl_losses_match_tinker_reference_value_and_gradient(spec):
    cases = [
        _rl_case(seed, prompt_len, completion_len)
        for seed, prompt_len, completion_len in ((1, 4, 6), (2, 3, 9), (3, 6, 2))
    ]
    datums = [case[0] for case in cases]
    target_logprobs = [case[1] for case in cases]

    loss, metrics, grads = _run_server_rl_loss(spec, datums, target_logprobs)

    ref_total = torch.tensor(0.0)
    for (datum, t, s, a, _), grad in zip(cases, grads, strict=False):
        leaf = t.clone().requires_grad_(True)
        if spec["name"] == "importance_sampling":
            ref = ref_importance_sampling(leaf, s, a)
        elif spec["name"] == "ppo":
            ref = ref_ppo(leaf, s, a, spec["clip_low_threshold"], spec["clip_high_threshold"])
        elif spec["name"] == "cispo":
            ref = ref_cispo(leaf, s, a, spec["clip_low_threshold"], spec["clip_high_threshold"])
        else:
            # DRO's quadratic term is evaluated on the RL positions (advantage or
            # sampling logprob nonzero); Tinker-style prompt padding is excluded.
            evaluated = (a != 0) | (s != 0)
            ref = ref_dro(leaf[evaluated], s[evaluated], a[evaluated], spec["beta"])
        ref.backward()
        ref_total = ref_total + ref.detach()
        torch.testing.assert_close(grad[:-1], leaf.grad, rtol=1e-5, atol=1e-6)
        assert grad[-1].item() == 0.0  # appended last token is never scored
    for filler_grad in grads[len(cases) :]:
        assert torch.count_nonzero(filler_grad) == 0

    assert loss.item() == pytest.approx(ref_total.item(), rel=1e-5, abs=1e-5)
    coerced = finalize_gspo_metrics(_coerce_verl_metrics_to_floats(metrics))
    assert coerced["loss:sum"] == pytest.approx(ref_total.item(), rel=1e-5, abs=1e-5)


def test_importance_sampling_is_untruncated_like_tinker():
    """Upstream mapped IS onto VeRL token-TIS (weights capped at 2.0); Tinker's ratio is unbounded."""
    datum, t, s, a, _ = _rl_case(7, 3, 4)
    t = t.clone()
    t[2:] = s[2:] + math.log(5.0)  # ratio = 5 on every sampled token
    loss, _, grads = _run_server_rl_loss(normalize_loss_spec("importance_sampling"), [datum], [t], pad_to_multiple=1)
    expected = -(5.0 * a).sum()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    torch.testing.assert_close(grads[0][:-1], -5.0 * a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize(
    "config",
    [None, {"clip_low_threshold": 0.5, "clip_high_threshold": 2.0}],
    ids=["tinker-defaults", "wide-clip"],
)
def test_gspo_matches_tinker_reference_value_gradient_and_metrics(with_mask, config):
    spec = normalize_loss_spec("gspo", config)
    cases = [
        _rl_case(11, 4, 6),
        _rl_case(12, 3, 7, near_on_policy=True),  # sequence ratio inside the narrow default band
        _rl_case(13, 5, 3),
    ]
    datums = []
    for datum, *_rest, ratio_mask in cases:
        if with_mask:
            datum.loss_fn_inputs["sequence_ratio_mask"] = _td(ratio_mask.tolist())
        datums.append(datum)

    loss, metrics, grads = _run_server_rl_loss(spec, datums, [case[1] for case in cases])

    ref_total, log_ratios, clipped = 0.0, [], []
    for (_, t, s, a, ratio_mask), grad in zip(cases, grads, strict=False):
        leaf = t.clone().requires_grad_(True)
        # Without an explicit mask the ratio is taken over the sampled tokens
        # (logprob or advantage nonzero), which is the completion here.
        ref, log_ratio, was_clipped = ref_gspo(
            leaf, s, a, ratio_mask, spec["clip_low_threshold"], spec["clip_high_threshold"]
        )
        ref.backward()
        ref_total += ref.item()
        log_ratios.append(log_ratio)
        clipped.append(was_clipped)
        torch.testing.assert_close(grad[:-1], leaf.grad, rtol=1e-5, atol=1e-6)

    assert loss.item() == pytest.approx(ref_total, rel=1e-5, abs=1e-5)
    coerced = finalize_gspo_metrics(_coerce_verl_metrics_to_floats(metrics))
    assert coerced["loss:sum"] == pytest.approx(ref_total, rel=1e-5, abs=1e-5)
    assert coerced[GSPO_LOG_RATIO_METRIC] == pytest.approx(sum(log_ratios) / len(log_ratios), rel=1e-5, abs=1e-6)
    assert coerced[GSPO_CLIPPED_FRACTION_METRIC] == pytest.approx(sum(clipped) / len(clipped))
    assert not any(key.startswith("__gspo") for key in coerced)
    if config is None:
        assert clipped == [True, False, True]


def test_gspo_ratio_mask_controls_sequence_ratio_not_loss_positions():
    """A sequence_ratio_mask covering only part of the completion changes the ratio, not which tokens train."""
    datum, t, s, a, ratio_mask = _rl_case(21, 3, 6)
    partial = ratio_mask.clone()
    partial[-3:] = 0.0
    datum.loss_fn_inputs["sequence_ratio_mask"] = _td(partial.tolist())
    spec = normalize_loss_spec("gspo", {"clip_low_threshold": 0.0, "clip_high_threshold": 100.0})

    loss, _, grads = _run_server_rl_loss(spec, [datum], [t], pad_to_multiple=1)

    leaf = t.clone().requires_grad_(True)
    ref, _, _ = ref_gspo(leaf, s, a, partial, 0.0, 100.0)
    ref.backward()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)
    torch.testing.assert_close(grads[0][:-1], leaf.grad, rtol=1e-5, atol=1e-6)


def test_rl_datum_without_evaluated_positions_contributes_zero_loss():
    datum, t, s, a, _ = _rl_case(31, 3, 3)
    zero = SimpleNamespace(
        model_input=datum.model_input,
        loss_fn_inputs={
            "target_tokens": datum.loss_fn_inputs["target_tokens"],
            "logprobs": _td([0.0] * len(s)),
            "advantages": _td([0.0] * len(a)),
        },
    )
    loss, _, grads = _run_server_rl_loss(normalize_loss_spec("ppo"), [datum, zero], [t, t], pad_to_multiple=1)
    leaf = t.clone().requires_grad_(True)
    ref = ref_ppo(leaf, s, a, 0.8, 1.2)
    assert loss.item() == pytest.approx(ref.item(), rel=1e-5)
    assert torch.count_nonzero(grads[1]) == 0


def test_rl_loss_scales_by_dp_size_for_global_token_sum():
    datum, t, *_ = _rl_case(41, 3, 4)
    loss1, _, _ = _run_server_rl_loss(normalize_loss_spec("importance_sampling"), [datum], [t], pad_to_multiple=1)
    loss2, _, _ = _run_server_rl_loss(
        normalize_loss_spec("importance_sampling"), [datum], [t], pad_to_multiple=1, dp_size=2
    )
    assert loss2.item() == pytest.approx(2 * loss1.item())


# ==================== cross_entropy with float / signed weights ====================


def _ce_case(seed=0, n=5, k=None):
    g = torch.Generator().manual_seed(seed)
    full = torch.randint(5, 50, (n + 1,), generator=g).tolist()
    if k is None:
        weights = torch.randn(n, generator=g)
        weights[0] = 0.0
        inputs = {"target_tokens": _td(full[1:], "int64"), "weights": _td(weights.tolist())}
        target = None
    else:
        target = torch.randint(0, 17, (n, k), generator=g)
        weights = torch.randn(n, k, generator=g)
        weights[0] = 0.0
        inputs = {
            "target_tokens": _td(target.flatten().tolist(), "int64", [n, k]),
            "weights": _td(weights.flatten().tolist(), "float32", [n, k]),
        }
    datum = SimpleNamespace(model_input=SimpleNamespace(to_ints=lambda: full[:-1]), loss_fn_inputs=inputs)
    return datum, target, weights


def test_cross_entropy_linear_surrogate_gradient_matches_custom_loss():
    """forward_backward_custom sends weights = -dC/dlogprobs; the CE gradient must equal dC/dlogprobs."""
    datum, _, _ = _ce_case(1)
    n = len(datum.model_input.to_ints())
    logprobs = -torch.rand(n) - 0.1
    lp_leaf = logprobs.clone().requires_grad_(True)
    custom = (lp_leaf**2).sum() + 3 * lp_leaf[2]  # any nonlinear client-side loss
    custom.backward()
    datum.loss_fn_inputs["weights"] = _td((-lp_leaf.grad).tolist())

    td = _datums_to_sft_td([datum], mini_batch_size=1)
    tu.assign_non_tensor_data(td, "dp_size", 1)
    flat = torch.cat([logprobs, torch.tensor([-1.0])]).requires_grad_(True)
    nested = torch.nested.nested_tensor_from_jagged(flat, td["input_ids"].offsets())
    loss, _ = make_branching_loss(_branching_config())(model_output={"log_probs": nested}, data=td)
    loss.backward()

    torch.testing.assert_close(flat.grad[:-1], lp_leaf.grad)
    assert loss.item() == pytest.approx(-(logprobs * -lp_leaf.grad).sum().item(), rel=1e-5)


def test_topk_cross_entropy_aligns_rows_with_next_token_logits_and_keeps_signed_weights():
    """Row i of (N, K) targets is scored by the logits at position i (which predict token i + 1)."""
    datum, target, weights = _ce_case(2, n=4, k=3)
    vocab = 17
    td = _datums_to_sft_td([datum], mini_batch_size=1)
    loss_fn = make_branching_loss(_branching_config())
    logits = torch.randn(1, 5, vocab, requires_grad=True)

    outputs = loss_fn(student_logits=logits, data=td)
    offsets = td["input_ids"].offsets()
    model_output = {key: torch.nested.nested_tensor_from_jagged(v.squeeze(0), offsets) for key, v in outputs.items()}
    tu.assign_non_tensor_data(td, "dp_size", 1)
    loss, _ = loss_fn(model_output=model_output, data=td)

    log_softmax = torch.log_softmax(logits[0], dim=-1)
    reference = -sum((weights[i] * log_softmax[i, target[i]]).sum() for i in range(4))
    assert loss.item() == pytest.approx(reference.item(), rel=1e-5)
    (g_ours,) = torch.autograd.grad(loss, logits, retain_graph=True)
    (g_ref,) = torch.autograd.grad(reference, logits)
    torch.testing.assert_close(g_ours, g_ref, rtol=1e-5, atol=1e-6)


def test_sparse_csr_tensor_data_is_densified():
    dense = torch.tensor([[0, 3, 0], [4, 0, 5]], dtype=torch.int64)
    csr = dense.to_sparse_csr()
    td = SimpleNamespace(
        data=csr.values().tolist(),
        dtype="int64",
        shape=[2, 3],
        sparse_crow_indices=csr.crow_indices().tolist(),
        sparse_col_indices=csr.col_indices().tolist(),
    )
    torch.testing.assert_close(tensor_data_to_torch(td), dense)


# ==================== loss_fn / loss_fn_config / loss_fn_inputs contract ====================


@pytest.mark.parametrize(
    ("loss_fn", "extra", "match"),
    [
        ("ppo", {"weights": [1.0] * 5}, "unexpected=\\['weights'\\]"),
        ("importance_sampling", {"response_mask": [1.0] * 5}, "unexpected=\\['response_mask'\\]"),
        ("dro", {"sequence_ratio_mask": [1.0] * 5}, "unexpected=\\['sequence_ratio_mask'\\]"),
    ],
)
def test_rl_losses_accept_exactly_target_tokens_logprobs_advantages(loss_fn, extra, match):
    datum, *_ = _rl_case(51, 2, 4)
    for key, values in extra.items():
        datum.loss_fn_inputs[key] = _td(values)
    with pytest.raises(LossContractError, match=match):
        validate_loss_fn_inputs(loss_fn, [datum])


def test_contract_rejects_missing_keys_bad_shapes_and_unshifted_targets():
    datum, *_ = _rl_case(52, 3, 3)
    missing = SimpleNamespace(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs))
    del missing.loss_fn_inputs["advantages"]
    with pytest.raises(LossContractError, match="missing=\\['advantages'\\]"):
        validate_loss_fn_inputs("ppo", [missing])

    short = SimpleNamespace(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs))
    short.loss_fn_inputs["advantages"] = _td([1.0])
    with pytest.raises(LossContractError, match="must have shape"):
        validate_loss_fn_inputs("ppo", [short])

    unshifted = SimpleNamespace(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs))
    targets = list(datum.loss_fn_inputs["target_tokens"].data)
    targets[0] += 1
    unshifted.loss_fn_inputs["target_tokens"] = _td(targets, "int64")
    with pytest.raises(LossContractError, match="shifted by one"):
        validate_loss_fn_inputs("ppo", [unshifted])

    empty_ratio = SimpleNamespace(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs))
    empty_ratio.loss_fn_inputs["sequence_ratio_mask"] = _td([0.0] * len(targets))
    with pytest.raises(LossContractError, match="at least one token"):
        validate_loss_fn_inputs("gspo", [empty_ratio])

    with pytest.raises(LossContractError, match="at least one datum"):
        validate_loss_fn_inputs("ppo", [])


def test_cross_entropy_contract():
    datum, _, _ = _ce_case(3)
    validate_loss_fn_inputs("cross_entropy", [datum])
    topk, _, _ = _ce_case(4, k=2)
    validate_loss_fn_inputs("cross_entropy", [topk])

    no_weights = SimpleNamespace(
        model_input=datum.model_input, loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]}
    )
    with pytest.raises(LossContractError, match="missing=\\['weights'\\]"):
        validate_loss_fn_inputs("cross_entropy", [no_weights])
    # forward-only (forward_backward_custom's first pass may omit weights)
    validate_loss_fn_inputs("cross_entropy", [no_weights], forward_only=True)
    with pytest.raises(LossContractError, match="supports only 1-D target_tokens"):
        validate_loss_fn_inputs("cross_entropy", [topk], forward_only=True)

    legacy_mask = SimpleNamespace(
        model_input=datum.model_input,
        loss_fn_inputs={
            "target_tokens": datum.loss_fn_inputs["target_tokens"],
            "mask": datum.loss_fn_inputs["weights"],
        },
    )
    with pytest.raises(LossContractError, match="unexpected=\\['mask'\\]"):
        validate_loss_fn_inputs("cross_entropy", [legacy_mask])
    # verl-tinker extension keeps the permissive legacy contract
    validate_loss_fn_inputs("custom_from_config", [legacy_mask])


def _wire_rl_datum(seed=61, with_ratio_mask=False):
    datum, *_ = _rl_case(seed, 3, 4)
    inputs = {
        key: {"data": list(value.data), "dtype": value.dtype, "shape": list(value.shape)}
        for key, value in datum.loss_fn_inputs.items()
    }
    if with_ratio_mask:
        n = len(inputs["target_tokens"]["data"])
        inputs["sequence_ratio_mask"] = {"data": [0.0, 0.0] + [1.0] * (n - 2), "dtype": "float32", "shape": [n]}
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(datum.model_input.to_ints())}]},
        "loss_fn_inputs": inputs,
    }


# The server parses requests with its pinned tinker 0.19 pydantic types; newer
# SDK releases ship plain dataclasses that FastAPI cannot parse (upstream too).
_PINNED_SERVER_TYPES = isinstance(Datum, type) and issubclass(Datum, BaseModel)


@pytest.mark.skipif(not _PINNED_SERVER_TYPES, reason="server request schemas need the pinned tinker 0.19 types")
def test_wire_schema_parses_gspo_and_string_config_values_for_typed_rejection():
    body = {
        "forward_backward_input": {
            "data": [_wire_rl_datum(with_ratio_mask=True)],
            "loss_fn": "gspo",
            "loss_fn_config": {"clip_low_threshold": 0.999, "clip_high_threshold": "1.001"},
        },
        "model_id": "m",
    }
    req = TypeAdapter(TinkerForwardBackwardRequest).validate_python(body)
    assert req.forward_backward_input.loss_fn == "gspo"
    assert req.forward_backward_input.loss_fn_config["clip_high_threshold"] == "1.001"
    with pytest.raises(LossContractError, match="must be a finite number"):
        normalize_loss_spec("gspo", req.forward_backward_input.loss_fn_config)
    validate_loss_fn_inputs("gspo", req.forward_backward_input.data)


def test_sample_request_schema_accepts_sdk_030_fields():
    body = {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "sampling_params": {"max_tokens": 4},
        "num_samples": 2,
        "topk_sample_logprobs": 5,
        "record_stability_info": False,
        "target_prompt_logprobs": {"data": [1, -1], "dtype": "int64", "shape": [2, 1]},
        "sampling_session_id": "s",
        "seq_id": 0,
    }
    req = TinkerSampleRequest.model_validate(body)
    assert req.topk_sample_logprobs == 5
    assert req.target_prompt_logprobs["shape"] == [2, 1]
    create = TinkerCreateModelRequest.model_validate(
        {
            "session_id": "s",
            "model_seq_id": 0,
            "base_model": "m",
            "lora_config": {"rank": 8},
            "optimizer_config": {"type": "adamw"},
        }
    )
    assert create.lora_config.rank == 8


# ==================== sampling: topk_sample_logprobs ====================


class _SamplingEngine:
    """Rollout stub: sampling requests return fixed tokens; scoring requests return vLLM-shaped top-k rows."""

    def __init__(self, completion, *, k, max_model_len=None):
        self.completion = completion
        self.k = k
        self.calls = []
        self.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"max_model_len": max_model_len}}})

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append((list(prompt_ids), dict(sampling_params)))
        if "prompt_logprobs" not in sampling_params:
            return SimpleNamespace(
                token_ids=list(self.completion),
                log_probs=[-0.1 * (i + 1) for i in range(len(self.completion))],
                stop_reason="stop",
                extra_fields={},
            )
        k = sampling_params["prompt_logprobs"]
        # verl extract_prompt_logprobs: rows for positions 1..L-1 plus a dummy last row.
        ids = [[1000 * position + rank for rank in range(k)] for position in range(1, len(prompt_ids))] + [[0] * k]
        lps = [[-(position + rank / 10) for rank in range(k)] for position in range(1, len(prompt_ids))] + [[0.0] * k]
        return SimpleNamespace(
            token_ids=[7],
            log_probs=None,
            stop_reason="length",
            extra_fields={"prompt_ids": ids, "prompt_logprobs": lps},
        )


def _sample_req(prompt, *, k, num_samples=1, max_tokens=8):
    return SimpleNamespace(
        prompt=SimpleNamespace(to_ints=lambda: list(prompt)),
        sampling_params=SimpleNamespace(
            max_tokens=max_tokens, temperature=0.7, top_p=0.9, top_k=-1, stop=None, seed=None
        ),
        num_samples=num_samples,
        prompt_logprobs=False,
        topk_prompt_logprobs=0,
        topk_sample_logprobs=k,
    )


@pytest.mark.asyncio
async def test_topk_sample_logprobs_are_aligned_with_sampled_positions():
    prompt, completion = [11, 12, 13], [21, 22, 23, 24]
    engine = _SamplingEngine(completion, k=3)

    result = await sample(engine, _sample_req(prompt, k=3, num_samples=2))

    for sequence in result["sequences"]:
        rows = sequence["topk_logprobs"]
        assert len(rows) == len(completion)
        for t, row in enumerate(rows):
            position = len(prompt) + t  # full-sequence position of sampled token t
            assert row == [(1000 * position + rank, -(position + rank / 10)) for rank in range(3)]
    scoring = [call for call in engine.calls if "prompt_logprobs" in call[1]]
    assert len(scoring) == 2
    assert scoring[0][0] == prompt + completion
    assert scoring[0][1]["prompt_logprobs"] == 3 and scoring[0][1]["max_tokens"] == 1
    assert scoring[0][1]["temperature"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_topk_sample_logprobs_drop_last_position_without_context_headroom():
    prompt, completion = [11, 12, 13], [21, 22]
    engine = _SamplingEngine(completion, k=2, max_model_len=5)

    result = await sample(engine, _sample_req(prompt, k=2))

    rows = result["sequences"][0]["topk_logprobs"]
    assert rows[0] == [(3000, -3.0), (3001, -3.1)]
    assert rows[1] is None
    scoring = [call for call in engine.calls if "prompt_logprobs" in call[1]]
    assert scoring[0][0] == prompt + completion[:-1]


@pytest.mark.asyncio
async def test_topk_sample_logprobs_retry_without_last_token_only_on_context_overflow():
    prompt, completion = [11, 12, 13], [21, 22]

    class _Overflow(_SamplingEngine):
        async def generate(self, request_id, prompt_ids, sampling_params):
            if "prompt_logprobs" in sampling_params and len(prompt_ids) >= 5:
                self.calls.append((list(prompt_ids), dict(sampling_params)))
                raise ValueError(
                    "Prompt length (5) leaves no room to generate within the model's maximum context length"
                )
            return await super().generate(request_id, prompt_ids, sampling_params)

    engine = _Overflow(completion, k=1)  # max_model_len unknown (None)
    result = await sample(engine, _sample_req(prompt, k=1))
    assert result["sequences"][0]["topk_logprobs"] == [[(3000, -3.0)], None]

    class _Broken(_SamplingEngine):
        async def generate(self, request_id, prompt_ids, sampling_params):
            if "prompt_logprobs" in sampling_params:
                raise RuntimeError("max_logprobs exceeded")
            return await super().generate(request_id, prompt_ids, sampling_params)

    with pytest.raises(RuntimeError, match="max_logprobs"):
        await sample(_Broken(completion, k=1), _sample_req(prompt, k=1))


@pytest.mark.asyncio
async def test_sample_without_topk_keeps_upstream_shape():
    engine = _SamplingEngine([21], k=0)
    result = await sample(engine, _sample_req([1, 2], k=0))
    assert "topk_logprobs" not in result["sequences"][0]
    assert all("prompt_logprobs" not in call[1] for call in engine.calls)


# ==================== forward-only ====================


class _ForwardOnlyEngine:
    world_size = 2

    def __init__(self, target_logprobs):
        self.target_logprobs = target_logprobs
        self.calls = []

    def compute_log_prob(self, data):
        self.calls.append("compute_log_prob")
        rows = [torch.cat([lp, torch.tensor([-9.0])]) for lp in self.target_logprobs]
        rows += [rows[0].clone()] * (data.batch_size[0] - len(rows))  # DP padding rows
        return TensorDict(
            {"log_probs": torch.nested.as_nested_tensor(rows, layout=torch.jagged)}, batch_size=[len(rows)]
        )

    def forward_backward(self, data):  # pragma: no cover - must never be called
        self.calls.append("forward_backward")
        raise AssertionError("forward must not run backward")

    def optim_step(self, *args, **kwargs):  # pragma: no cover - must never be called
        self.calls.append("optim_step")
        raise AssertionError("forward must not step the optimizer")


async def _to_thread_inline(func, /, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_forward_only_computes_logprobs_and_tinker_loss_without_gradients(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    cases = [_rl_case(71, 3, 4), _rl_case(72, 4, 3, near_on_policy=True), _rl_case(73, 2, 5)]
    engine = _ForwardOnlyEngine([case[1] for case in cases])

    result = await forward(engine, [case[0] for case in cases], "gspo", None)

    assert engine.calls == ["compute_log_prob"]
    assert result["type"] == "forward_backward"
    assert result["loss_fn_output_type"] == "gspo"
    for output, case in zip(result["loss_fn_outputs"], cases, strict=True):
        assert output["logprobs"]["shape"] == [case[1].numel()]
        torch.testing.assert_close(torch.tensor(output["logprobs"]["data"]), case[1])
    spec = normalize_loss_spec("gspo")
    ref = [
        ref_gspo(t, s, a, mask, spec["clip_low_threshold"], spec["clip_high_threshold"]) for _, t, s, a, mask in cases
    ]
    assert result["metrics"]["loss:sum"] == pytest.approx(sum(r[0].item() for r in ref), rel=1e-5)
    assert result["metrics"][GSPO_LOG_RATIO_METRIC] == pytest.approx(sum(r[1] for r in ref) / 3, rel=1e-5)
    assert result["metrics"][GSPO_CLIPPED_FRACTION_METRIC] == pytest.approx(sum(r[2] for r in ref) / 3)


@pytest.mark.asyncio
async def test_forward_only_cross_entropy_loss_and_optional_weights(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    datum, _, weights = _ce_case(81)
    logprobs = -torch.rand(len(weights)) - 0.1
    engine = _ForwardOnlyEngine([logprobs])

    result = await forward(engine, [datum], "cross_entropy", None)
    assert result["metrics"] == {"loss:sum": pytest.approx(-(logprobs * weights).sum().item(), rel=1e-5)}

    no_weights = SimpleNamespace(
        model_input=datum.model_input, loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]}
    )
    result = await forward(engine, [no_weights], "cross_entropy", None)
    assert result["metrics"] == {}
    result = await forward(engine, [no_weights])  # upstream call shape still works
    assert result["metrics"] == {} and engine.calls.count("compute_log_prob") == 3


@pytest.mark.asyncio
async def test_forward_backward_reports_tinker_metric_names(monkeypatch):
    monkeypatch.setattr("verl_tinker.tinker_ops.asyncio.to_thread", _to_thread_inline)
    datum, *_ = _rl_case(91, 3, 3)
    result_td = TensorDict({}, batch_size=[])
    result_td.set_non_tensor(
        "metrics",
        {
            "loss": 1.5,
            "__gspo_seq_log_ratio_total": [Metric(value=0.2, aggregation="sum"), Metric(value=0.4, aggregation="sum")],
            "__gspo_num_sequences": [Metric(value=1.0, aggregation="sum"), Metric(value=1.0, aggregation="sum")],
            "__gspo_clipped_sequences": [Metric(value=1.0, aggregation="sum"), Metric(value=0.0, aggregation="sum")],
        },
    )

    class _Engine:
        world_size = 1

        def forward_backward(self, data):
            assert tu.get_non_tensor_data(data, "__tinker_loss_spec__", None)["name"] == "gspo"
            assert "sequence_ratio_mask" in data.keys()
            return result_td

    result = await forward_backward(_Engine(), [datum], "gspo")
    assert result["metrics"]["loss:sum"] == 1.5
    assert result["metrics"][GSPO_LOG_RATIO_METRIC] == pytest.approx(0.3)
    assert result["metrics"][GSPO_CLIPPED_FRACTION_METRIC] == pytest.approx(0.5)
    assert not any(key.startswith("__gspo") for key in result["metrics"])
