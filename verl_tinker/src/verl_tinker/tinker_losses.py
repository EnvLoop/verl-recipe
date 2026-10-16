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

"""Tinker loss formulas and the ``loss_fn`` wire contract (pure torch).

This module has no VeRL / Ray imports so the same code is used by:

* the training loss callback (``backends/_loss.py``) on the GPU workers,
* the forward-only ``/api/v1/forward`` endpoint (loss value from logprobs),
* request validation in the router (synchronous 4xx before scheduling), and
* the CPU unit tests.

Every formula follows the Tinker loss documentation (token-level, sum
reduction). Tensors are ``(B, T)`` (or ``(T,)``) with ``target_logprobs`` the
learner log-probs, ``sampling_logprobs`` the sampler log-probs (``logprobs`` on
the wire) and ``advantages`` per token.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch

CROSS_ENTROPY = "cross_entropy"
CUSTOM_FROM_CONFIG = "custom_from_config"
TINKER_RL_LOSSES = ("importance_sampling", "ppo", "cispo", "dro", "gspo")
TINKER_LOSSES = (CROSS_ENTROPY, *TINKER_RL_LOSSES)
SUPPORTED_LOSSES = (*TINKER_LOSSES, CUSTOM_FROM_CONFIG)

# Tinker defaults (docs: losses/ppo, losses/cispo, losses/gspo).
DEFAULT_CLIP_THRESHOLDS: dict[str, tuple[float, float]] = {
    "ppo": (0.8, 1.2),
    "cispo": (0.0, 4.0),
    "gspo": (1.0 - 3e-4, 1.0 + 4e-4),
}

_RL_INPUT_KEYS = frozenset({"target_tokens", "logprobs", "advantages"})
# loss name -> (required keys, optional keys)
LOSS_FN_INPUT_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    CROSS_ENTROPY: (frozenset({"target_tokens", "weights"}), frozenset()),
    "importance_sampling": (_RL_INPUT_KEYS, frozenset()),
    "ppo": (_RL_INPUT_KEYS, frozenset()),
    "cispo": (_RL_INPUT_KEYS, frozenset()),
    "dro": (_RL_INPUT_KEYS, frozenset()),
    "gspo": (_RL_INPUT_KEYS, frozenset({"sequence_ratio_mask"})),
}

# Internal SUM metrics emitted per micro-batch by the GSPO training loss and
# folded into the documented ``:mean`` metrics by ``finalize_gspo_metrics``.
GSPO_LOG_RATIO_TOTAL = "__gspo_seq_log_ratio_total"
GSPO_NUM_SEQUENCES = "__gspo_num_sequences"
GSPO_CLIPPED_SEQUENCES = "__gspo_clipped_sequences"
GSPO_LOG_RATIO_METRIC = "gspo_sequence_log_ratio:mean"
GSPO_CLIPPED_FRACTION_METRIC = "gspo_clipped_fraction:mean"


class LossContractError(ValueError):
    """Invalid ``loss_fn`` / ``loss_fn_config`` / ``loss_fn_inputs`` request."""


# ==================== loss_fn / loss_fn_config ====================


def _finite_number(name: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LossContractError(
            f"loss_fn={name!r}: loss_fn_config[{key!r}] must be a finite number, got {type(value).__name__}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise LossContractError(f"loss_fn={name!r}: loss_fn_config[{key!r}] must be a finite number")
    return value


def normalize_loss_spec(loss_name: str, loss_config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Validate ``loss_fn`` + ``loss_fn_config`` and fill Tinker defaults.

    Returns a plain dict (it travels as TensorDict non-tensor metadata):

    * ``cross_entropy`` / ``importance_sampling``: ``{"name"}`` (no config keys)
    * ``ppo`` / ``cispo`` / ``gspo``: ``{"name", "clip_low_threshold", "clip_high_threshold"}``
    * ``dro``: ``{"name", "beta"}`` (``beta`` required, Tinker documents no default)
    * ``custom_from_config``: ``{"name"}``; the wire config is ignored and the
      server-start actor loss config is used (verl-tinker extension).
    """
    if loss_name not in SUPPORTED_LOSSES:
        raise LossContractError(f"Unsupported loss {loss_name!r}; expected one of {sorted(SUPPORTED_LOSSES)}")

    if loss_name == CUSTOM_FROM_CONFIG:
        return {"name": loss_name}

    config = dict(loss_config or {})
    numbers = {key: _finite_number(loss_name, key, value) for key, value in config.items()}

    if loss_name in {CROSS_ENTROPY, "importance_sampling"}:
        if numbers:
            raise LossContractError(f"loss_fn={loss_name!r} does not accept loss_fn_config; got {sorted(numbers)}")
        return {"name": loss_name}

    if loss_name in DEFAULT_CLIP_THRESHOLDS:
        allowed = {"clip_low_threshold", "clip_high_threshold"}
        unknown = sorted(set(numbers) - allowed)
        if unknown:
            raise LossContractError(f"loss_fn={loss_name!r} received unsupported config fields: {unknown}")
        default_low, default_high = DEFAULT_CLIP_THRESHOLDS[loss_name]
        low = numbers.get("clip_low_threshold", default_low)
        high = numbers.get("clip_high_threshold", default_high)
        if not 0.0 <= low <= 1.0 <= high:
            raise LossContractError(
                f"loss_fn={loss_name!r} requires 0 <= clip_low_threshold <= 1 <= "
                f"clip_high_threshold; got low={low}, high={high}"
            )
        return {"name": loss_name, "clip_low_threshold": low, "clip_high_threshold": high}

    # dro
    unknown = sorted(set(numbers) - {"beta"})
    if unknown:
        raise LossContractError(f"loss_fn='dro' received unsupported config fields: {unknown}")
    if "beta" not in numbers:
        raise LossContractError("loss_fn='dro' requires loss_fn_config['beta'] (a non-negative number)")
    if numbers["beta"] < 0:
        raise LossContractError(f"loss_fn='dro' requires beta >= 0; got {numbers['beta']}")
    return {"name": "dro", "beta": numbers["beta"]}


# ==================== loss_fn_inputs ====================


def tensor_data_to_torch(td: Any) -> torch.Tensor:
    """Wire ``TensorData`` (dense or sparse CSR) -> dense torch tensor."""
    dtype = torch.float32 if getattr(td, "dtype", "float32") == "float32" else torch.int64
    shape = list(getattr(td, "shape", None) or [])
    crow = getattr(td, "sparse_crow_indices", None)
    if crow is not None:
        col = getattr(td, "sparse_col_indices", None)
        if col is None or len(shape) != 2:
            raise LossContractError("sparse CSR TensorData requires sparse_col_indices and a 2-D shape")
        return torch.sparse_csr_tensor(
            torch.tensor(crow, dtype=torch.int64),
            torch.tensor(col, dtype=torch.int64),
            torch.tensor(td.data, dtype=dtype),
            size=shape,
        ).to_dense()
    t = torch.tensor(td.data, dtype=dtype)
    if len(shape) > 1:
        t = t.view(*shape)
    return t


def rl_token_mask(
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    sequence_ratio_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Positions an RL loss is evaluated on (target-indexed, bool).

    Tinker's RL datums pad prompt positions with ``logprobs=0`` and
    ``advantages=0``; a position contributes to IS / PPO / CISPO / GSPO only
    through a nonzero advantage, so evaluating the loss on
    ``advantage != 0 or logprob != 0 (or sequence_ratio_mask != 0)`` is
    equivalent to Tinker's full-sequence sum for those losses. For DRO the
    quadratic penalty is applied on the same positions (prompt padding with
    ``logprobs == 0`` and ``advantages == 0`` is excluded).
    """
    mask = (advantages != 0) | (sampling_logprobs != 0)
    if sequence_ratio_mask is not None:
        mask = mask | (sequence_ratio_mask != 0)
    return mask


def validate_loss_fn_inputs(loss_name: str, datums: Sequence[Any], *, forward_only: bool = False) -> None:
    """Enforce Tinker's ``loss_fn_inputs`` contract before any GPU work.

    * RL losses accept exactly ``target_tokens``, ``logprobs``, ``advantages``
      (``gspo`` additionally accepts ``sequence_ratio_mask``).
    * ``cross_entropy`` accepts ``target_tokens`` and ``weights`` (float,
      any sign); on ``/forward`` ``weights`` may be omitted.
    * Tensors are 1-D of length ``len(model_input)`` (``cross_entropy`` may use
      ``(N, K)`` targets/weights for top-K distillation on ``forward_backward``).
    * This backend reconstructs the sequence as ``model_input + [target[-1]]``,
      so 1-D ``target_tokens[:-1]`` must equal ``model_input[1:]`` (the
      shift-by-one layout the Tinker docs prescribe); anything else would be
      silently mis-scored and is rejected.

    ``custom_from_config`` keeps the permissive legacy behaviour.
    """
    if loss_name == CUSTOM_FROM_CONFIG:
        return
    if not datums:
        raise LossContractError("forward_backward_input.data must contain at least one datum")
    required, optional = LOSS_FN_INPUT_KEYS[loss_name]
    if forward_only and loss_name == CROSS_ENTROPY:
        required = required - {"weights"}
        optional = optional | {"weights"}

    for index, datum in enumerate(datums):
        where = f"data[{index}]"
        keys = set(datum.loss_fn_inputs)
        missing = sorted(required - keys)
        extra = sorted(keys - required - optional)
        if missing or extra:
            accepted = sorted(required | optional)
            raise LossContractError(
                f"{where}: loss_fn={loss_name!r} accepts loss_fn_inputs {accepted}; "
                f"missing={missing} unexpected={extra}"
            )
        prefix = list(datum.model_input.to_ints())
        n = len(prefix)
        if n == 0:
            raise LossContractError(f"{where}: model_input must contain at least one token")
        tensors = {key: tensor_data_to_torch(value) for key, value in datum.loss_fn_inputs.items()}
        target = tensors["target_tokens"]
        if target.dim() == 2 and loss_name == CROSS_ENTROPY and not forward_only:
            weights = tensors["weights"]
            if tuple(weights.shape) != tuple(target.shape) or target.shape[0] != n:
                raise LossContractError(
                    f"{where}: top-K cross_entropy requires target_tokens and weights of shape "
                    f"(len(model_input)={n}, K); got {tuple(target.shape)} and {tuple(weights.shape)}"
                )
            continue
        if target.dim() != 1:
            what = "/forward" if forward_only else f"loss_fn={loss_name!r}"
            raise LossContractError(
                f"{where}: {what} supports only 1-D target_tokens of length len(model_input); "
                f"got shape {tuple(target.shape)}"
            )
        for key, tensor in tensors.items():
            if tensor.dim() != 1 or tensor.numel() != n:
                raise LossContractError(
                    f"{where}: loss_fn_inputs[{key!r}] must have shape ({n},) == len(model_input); "
                    f"got {tuple(tensor.shape)}"
                )
        if n > 1 and not torch.equal(target[:-1], torch.tensor(prefix[1:], dtype=target.dtype)):
            bad = int((target[:-1] != torch.tensor(prefix[1:], dtype=target.dtype)).nonzero()[0].item())
            raise LossContractError(
                f"{where}: target_tokens must be model_input shifted by one "
                f"(target_tokens[i] == model_input[i + 1]); first mismatch at position {bad}. "
                "This backend scores the next token of the sequence and cannot score arbitrary target ids."
            )
        if loss_name == "gspo" and "sequence_ratio_mask" in tensors:
            if not bool((tensors["sequence_ratio_mask"] != 0).any()):
                raise LossContractError(f"{where}: sequence_ratio_mask must mark at least one token")


# ==================== formulas ====================


@dataclass
class RLTokenLosses:
    """Per-token Tinker RL loss (before masking) plus GSPO per-sequence stats."""

    loss_mat: torch.Tensor
    seq_log_ratio: Optional[torch.Tensor] = None
    seq_has_ratio: Optional[torch.Tensor] = None
    seq_clipped: Optional[torch.Tensor] = None


def rl_token_losses(
    spec: Mapping[str, Any],
    *,
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    sequence_ratio_mask: Optional[torch.Tensor] = None,
) -> RLTokenLosses:
    """Elementwise Tinker RL losses; ``loss = loss_mat[mask].sum()``.

    Gradients flow only through ``target_logprobs``.
    """
    name = spec["name"]
    squeeze = target_logprobs.dim() == 1
    lp = target_logprobs.unsqueeze(0) if squeeze else target_logprobs
    q = sampling_logprobs.reshape(lp.shape).to(lp.dtype)
    adv = advantages.reshape(lp.shape).to(lp.dtype)

    out: RLTokenLosses
    if name == "importance_sampling":
        out = RLTokenLosses(loss_mat=-(torch.exp(lp - q) * adv))
    elif name == "ppo":
        ratio = torch.exp(lp - q)
        clipped = torch.clamp(ratio, spec["clip_low_threshold"], spec["clip_high_threshold"])
        out = RLTokenLosses(loss_mat=-torch.minimum(ratio * adv, clipped * adv))
    elif name == "cispo":
        ratio = torch.exp(lp - q)
        clipped = torch.clamp(ratio, spec["clip_low_threshold"], spec["clip_high_threshold"])
        out = RLTokenLosses(loss_mat=-(clipped.detach() * lp * adv))
    elif name == "dro":
        out = RLTokenLosses(loss_mat=-(lp * adv - 0.5 * spec["beta"] * (lp - q) ** 2))
    elif name == "gspo":
        if sequence_ratio_mask is None:
            raise ValueError("gspo requires a sequence_ratio_mask")
        in_ratio = sequence_ratio_mask.reshape(lp.shape) != 0
        token_log_ratio = torch.where(in_ratio, lp.detach() - q, torch.zeros_like(lp))
        count = in_ratio.sum(dim=-1)
        seq_log_ratio = token_log_ratio.sum(dim=-1) / count.clamp_min(1).to(lp.dtype)
        # Every token carries the sequence ratio's value with its own logprob's gradient.
        ratio = torch.exp(seq_log_ratio.unsqueeze(-1) + lp - lp.detach())
        low, high = spec["clip_low_threshold"], spec["clip_high_threshold"]
        clipped = torch.clamp(ratio, low, high)
        seq_ratio = torch.exp(seq_log_ratio)
        out = RLTokenLosses(
            loss_mat=-torch.minimum(ratio * adv, clipped * adv),
            seq_log_ratio=seq_log_ratio,
            seq_has_ratio=count > 0,
            seq_clipped=(seq_ratio < low) | (seq_ratio > high),
        )
    else:
        raise ValueError(f"{name!r} is not a Tinker RL loss")

    if squeeze:
        out.loss_mat = out.loss_mat.squeeze(0)
    return out


def masked_loss_mat(loss_mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero masked-out positions without propagating NaN/inf from padding."""
    return torch.where(mask.to(torch.bool), loss_mat, torch.zeros_like(loss_mat))


def cross_entropy_loss(target_logprobs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Tinker ``cross_entropy``: ``-(target_logprobs * weights).sum()`` (any float weights)."""
    return -(target_logprobs * weights.to(target_logprobs.dtype)).sum()


def finalize_gspo_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Fold the internal GSPO SUM counters into Tinker's documented metrics."""
    totals = {}
    for key in list(metrics):
        base = key.split(":", 1)[0]
        if base in (GSPO_LOG_RATIO_TOTAL, GSPO_NUM_SEQUENCES, GSPO_CLIPPED_SEQUENCES):
            totals[base] = metrics.pop(key)
    if not totals:
        return metrics
    num = totals.get(GSPO_NUM_SEQUENCES, 0.0)
    metrics[GSPO_LOG_RATIO_METRIC] = totals.get(GSPO_LOG_RATIO_TOTAL, 0.0) / num if num else 0.0
    metrics[GSPO_CLIPPED_FRACTION_METRIC] = totals.get(GSPO_CLIPPED_SEQUENCES, 0.0) / num if num else 0.0
    return metrics


def loss_metrics_from_logprobs(
    spec: Mapping[str, Any],
    datums: Sequence[Any],
    target_logprobs: Sequence[torch.Tensor],
) -> dict[str, float]:
    """Tinker loss diagnostics for ``/forward`` from per-datum target logprobs.

    Uses the same masking as the training path. Returns ``{}`` when the loss
    cannot be evaluated from the request (``custom_from_config``, or
    ``cross_entropy`` without ``weights``).
    """
    name = spec["name"]
    if name == CUSTOM_FROM_CONFIG:
        return {}
    total = 0.0
    log_ratio_total = 0.0
    num_sequences = 0
    clipped_sequences = 0
    for datum, lp in zip(datums, target_logprobs, strict=True):
        inputs = {key: tensor_data_to_torch(value) for key, value in datum.loss_fn_inputs.items()}
        lp = lp.detach().to(torch.float32).reshape(-1)
        if name == CROSS_ENTROPY:
            if "weights" not in inputs:
                return {}
            total += float(cross_entropy_loss(lp, inputs["weights"].reshape(-1)).item())
            continue
        q = inputs["logprobs"].to(torch.float32)
        adv = inputs["advantages"].to(torch.float32)
        srm = inputs.get("sequence_ratio_mask")
        mask = rl_token_mask(q, adv, srm)
        if name == "gspo" and srm is None:
            srm = mask.to(torch.float32)
        out = rl_token_losses(spec, target_logprobs=lp, sampling_logprobs=q, advantages=adv, sequence_ratio_mask=srm)
        total += float(masked_loss_mat(out.loss_mat, mask).sum().item())
        if name == "gspo" and bool(out.seq_has_ratio[0]):
            num_sequences += 1
            log_ratio_total += float(out.seq_log_ratio[0].item())
            clipped_sequences += int(out.seq_clipped[0].item())
    metrics = {"loss:sum": total}
    if name == "gspo":
        metrics[GSPO_LOG_RATIO_METRIC] = log_ratio_total / num_sequences if num_sequences else 0.0
        metrics[GSPO_CLIPPED_FRACTION_METRIC] = clipped_sequences / num_sequences if num_sequences else 0.0
    return metrics
