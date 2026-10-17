# EnvLoop changes to `verl_tinker` (vs upstream `242c73fc`)

Fork: `EnvLoop/verl-recipe`, base branch `envloop/main` = upstream
`verl-project/verl-recipe@242c73fcadf331e7214c4b48f67423f45239cea8` (byte-identical
to the snapshot EnvLoop production runs). This branch aligns the server's
semantics with Thinking Machines' Tinker documentation so cookbook-style SL/RL
code behaves the same on EnvLoop. Only `verl_tinker/` changed; no dependency or
pin changed (`REQUIRED_VERL.txt`, `install_verl.sh`, `pyproject.toml`,
`uv.lock` untouched; the server still parses requests with its pinned
`tinker==0.19.0` types and stays compatible with SDK 0.30.x clients via the
EnvLoop gateway).

## Behaviour changes

### 1. `gspo` loss (new)

* Accepted everywhere RL losses are accepted: wire schema literal
  (`schemas.TinkerLossFnType`), `loss_fn_config` parsing, router validation,
  the RL TensorDict translator and the loss callback.
* Token-level GSPO exactly as the Tinker docs: per datum the sequence
  log-ratio is the mean of `target_logprob.detach() - sampling_logprob` over
  the tokens marked by `sequence_ratio_mask` (default when the key is absent:
  the sampled tokens, i.e. positions with `logprobs != 0` or
  `advantages != 0`); every token's ratio is
  `exp(seq_log_ratio + lp - lp.detach())` (sequence value, own gradient);
  `loss = -sum(min(ratio * A, clip(ratio, low, high) * A))`.
* Config: `clip_low_threshold` / `clip_high_threshold`, defaults
  `1 - 3e-4` / `1 + 4e-4`.
* Inputs: `target_tokens`, `logprobs`, `advantages`, optional
  `sequence_ratio_mask` (must mark at least one token when given).
* Metrics: `loss:sum`, `gspo_sequence_log_ratio:mean` (mean over sequences of
  log s), `gspo_clipped_fraction:mean` (fraction of sequences whose s fell
  outside the clip range). DP-filler rows are excluded.

### 2. Tinker loss semantics, defaults, input contract, metric names

All RL losses (`importance_sampling`, `ppo`, `cispo`, `dro`, `gspo`) now run a
native implementation of the Tinker formulas (`tinker_losses.rl_token_losses`,
called from `backends/_loss.py`) with token-sum reduction scaled by `dp_size`
(global token sum), instead of being mapped onto VeRL policy losses:

| loss | upstream 242c73fc | now |
|---|---|---|
| `importance_sampling` | VeRL `bypass_mode` REINFORCE with token-TIS **truncated at 2.0**; loss value `-sum(w*logp*A)` | `-sum(exp(lp - q) * A)`, untruncated ratio (Tinker) |
| `ppo` | VeRL vanilla PPO (log-ratio clamped to ±20, dual-clip sentinel) defaults 0.8/1.2 | Tinker formula, defaults 0.8/1.2 (unchanged numerically except the ±20 clamp) |
| `cispo` | VeRL cispo, defaults 0.0/4.0 | Tinker formula, defaults 0.0/4.0 (one-sided) |
| `dro` | VeRL dro, `beta` required and > 0 | Tinker formula, `beta` required (no documented default), `>= 0` |
| `gspo` | rejected | see 1 |
| `custom_from_config` | VeRL `ppo_loss` with startup actor config | unchanged (verl-tinker extension) |

* `loss_fn_config`: values must be finite numbers; unknown keys rejected; the
  wire schema now accepts `float | str` (SDK 0.30 `Mapping[str, float | str]`)
  so strings get a typed 422 message instead of a parse error.
* `loss_fn_inputs` contract (Tinker docs; `tinker_losses.validate_loss_fn_inputs`):
  RL losses accept exactly `target_tokens`, `logprobs`, `advantages` (+
  `sequence_ratio_mask` for gspo); `cross_entropy` accepts `target_tokens`,
  `weights` (1-D, or `(N, K)` for top-K). Extra keys such as `weights`,
  `mask`, `response_mask`, `rollout_is_weights` on Tinker losses are now
  rejected (upstream silently used them to pick the response window / IS
  weights); `custom_from_config` keeps the permissive legacy contract.
  Tensors must have length `len(model_input)` and 1-D `target_tokens` must be
  the shift-by-one of `model_input` (the backend scores the next token of the
  sequence; a mismatch would be silently mis-scored).
* RL losses are summed over **every** target position, like real Tinker
  (confirmed by the EnvLoop e2e against Tinker: `loss:sum` equals the doc
  formula summed over all tokens on the returned learner logprobs). For
  IS/PPO/CISPO/GSPO zero-advantage positions contribute nothing; DRO's
  `0.5 * beta * (lp - q) ** 2` also applies on prompt positions whose sampling
  logprob is padded with 0 (upstream only scored an inferred response window).
  A datum with all-zero advantages is accepted (upstream: 422 "no inferrable
  response window").
* All validation runs synchronously in the router: `/api/v1/forward_backward`
  and `/api/v1/forward` return HTTP 422 with a client-facing message before
  anything is scheduled (upstream surfaced most of these as failed futures).
* Metrics: `loss:sum` for every loss (Tinker name); gspo adds its two
  documented metrics. Tinker's undocumented extras (PPO
  `ppo_clipped_fraction`/`ppo_kl_div`/`ppo_mean_ratio`, `clock_cycle:unique`)
  are not emitted. The VeRL-internal `actor/pg_loss`, `actor/pg_clipfrac`,
  `actor/ppo_kl`, `rollout_corr/*` metrics are no longer emitted for Tinker
  losses (still emitted for `custom_from_config`); VeRL perf metrics remain.
* `cross_entropy` weights may be any float incl. negative (the SDK's
  `forward_backward_custom` linear surrogate sends `weights = -dC/dlogprobs`):
  verified for the 1-D path (gradient equals `dC/dlogprobs`) and fixed for the
  top-K `(N, K)` path, which upstream reconstructed as `exp(log(w))` with
  non-positive weights clamped to `exp(-10)`; raw weights are now used
  (`teacher_weights`), and a position is active when any weight is nonzero.
* **Bug fix, top-K cross_entropy alignment**: upstream placed target row `i`
  at student position `i + 1` (scoring it with the logits that predict token
  `i + 2`) and dropped the last row. Row `i` is now scored by the logits at
  position `i` (which predict token `i + 1`), matching Tinker and VeRL's own
  vLLM prompt-logprob layout.
* Sparse CSR `TensorData` (`sparse_crow_indices` / `sparse_col_indices`, used
  by the SDK for 2-D `target_tokens` / `weights`) is densified; upstream failed
  on it.

### 3. `AdamParams.grad_clip_norm`

* `grad_clip_norm > 0` clips the global gradient norm of the trainable
  parameters to that value before the step; `0` (Tinker default) / negative
  disables clipping. Implemented per step by temporarily overriding the actor
  engine's `optimizer_config.clip_grad` (a VeRL mutable field) inside the
  worker (`TinkerServerActorRolloutRefWorker.optimizer_step_with_grad_clip`),
  restored afterwards; "disabled" is `clip_grad = inf` (clip coefficient 1).
* **Behaviour change**: upstream always clipped at the startup
  `actor.optim.clip_grad` (VeRL default 1.0) and warned that `grad_clip_norm`
  was ignored. Now clients get exactly what they ask for; a request without
  any optimizer params keeps the startup `clip_grad`.
* The optim_step result carries the pre-clip global norm as
  `metrics["unclipped_grad_l2:mean"]` (Tinker's name) and, as before,
  `metrics["grad_norm:mean"]`. An optim_step without gradients leaves the
  weights unchanged (AdamW skips parameters with no gradient). The
  "accepted but not used" warning is removed.
* Dimuon params (`optimizer_params`, SDK 0.30) and a `create_model`
  `optimizer_config` other than AdamW are rejected with HTTP 400
  `{"code": "OPTIMIZER_UNSUPPORTED"}`.

### 4. Sampling

* `topk_sample_logprobs` (SDK 0.30 `SampleRequest` field) is supported:
  each sequence gets `topk_logprobs`, one entry per sampled token: a list of up
  to k `[token_id, logprob]` pairs (descending), or `null` where nothing was
  reported — the shape the EnvLoop gateway encodes into
  `SampledSequence.topk_logprobs`. VeRL's vLLM server forces
  `SamplingParams.logprobs` to 0 (sampled token only) and the VeRL pin cannot
  change, so the candidates come from one extra teacher-forced
  `prompt_logprobs=k` pass over `prompt + completion` on the same rollout
  weights (one extra prefill per sequence, only when k > 0). When the
  completion filled the context window the last position is `null`. Each row
  has exactly k entries, sorted. When the sampling distribution is the
  model's own (temperature 1, top_p 1, no top_k), the sampled token's entry is
  set to its sampled logprob so the two agree exactly, as on Tinker (they come
  from the same pass there); the other entries carry the scoring-pass values.
  `k > server.max_topk_logprobs` (default 20 = vLLM `max_logprobs`) → HTTP
  400 `TOPK_SAMPLE_LOGPROBS_TOO_LARGE`; `k < 0` → 422.
* `target_prompt_logprobs` (SDK 0.30) is **not feasible** on the vLLM rollout
  path (it only reports the actual prompt token and the top-k candidates per
  position; scoring arbitrary ids would need full-vocab prompt logprobs or one
  re-prefill per candidate). Requests carrying it get HTTP 400
  `{"code": "TARGET_PROMPT_LOGPROBS_UNSUPPORTED", "message": ...}` instead of
  being rejected by the schema / silently ignored.
* `record_stability_info` is accepted and ignored (upstream's pinned 0.19
  schema rejected it, so the gateway had to strip it).
* Requests with `topk_sample_logprobs > 0` never take the actor/reference
  scalar prompt-logprob shortcut.

### 5. Forward-only `/api/v1/forward`

* Confirmed no gradient side effects: it runs VeRL `compute_log_prob` →
  `infer_batch` → `forward_backward_batch(forward_only=True)` under
  `torch.no_grad`, in eval mode, without backward and without `zero_grad`;
  gradients accumulated by earlier `forward_backward` calls are untouched for
  the next `optim_step` (GPU script checks this end to end).
* Now honours `loss_fn` / `loss_fn_config` like Tinker's
  `forward(data, loss_fn, loss_fn_config)`: the request is validated like
  `forward_backward` (sync 422) and `metrics` carries `loss:sum` (+ gspo
  metrics) evaluated with the Tinker formulas on the returned logprobs.
  `cross_entropy` without `weights` (and `custom_from_config`) return logprobs
  with empty metrics. `loss_fn_output_type` is the requested loss name.
  Multi-target `(N, K)` `target_tokens` stay unsupported on `/forward` (422).

### 6. LoRA metadata

* `lora_info.configured_lora_info` reads the adapter the server was started
  with (`actor_rollout_ref.model.lora_rank|lora_alpha|target_modules`, or the
  `model.lora` dict when its `rank > 0`); EnvLoop sets the rank via
  `TINKER_LORA_RANK`. Alpha and `target_modules` come from the same section
  as the rank: VeRL's merged defaults always carry a Megatron-style
  `model.lora` dict (`rank: 0`, `alpha: 32`, `linear_qkv/...` modules) next to
  the FSDP keys.
  `train_attn` / `train_mlp` / `train_unembed` are derived from
  `target_modules` (`all-linear` → attn+mlp, no unembed — PEFT never adapts the
  output layer for `all-linear`).
* `get_info` reports `is_lora`, `lora_rank`, `lora_alpha`, `train_*` and
  `optimizer_config: {"type": "adamw"}` (upstream: `is_lora: false`,
  `lora_rank: null`).
* `create_model` metadata (persisted into checkpoint `metadata.json`) records
  the real LoRA fields; `weights_info` returns them (checkpoints saved by
  upstream get the server's adapter filled in). The cookbook-compat shape
  (`is_lora: true, lora_rank: 1`) is kept only for full-parameter servers
  (`lora_rank: 0`), because the SDK's `create_training_client_from_state`
  asserts LoRA metadata.
* `create_model` requesting a rank different from the configured one → HTTP
  409 with a clear message (nothing registered). Differing `train_*` flags are
  logged, not rejected (the SDK default `train_unembed=True` would otherwise
  break every all-linear server). Full-parameter servers accept any rank.

## Wire contract summary (head JSON, as forwarded by the EnvLoop gateway)

* `POST /api/v1/forward`: `{"forward_input": {"data", "loss_fn", "loss_fn_config"}, "model_id", "seq_id"}`
  → future `{"type": "forward_backward", "loss_fn_output_type", "loss_fn_outputs": [{"logprobs": TensorData}], "metrics"}`.
* `POST /api/v1/forward_backward`: losses `cross_entropy | importance_sampling | ppo | cispo | dro | gspo | custom_from_config`; sync 422 on contract violations.
* `POST /api/v1/asample`: accepts `topk_sample_logprobs`, `record_stability_info`; `target_prompt_logprobs` → 400 typed; sequences may carry `topk_logprobs`.
* `POST /api/v1/optim_step`: `adam_params.grad_clip_norm` honoured; `metrics["unclipped_grad_l2:mean"]` (= `grad_norm:mean`) = pre-clip norm.
* `POST /api/v1/create_model`: 409 on rank mismatch; accepts SDK 0.30 `optimizer_config` (AdamW only).
* Typed 4xx errors use `HTTPException(detail={"code", "message"})`.

## Known limitations

* `forward_backward` with `(N, K)` targets still returns 1-D next-token
  logprobs in `loss_fn_outputs` (Tinker returns `(N, K)`).
* The VeOmni fused top-K path still consumes `teacher_logprobs`
  (`log(w)`, non-positive weights clamped), so signed top-K weights are exact
  only on the eager FSDP path (EnvLoop's configuration).
* `topk_sample_logprobs` values come from vLLM prompt logprobs of the
  teacher-forced sequence; with `temperature != 1` they are vLLM's
  prompt-logprob semantics, which may differ from the processed sampling
  logprobs (`logprobs_mode=processed_logprobs`).
* `target_prompt_logprobs`: not implemented (see 4).
* Pre-existing, not caused by this branch (reproduced on upstream
  `242c73fc`): with the FSDP1 actor (`strategy: fsdp`) and a tied-embedding
  model (Qwen3-0.6B, LoRA), `forward` after `forward_backward` and before
  `optim_step` fails inside FSDP1 (`as_params=True
  type(prim_param)=<class 'torch.Tensor'>`) and the actor stays broken
  (`Changing shared parameters is not supported yet`, `optim_step`:
  `expected to be in states [IDLE] but current state is FORWARD_BACKWARD`).
  FSDP2 (EnvLoop's Qwen3.5-4B profile, prepared by the guest's
  `recipe_server.prepare_config`) is not affected.

## Files

New: `src/verl_tinker/tinker_losses.py`, `src/verl_tinker/lora_info.py`,
`tests/test_envloop_tinker_alignment.py`, `tests/test_envloop_router_alignment.py`,
`tests/gpu/test_envloop_alignment_gpu.py`, `ENVLOOP_CHANGES.md`.

Changed: `src/verl_tinker/tinker_router.py`, `src/verl_tinker/tinker_ops.py`,
`src/verl_tinker/schemas.py`, `src/verl_tinker/backends/_loss.py`,
`src/verl_tinker/backends/colocated.py`,
`src/verl_tinker/data/datum_processing.py`, `README.md`,
`tests/test_dynamic_loss.py`, `tests/test_tinker_ops_sample.py`,
`tests/test_tinker_router.py`, `tests/test_colocated_backend.py`.

## Validation

* CPU (no GPU, real VeRL `cfacd76a` Python code installed without vLLM, CPU
  torch, `tinker==0.19.0` = server pin): `python -m pytest -q tests` from
  `verl_tinker/`. The loss tests run the real translator + loss callback +
  VeRL `no_padding_2_padding` / `agg_loss` and compare loss values and
  gradients with literal transcriptions of the Tinker doc formulas; the grad
  clip tests run VeRL's real `TinkerTrainingWorker.optimizer_step` and
  `FSDPEngine.optimizer_step` on a CPU module.
* GPU: `tests/gpu/test_envloop_alignment_gpu.py` (skipped without CUDA); see
  its module docstring for the three ways to point it at a server. The
  self-started server defaults to an FSDP2 actor prepared like EnvLoop's guest
  (`TINKER_ALIGN_STRATEGY=fsdp` selects FSDP1, see Known limitations).
  2026-10-16, 1x H100 80GB SXM (Vast, driver 610.57, CUDA 13.0 wheels, vLLM
  0.26.0 from `uv.lock`), installed with `install_verl.sh --recipe
  verl_tinker` as on EnvLoop guests: 11/11 with Qwen/Qwen3-0.6B LoRA rank 8
  (self-started, FSDP2) and 11/11 with Qwen/Qwen3.5-4B LoRA rank 16 through
  EnvLoop's `trainingmaster-actor-rollout.yaml` + production env (FSDP2,
  rollout TP 1 / 0.4); CPU suite 253 passed on the same box. Every RL
  `loss:sum` matched the formula on the returned logprobs to float precision;
  forward vs forward_backward logprobs identical and the grad norm after
  forward_backward + 2 forwards equal to the forward_backward-only norm;
  clipped step vs lr/10 reference 0.6-0.7 (bf16 noise floor) vs 28-35 for
  the unclipped step; top-k rows 5 wide; prompt_logprobs with
  topk_prompt_logprobs identical to the compute_logprobs request.

## Weights-only checkpoint loads reset the optimizer

`load_weights` with `optimizer=false` (Tinker `create_training_client_from_state`) used to load the VeRL
checkpoint — parameters **and** optimizer state — and only clear gradients, so the restored Adam moments
silently carried over (found by the EnvLoop prod e2e: steps after a weights-only and a with-optimizer
resume differed by 2.6e-3 instead of ~0.1 on Tinker). The backend now calls the new worker method
`reset_optimizer_state`, which clears the torch optimizer's per-parameter state and gradients, so the next
step is a first step from the saved weights. `optimizer=true` is unchanged.

## Oversized weights are sent from the device (Qwen3.8-27B rollout sync)

VeRL's bucketed rollout weight transfer shares any weight larger than the 2 GB bucket directly with
`reduce_tensor(weight)`. With an FSDP1 actor the gathered full weights are CPU tensors, so the handle is a
CPU-storage handle and the vLLM receiver fails in `rebuild_ipc` (`list_args[6] = device_id`: IndexError),
killing the session at the first `save_weights_for_sampler` (EnvLoop prod e2e, Qwen3.8-27B, 2× GPU,
FSDP1, TP 2: the ~2.5 GB embedding exceeds the bucket; Qwen3.5-4B never takes this path). The backend
module now patches `BucketedWeightSender._direct_send_large_weight` at import (in every Ray worker that
loads the worker classes) to copy host tensors to the sender's device first.

## LoRA rollout sync gathers one FSDP unit at a time (Qwen3.8-27B on 80 GB cards)

`collect_lora_params(layered_summon=False)` summons the whole FSDP1 model to read the adapter: ~54 GB
per GPU for Qwen3.8-27B on top of the awake vLLM weights, an OOM at the first `save_weights_for_sampler`
on 2× H100 80 GB (EnvLoop prod e2e 2026-10-17; it only fit on B200). VeRL's `layered_summon_lora_params`
would gather per unit, but it tests the *unit-relative* parameter names for `lora_`, and VeRL's LoRA wrap
policy wraps every trainable LoRA leaf (`...q_proj.lora_A.default`) as its own FSDP unit whose only
parameter is `weight`, so it returns nothing and VeRL falls back to a full summon (to host memory).
The backend module replaces it at import with `_layered_lora_params`: it tests full names, gathers only
units that own adapter tensors, and raises unless it returns every adapter tensor of the model
(`tests/test_layered_lora_collection.py` runs both collectors on VeRL's wrapping of a tiny Llama).
Layered summon needs vLLM to load the base weights itself (`rollout.load_format: safetensors`); EnvLoop
sets both for its Qwen3.8-27B profile.

## Combined prompt logprobs stay on the sampler (reverts fa571c9)

fa571c9 scored the scalar `prompt_logprobs` of a request that also asks for `topk_prompt_logprobs` on the
actor, on the belief that `compute_logprobs` runs there. It does not when the rollout holds the sampler's
weights (`ModelResourceManager.resolve_resource` returns `ROLLOUT`), so the two APIs came from different
engines: EnvLoop prod e2e Qwen3.5-4B max abs diff 0.084 (was exactly 0.0 with the sampler pass), and the
recipe GPU suite failed on A100. The scalar pass is back on the sampler. The 0.118 gap first seen on
Qwen3.8-27B (2× B200, TP 2) is between two sampler passes and is tracked separately.

## Tensor-parallel vLLM rollout all-reduces through NCCL by default

vLLM's custom all-reduce writes to the peer GPU through CUDA IPC handles after a P2P probe that can pass
on hosts where those kernel stores fault. On a rented 2× H100 box without NVLink (EnvLoop prod
2026-10-17, Qwen3.8-27B, rollout TP 2), vLLM's CUDA-graph memory profiling hit "an illegal memory
access" and `EngineCore failed to start`. Another 2× H100 host ran the same config fine. For a vLLM
rollout with `tensor_model_parallel_size > 1`, `config_utils` now sets
`rollout.engine_kwargs.vllm.disable_custom_all_reduce: true` unless the config sets it, so NCCL picks a
working transport itself. Upstream's own TP-2 config (`configs/advance/qwen3_8b_actor_rollout_lora.yaml`)
already sets this. TP 1 rollouts have no all-reduce and are unchanged.
