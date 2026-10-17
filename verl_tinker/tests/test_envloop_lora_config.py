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

"""Tinker LoraConfig parity on CPU: explicit target modules, seed, reporting, vLLM lm_head plugin."""

import re
from http import HTTPStatus
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from omegaconf import OmegaConf
from test_envloop_router_alignment import _server
from verl_tinker import vllm_backports
from verl_tinker.backends.colocated import lora_init_seed
from verl_tinker.lora_info import configured_lora_info

# EnvLoop's PEFT modules for Qwen3.5 per Tinker layer family (EnvPlatform lora_targets.py).
ATTN = ["in_proj_qkv", "in_proj_z", "k_proj", "o_proj", "out_proj", "q_proj", "v_proj"]
MLP = ["down_proj", "gate_proj", "up_proj"]
UNEMBED = ["lm_head"]

# tinker_cookbook 0.5.7 hyperparam_utils._LORA_PARAMS_PER_RANK_BY_COMPONENT
TINKER_PER_RANK = {
    "Qwen/Qwen3.5-4B": {"mlp": 1_130_496, "attn": 897_024, "unembed": 250_880},
    "Qwen/Qwen3.8-27B": {"mlp": 4_325_376, "attn": 2_965_504, "unembed": 253_440},
}

# The text configs of the pinned checkpoints (Qwen/Qwen3.5-4B@851bf6e8, Qwen/Qwen3.8-27B@1d4bf0f2),
# reduced to the fields that shape the LoRA-adapted layers.
_LAYERS_4B = ["linear_attention"] * 3 + ["full_attention"]
TEXT_CONFIGS = {
    "Qwen/Qwen3.5-4B": dict(
        hidden_size=2560, intermediate_size=9216, num_hidden_layers=32, layer_types=_LAYERS_4B * 8,
        num_attention_heads=16, num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
        linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128, vocab_size=248320,
        tie_word_embeddings=True,
    ),
    "Qwen/Qwen3.8-27B": dict(
        hidden_size=5120, intermediate_size=17408, num_hidden_layers=64, layer_types=_LAYERS_4B * 16,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
        linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128, vocab_size=248320,
        tie_word_embeddings=False,
    ),
}


def _lora_param_count(model_name, target_modules, rank=1):
    """Trainable LoRA parameters PEFT creates on the real architecture (meta device, no weights)."""
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config_cls = getattr(transformers, "Qwen3_5TextConfig", None)
    if config_cls is None:
        pytest.skip("transformers without Qwen3.5")
    config = config_cls(**TEXT_CONFIGS[model_name])
    with torch.device("meta"):
        model = transformers.Qwen3_5ForCausalLM(config)
    model = peft.get_peft_model(model, peft.LoraConfig(r=rank, lora_alpha=32, target_modules=target_modules))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("model_name", sorted(TINKER_PER_RANK))
def test_target_modules_reproduce_tinker_lora_param_counts(model_name):
    tinker = TINKER_PER_RANK[model_name]
    config = TEXT_CONFIGS[model_name]
    assert _lora_param_count(model_name, MLP) == tinker["mlp"]
    assert _lora_param_count(model_name, UNEMBED) == tinker["unembed"]
    assert _lora_param_count(model_name, MLP, rank=24) == 24 * tinker["mlp"]
    # Tinker trains separate in_proj_q/k/v LoRAs on the Gated DeltaNet layers; the HF model's
    # fused in_proj_qkv shares one A, i.e. 2 * hidden_size fewer parameters per rank and GDN layer.
    gdn_layers = config["layer_types"].count("linear_attention")
    assert _lora_param_count(model_name, ATTN) == tinker["attn"] - 2 * config["hidden_size"] * gdn_layers


def _flags(target_modules):
    info = configured_lora_info(
        {"actor_rollout_ref": {"model": {"lora_rank": 8, "lora_alpha": 32, "target_modules": target_modules}}}
    )
    return info.train_attn, info.train_mlp, info.train_unembed


def test_train_flags_follow_explicit_qwen3_5_target_modules():
    assert _flags(ATTN + MLP + UNEMBED) == (True, True, True)
    assert _flags(ATTN) == (True, False, False)
    assert _flags(["in_proj_qkv", "in_proj_z", "out_proj"]) == (True, False, False)
    assert _flags(MLP) == (False, True, False)
    assert _flags(UNEMBED) == (False, False, True)
    assert _flags("all-linear") == (True, True, False)


def test_guest_yaml_target_regex_and_seed_survive_in_place_resolve(monkeypatch):
    yaml = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "lora_rank": 24,
                    "lora_alpha": 32,
                    "target_modules": "${oc.env:TINKER_LORA_TARGET_MODULES,all-linear}",
                    "lora": {"seed": "${oc.decode:${oc.env:TINKER_LORA_SEED,null}}"},
                }
            }
        }
    )
    # EnvPlatform lora_targets.peft_regex: PEFT full-matches it against module paths.
    regex = r"(?:.*\.)?(?:down_proj|gate_proj|lm_head|up_proj)"
    monkeypatch.setenv("TINKER_LORA_TARGET_MODULES", regex)
    monkeypatch.setenv("TINKER_LORA_SEED", "1234")
    OmegaConf.resolve(yaml)  # the recipe resolves its config in place
    assert yaml.actor_rollout_ref.model.target_modules == regex
    assert re.fullmatch(regex, "model.language_model.layers.0.mlp.down_proj")
    assert re.fullmatch(regex, "lm_head")
    assert not re.fullmatch(regex, "model.visual.blocks.0.mlp.linear_fc1")
    assert not re.fullmatch(regex, "model.language_model.layers.3.self_attn.q_proj")
    info = configured_lora_info(yaml)
    assert info.rank == 24 and info.target_modules == ["down_proj", "gate_proj", "lm_head", "up_proj"]
    assert (info.train_attn, info.train_mlp, info.train_unembed) == (False, True, True)
    assert info.metadata()["target_modules"] == ["down_proj", "gate_proj", "lm_head", "up_proj"]
    model = OmegaConf.to_container(yaml.actor_rollout_ref.model, resolve=True)
    assert lora_init_seed(SimpleNamespace(**model)) == 1234

    assert lora_init_seed(SimpleNamespace(lora={"seed": None}, lora_adapter_path=None)) is None
    # A continued adapter keeps its saved weights.
    assert lora_init_seed(SimpleNamespace(lora={"seed": 3}, lora_adapter_path="/a")) is None


def _explicit_config(target_modules):
    return {
        "server": {"model_name": "Qwen/Qwen3.5-4B"},
        "actor_rollout_ref": {
            "model": {"path": "/m", "lora_rank": 8, "lora_alpha": 32, "target_modules": target_modules}
        },
    }


@pytest.mark.asyncio
async def test_create_model_rejects_other_layer_selection_for_explicit_targets():
    server = _server(_explicit_config(ATTN + MLP))
    session_id = server._model_resource_manager.create_session()
    with pytest.raises(HTTPException) as exc_info:
        await server.create_model(
            SimpleNamespace(
                session_id=session_id,
                model_seq_id=0,
                base_model="Qwen/Qwen3.5-4B",
                lora_config=SimpleNamespace(rank=8, train_attn=True, train_mlp=True, train_unembed=True),
            )
        )
    assert exc_info.value.status_code == HTTPStatus.CONFLICT
    assert "train_unembed" in exc_info.value.detail
    assert server._model_resource_manager.model_ids_for_session(session_id) == []

    response = await server.create_model(
        SimpleNamespace(
            session_id=session_id,
            model_seq_id=0,
            base_model="Qwen/Qwen3.5-4B",
            lora_config=SimpleNamespace(rank=8, train_attn=True, train_mlp=True, train_unembed=False),
        )
    )
    metadata = server._model_metadata[response["model_id"]]
    assert metadata["train_unembed"] is False and metadata["lora_alpha"] == 32.0
    assert metadata["target_modules"] == sorted(ATTN + MLP)


def _peft_engine_self(seed, target_modules):
    model_config = SimpleNamespace(
        lora_rank=4,
        lora_alpha=32,
        target_modules=target_modules,
        target_parameters=None,
        exclude_modules=None,
        lora_adapter_path=None,
        lora={"seed": seed},
    )
    return SimpleNamespace(model_config=model_config)


def _tiny_module():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128,
    )
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config)


def _lora_a(module):
    return {k: v.detach().clone() for k, v in module.named_parameters() if "lora_A" in k}


def test_lora_seed_makes_adapter_init_reproducible_and_restores_rng():
    pytest.importorskip("peft")
    engine = pytest.importorskip("verl.workers.engine.fsdp.transformer_impl")
    build = engine.FSDPEngine._build_lora_module
    assert getattr(build, "_envloop_seed_patch", False), "colocated import must patch the FSDP engine"

    targets = ["q_proj", "v_proj", "lm_head"]
    module = _tiny_module()
    rng_before = torch.random.get_rng_state()
    first = _lora_a(build(_peft_engine_self(7, targets), module))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    second = _lora_a(build(_peft_engine_self(7, targets), _tiny_module()))
    other = _lora_a(build(_peft_engine_self(8, targets), _tiny_module()))

    assert first and first.keys() == second.keys()
    assert any(".lm_head." in key for key in first)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert not all(torch.equal(first[key], other[key]) for key in first)


def test_vllm_backports_apply_once_and_refuse_foreign_code(tmp_path):
    """Each shipped diff applies to a tree holding its original code, is idempotent and fails closed."""
    for fix in vllm_backports.FIXES:
        diff = vllm_backports.PATCHES / f"{fix}-0.26.0.diff"
        files = vllm_backports.hunks(diff.read_text())
        assert files, fix
        root = tmp_path / fix / "vllm"
        for relative, file_hunks in files.items():
            target = root.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# head\n" + "".join(f"{old}# --\n" for old, _ in file_hunks))
        assert sorted(vllm_backports.apply(root, "0.26.0", fixes=(fix,))) == sorted(files)
        for relative, file_hunks in files.items():
            text = (root.parent / relative).read_text()
            assert all(text.count(new) == 1 for _, new in file_hunks)
        assert vllm_backports.apply(root, "0.26.0", fixes=(fix,)) == []  # idempotent
        assert vllm_backports.apply(root, "9.9.9", fixes=(fix,)) == []  # no backport for that release
        first = next(iter(files))
        (root.parent / first).write_text("unrelated code\n")
        with pytest.raises(vllm_backports.BackportError):
            vllm_backports.apply(root, "0.26.0", fixes=(fix,))


def test_qwen3_5_backport_declares_lm_head_lora():
    diff = (vllm_backports.PATCHES / "qwen3_5-lm-head-lora-0.26.0.diff").read_text()
    [(path, [(old, new)])] = vllm_backports.hunks(diff).items()
    assert path == "vllm/model_executor/models/qwen3_5.py"
    assert '"lm_head": "output_embeddings"' in new and "embedding_modules" not in old
