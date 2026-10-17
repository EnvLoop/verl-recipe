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

"""Tinker LoraConfig parity on CPU: explicit target modules, seed, reporting, vLLM backports."""

import copy
import re
from http import HTTPStatus
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from omegaconf import OmegaConf
from test_envloop_router_alignment import _server
from verl_tinker import capabilities, vllm_backports, vllm_split_qkv
from verl_tinker.backends.colocated import lora_init_seed
from verl_tinker.lora_info import configured_lora_info
from verl_tinker.lora_split import lora_config_class, mark_split_qkv, split_legacy_tensors, split_peft_name

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
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        layer_types=_LAYERS_4B * 8,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=248320,
        tie_word_embeddings=True,
    ),
    "Qwen/Qwen3.8-27B": dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        layer_types=_LAYERS_4B * 16,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=248320,
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
    mark_split_qkv(model)
    model = peft.get_peft_model(model, lora_config_class()(r=rank, lora_alpha=32, target_modules=target_modules))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("model_name", sorted(TINKER_PER_RANK))
def test_target_modules_reproduce_tinker_lora_param_counts(model_name):
    tinker = TINKER_PER_RANK[model_name]
    assert _lora_param_count(model_name, MLP) == tinker["mlp"]
    assert _lora_param_count(model_name, UNEMBED) == tinker["unembed"]
    assert _lora_param_count(model_name, MLP, rank=24) == 24 * tinker["mlp"]
    # The Gated DeltaNet in_proj_qkv carries separate q/k/v LoRAs (lora_split), like the reference.
    assert _lora_param_count(model_name, ATTN) == tinker["attn"]
    assert _lora_param_count(model_name, ATTN + MLP + UNEMBED, rank=8) == 8 * sum(tinker.values())


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
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
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


def test_capability_reports_only_verified_backports(tmp_path, monkeypatch):
    root = tmp_path / "vllm"
    for fix in vllm_backports.FIXES:
        for relative, file_hunks in vllm_backports.hunks(
            (vllm_backports.PATCHES / f"{fix}-0.26.0.diff").read_text()
        ).items():
            target = root.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                handle.write("".join(f"{old}# --\n" for old, _ in file_hunks))
    assert vllm_backports.present(root, "0.26.0") is False
    monkeypatch.setattr(vllm_backports, "_installed_root_and_version", lambda: (root, "0.26.0"))
    monkeypatch.setattr(vllm_split_qkv, "enabled", lambda vllm_root: vllm_root == root)
    with pytest.raises(vllm_backports.BackportError):
        capabilities.assert_startup_capabilities()
    assert capabilities.runtime_capabilities() == {"lm_head_lora_prompt_logprobs": False, "lora_split_qkv": True}

    vllm_backports.apply_installed()
    capabilities.assert_startup_capabilities()
    assert capabilities.runtime_capabilities() == {"lm_head_lora_prompt_logprobs": True, "lora_split_qkv": True}
    assert vllm_backports.present(root, "9.9.9")  # a release without diffs ships the fixes
    capabilities.runtime_capabilities.cache_clear()


def test_split_qkv_capability_needs_the_plugin_loaded(tmp_path, monkeypatch):
    from importlib.metadata import EntryPoint

    root = tmp_path / "vllm"
    for relative, marker in vllm_split_qkv.HOOKS.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(f"x = 1\n{marker}\n")
    plugin = EntryPoint("verl_tinker_split_qkv_lora", vllm_split_qkv.ENTRY_POINT, "vllm.general_plugins")
    installed = [plugin]
    monkeypatch.setattr(vllm_split_qkv.importlib.metadata, "entry_points", lambda group: list(installed))
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    assert vllm_split_qkv.enabled(root)
    assert not vllm_split_qkv.enabled(None)  # no vLLM
    monkeypatch.setenv("VLLM_PLUGINS", "some_other_plugin")
    assert not vllm_split_qkv.enabled(root)  # excluded by vLLM's plugin allowlist
    monkeypatch.setenv("VLLM_PLUGINS", "some_other_plugin,verl_tinker_split_qkv_lora")
    assert vllm_split_qkv.enabled(root)
    (root / "lora/model_manager.py").write_text("x = 1\n")
    assert not vllm_split_qkv.enabled(root)  # a vLLM without the hook the plugin wraps
    (root / "lora/model_manager.py").write_text(vllm_split_qkv.HOOKS["lora/model_manager.py"])
    installed.clear()
    assert not vllm_split_qkv.enabled(root)  # package installed without its entry point

    monkeypatch.setattr(vllm_backports, "_installed_root_and_version", lambda: (root, "9.9.9"))
    monkeypatch.setattr(vllm_backports, "present_installed", lambda: True)
    capabilities.runtime_capabilities.cache_clear()
    with pytest.raises(vllm_backports.BackportError, match="split q/k/v"):
        capabilities.assert_startup_capabilities()
    assert capabilities.runtime_capabilities()["lora_split_qkv"] is False
    installed.append(plugin)
    capabilities.assert_startup_capabilities()
    assert capabilities.runtime_capabilities()["lora_split_qkv"] is True
    capabilities.runtime_capabilities.cache_clear()


class _GDN(torch.nn.Module):
    """The shape of a Gated DeltaNet block as far as LoRA is concerned."""

    def __init__(self, hidden=8, key_dim=4, value_dim=6):
        super().__init__()
        self.key_dim, self.value_dim = key_dim, value_dim
        self.in_proj_qkv = torch.nn.Linear(hidden, 2 * key_dim + value_dim, bias=False)
        self.out_proj = torch.nn.Linear(value_dim, hidden, bias=False)

    def forward(self, x):
        return self.in_proj_qkv(x)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = _GDN()

    def forward(self, x):
        return self.linear_attn(x)


def _split_model(rank=2, alpha=32):
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = _Block()
    assert mark_split_qkv(model) == 1
    model = peft.get_peft_model(model, lora_config_class()(r=rank, lora_alpha=alpha, target_modules=["in_proj_qkv"]))
    layer = model.base_model.model.linear_attn.in_proj_qkv
    for param in layer.lora_B["default"].parameters():  # B starts at zero; make the delta visible
        torch.nn.init.normal_(param)
    return model, layer


def test_split_qkv_lora_is_three_independent_loras_on_the_fused_linear():
    model, layer = _split_model(rank=2, alpha=32)
    x = torch.randn(3, 8)
    lora_a, lora_b = layer.lora_A["default"], layer.lora_B["default"]
    expected = layer.base_layer(x) + 16.0 * torch.cat(
        [lora_b.q(lora_a.q(x)), lora_b.k(lora_a.k(x)), lora_b.v(lora_a.v(x))], dim=-1
    )
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(layer.get_delta_weight("default") @ x.T, (expected - layer.base_layer(x)).T)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters() if p.requires_grad}
    prefix = "base_model.model.linear_attn.in_proj_qkv"
    assert shapes == {
        f"{prefix}.lora_A.default.q.weight": (2, 8),
        f"{prefix}.lora_A.default.k.weight": (2, 8),
        f"{prefix}.lora_A.default.v.weight": (2, 8),
        f"{prefix}.lora_B.default.q.weight": (4, 2),
        f"{prefix}.lora_B.default.k.weight": (4, 2),
        f"{prefix}.lora_B.default.v.weight": (6, 2),
    }
    # Each A is initialised like PEFT's / nn.Linear's (independently, not copies).
    assert not torch.equal(lora_a.q.weight, lora_a.k.weight)


def test_split_names_and_legacy_fused_adapters_convert_exactly():
    peft = pytest.importorskip("peft")
    from peft.utils.save_and_load import get_peft_model_state_dict

    model, _ = _split_model()
    names = sorted(split_peft_name(name) for name in get_peft_model_state_dict(model))
    assert names == sorted(f"base_model.model.linear_attn.in_proj_{s}.lora_{ab}.weight" for ab in "AB" for s in "qkv")
    assert split_peft_name("x.out_proj.lora_A.weight") == "x.out_proj.lora_A.weight"

    # A legacy adapter: one LoRA on the fused Linear (shared A).
    torch.manual_seed(1)
    legacy = peft.get_peft_model(_GDN(), peft.LoraConfig(r=2, lora_alpha=32, target_modules=["in_proj_qkv"]))
    fused = legacy.base_model.model.in_proj_qkv
    torch.nn.init.normal_(fused.lora_B["default"].weight)
    tensors = {
        "m.in_proj_qkv.lora_A.weight": fused.lora_A["default"].weight.detach(),
        "m.in_proj_qkv.lora_B.weight": fused.lora_B["default"].weight.detach(),
    }
    split = split_legacy_tensors(tensors, (4, 4, 6))
    assert sorted(split) == sorted(f"m.in_proj_{s}.lora_{ab}.weight" for ab in "AB" for s in "qkv")
    x = torch.randn(3, 8)
    delta = torch.cat(
        [x @ split[f"m.in_proj_{s}.lora_A.weight"].T @ split[f"m.in_proj_{s}.lora_B.weight"].T for s in "qkv"], dim=-1
    )
    torch.testing.assert_close(delta * 16.0, legacy(x) - fused.base_layer(x))


def test_engine_lora_build_uses_split_qkv_and_restores_the_config_class():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    engine = pytest.importorskip("verl.workers.engine.fsdp.transformer_impl")
    if getattr(transformers, "Qwen3_5TextConfig", None) is None:
        pytest.skip("transformers without Qwen3.5")
    config = transformers.Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=_LAYERS_4B,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=256,
    )
    with torch.device("meta"):
        model = transformers.Qwen3_5ForCausalLM(config)
    default_config = engine.LoraConfig
    regex = r"(?:.*\.)?(?:in_proj_qkv|in_proj_z|out_proj|q_proj|k_proj|v_proj|o_proj|lm_head)"
    built = engine.FSDPEngine._build_lora_module(_peft_engine_self(None, regex), model)
    assert engine.LoraConfig is default_config
    layer = built.base_model.model.model.layers[0].linear_attn.in_proj_qkv
    assert type(layer).__name__ == "SplitQKVLoraLinear"
    assert type(built.base_model.model.model.layers[3].self_attn.q_proj).__name__ == "Linear"


def test_legacy_fused_checkpoint_resumes_exactly_in_the_split_layer():
    peft = pytest.importorskip("peft")
    from verl_tinker.lora_split import convert_legacy_full, full_tensor, legacy_split_names

    torch.manual_seed(2)
    base = _Block()
    legacy = peft.get_peft_model(
        copy.deepcopy(base), peft.LoraConfig(r=2, lora_alpha=32, target_modules=["in_proj_qkv", "out_proj"])
    )
    for name, param in legacy.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)
    saved = {k: v for k, v in legacy.state_dict().items() if "lora_" in k}  # what a LoRA-only checkpoint holds
    names = legacy_split_names(saved)
    assert sorted(names) == [
        "base_model.model.linear_attn.in_proj_qkv.lora_A.default.weight",
        "base_model.model.linear_attn.in_proj_qkv.lora_B.default.weight",
    ]
    full = {key: full_tensor([value]) for key, value in saved.items()}
    converted = convert_legacy_full(full, names, lambda key: (4, 4, 6))

    split = copy.deepcopy(base)
    mark_split_qkv(split)
    split = peft.get_peft_model(
        split, lora_config_class()(r=2, lora_alpha=32, target_modules=["in_proj_qkv", "out_proj"])
    )
    result = split.load_state_dict(converted, strict=False)
    assert not result.unexpected_keys
    x = torch.randn(3, 8)
    torch.testing.assert_close(split(x), legacy(x))


def test_full_tensor_merges_sharded_tensor_shards():
    class _Meta:
        def __init__(self, offsets, sizes):
            self.shard_offsets, self.shard_sizes = offsets, sizes

    class _Shard:
        def __init__(self, tensor, offsets):
            self.tensor, self.metadata = tensor, _Meta(offsets, list(tensor.shape))

    class _Sharded:  # the parts of torch's ShardedTensor that full_tensor uses
        dtype = torch.float32

        def __init__(self, shards):
            self._shards = shards

        def local_shards(self):
            return self._shards

        def metadata(self):
            return None

        def size(self):
            return (4, 3)

    full = torch.arange(12.0).reshape(4, 3)
    parts = [_Sharded([_Shard(full[:2], [0, 0])]), _Sharded([_Shard(full[2:], [2, 0])])]
    from verl_tinker.lora_split import full_tensor

    torch.testing.assert_close(full_tensor(parts), full)


def test_legacy_optimizer_state_converts_so_the_next_step_matches_on_b():
    """After resuming a fused checkpoint *with* its Adam state, the first step updates B exactly like the fused
    layout would; A starts from the same moments (copied) but gets per-slice gradients, so it differs by design."""
    peft = pytest.importorskip("peft")
    from verl_tinker.lora_split import convert_legacy_full, convert_legacy_optimizer_state, legacy_split_names

    torch.manual_seed(4)
    base = _Block()
    x, target = torch.randn(5, 8), torch.randn(5, 14)

    def train_step(model, optimizer):
        optimizer.zero_grad()
        ((model(x) - target) ** 2).sum().backward()
        optimizer.step()

    legacy = peft.get_peft_model(
        copy.deepcopy(base), peft.LoraConfig(r=2, lora_alpha=32, target_modules=["in_proj_qkv"])
    )
    legacy_params = [p for p in legacy.parameters() if p.requires_grad]
    legacy_opt = torch.optim.AdamW(legacy_params, lr=1e-2)
    for _ in range(3):
        train_step(legacy, legacy_opt)
    names = [n for n, p in legacy.named_parameters() if p.requires_grad]
    saved_model = {n: p.detach().clone() for n, p in legacy.named_parameters() if p.requires_grad}
    saved_opt = legacy_opt.state_dict()
    full_states = {names[i]: saved_opt["state"][i] for i in range(len(names))}

    split = copy.deepcopy(base)
    mark_split_qkv(split)
    split = peft.get_peft_model(split, lora_config_class()(r=2, lora_alpha=32, target_modules=["in_proj_qkv"]))
    mapping = legacy_split_names(saved_model)
    sizes = lambda key: (4, 4, 6)  # noqa: E731
    split.load_state_dict(convert_legacy_full(saved_model, mapping, sizes), strict=False)
    split_names = [n for n, p in split.named_parameters() if p.requires_grad]
    split_opt = torch.optim.AdamW([p for p in split.parameters() if p.requires_grad], lr=1e-2)
    new_states = convert_legacy_optimizer_state(full_states, mapping, sizes)
    split_opt.load_state_dict(
        {
            "state": {i: new_states[n] for i, n in enumerate(split_names)},
            "param_groups": split_opt.state_dict()["param_groups"],
        }
    )
    torch.testing.assert_close(split(x), legacy(x))

    train_step(legacy, legacy_opt)
    train_step(split, split_opt)
    fused_b = legacy.base_model.model.linear_attn.in_proj_qkv.lora_B["default"].weight
    split_b = split.base_model.model.linear_attn.in_proj_qkv.lora_B["default"]
    torch.testing.assert_close(
        torch.cat([split_b.q.weight, split_b.k.weight, split_b.v.weight]), fused_b, rtol=0, atol=1e-6
    )
    assert all(split_opt.state[p]["step"] == 4 for p in split_opt.state)


def test_fsdp1_optimizer_state_round_trips_through_flat_rank_views():
    from verl_tinker.backends.colocated import _full_state_value, _local_state_value

    full = torch.arange(15.0).reshape(5, 3)  # 15 elements over 2 ranks: chunks of 8 (padded to 16)
    views = [full.reshape(-1)[:8], full.reshape(-1)[8:]]
    torch.testing.assert_close(_full_state_value(views, (5, 3)), full)
    for rank, view in enumerate(views):
        param = torch.empty(view.shape)
        torch.testing.assert_close(_local_state_value(full, param, 2, rank), view)
    assert _full_state_value([torch.tensor(3.0), torch.tensor(3.0)], (5, 3)).item() == 3.0


def test_legacy_optimizer_load_maps_verl_whole_model_param_order(tmp_path):
    """VeRL's optimizer holds every parameter (frozen ones without state); the loader maps indices by the
    legacy parameter order and restores the converted state into the split model's optimizer."""
    peft = pytest.importorskip("peft")
    from verl_tinker.backends.colocated import _load_legacy_optimizer
    from verl_tinker.lora_split import legacy_split_names

    torch.manual_seed(5)
    base = _Block()
    x, target = torch.randn(5, 8), torch.randn(5, 14)
    targets = ["in_proj_qkv", "out_proj"]
    legacy = peft.get_peft_model(copy.deepcopy(base), peft.LoraConfig(r=2, lora_alpha=32, target_modules=targets))
    legacy_opt = torch.optim.AdamW(list(legacy.parameters()), lr=1e-2)  # all parameters, like VeRL
    legacy_opt.zero_grad()
    out_proj = legacy.base_model.model.linear_attn.out_proj
    (((legacy(x) - target) ** 2).sum() + out_proj(torch.randn(5, 6)).sum()).backward()
    legacy_opt.step()
    torch.save(legacy_opt.state_dict(), tmp_path / "optim_world_size_1_rank_0.pt")
    full = {n: p.detach().clone() for n, p in legacy.named_parameters() if "lora_" in n}

    split = copy.deepcopy(base)
    mark_split_qkv(split)
    split = peft.get_peft_model(split, lora_config_class()(r=2, lora_alpha=32, target_modules=targets))
    split_opt = torch.optim.AdamW(list(split.parameters()), lr=1e-2)
    manager = SimpleNamespace(world_size=1, rank=0, model=split, optimizer=split_opt)
    _load_legacy_optimizer(manager, str(tmp_path), full, legacy_split_names(full), lambda key: (4, 4, 6))

    by_name = {n: p for n, p in split.named_parameters()}
    fused_state = legacy_opt.state[
        dict(legacy.named_parameters())["base_model.model.linear_attn.in_proj_qkv.lora_B.default.weight"]
    ]
    q_state = split_opt.state[by_name["base_model.model.linear_attn.in_proj_qkv.lora_B.default.q.weight"]]
    torch.testing.assert_close(q_state["exp_avg"], fused_state["exp_avg"][:4])
    out_state = split_opt.state[by_name["base_model.model.linear_attn.out_proj.lora_A.default.weight"]]
    torch.testing.assert_close(
        out_state["exp_avg_sq"],
        legacy_opt.state[
            dict(legacy.named_parameters())["base_model.model.linear_attn.out_proj.lora_A.default.weight"]
        ]["exp_avg_sq"],
    )
    assert not split_opt.state[by_name["base_model.model.linear_attn.out_proj.base_layer.weight"]]
