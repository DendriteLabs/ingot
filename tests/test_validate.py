"""Validation rules: genesis config values outside the free keys, the genesis tensor layout with
a free fp8/bf16/f32 split in the text model, safetensors no larger than genesis, files
allowlisted. Table-driven on a toy family plus the real pinned one."""

from __future__ import annotations

import json
import struct
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from common.config import REPO_ROOT, _king_names, _upload_limits, chain_spec
from model_validation import families
from model_validation.checks import (
    check_architecture,
    check_config_keys,
    check_files,
    check_index,
    check_json,
    check_layout,
    check_quant,
    check_values,
    check_weight_bytes,
)
from model_validation.families.base import Family, Layout

QUANT = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": ["lm_head", "model.visual", "model.norm"],
}
UP = "model.layers.0.mlp.up_proj.weight"
SCALE = UP + "_scale_inv"
GENESIS: Layout = {
    "model.embed_tokens.weight": ("BF16", (1000, 64)),
    UP: ("F8_E4M3", (256, 64)),
    SCALE: ("BF16", (2, 1)),
    "model.norm.weight": ("BF16", (64,)),
    "model.visual.blocks.0.attn.qkv.weight": ("BF16", (64, 64)),
    "lm_head.weight": ("BF16", (1000, 64)),
}
FAMILY = Family(
    name="toy",
    genesis_config={
        "architectures": ["Qwen"],
        "transformers_version": "5.0",
        "text_config": {"num_hidden_layers": 64, "rms_norm_eps": 1e-6, "use_cache": True},
        "quantization_config": QUANT,
    },
    arch_keys=(
        "architectures",
        "text_config.num_hidden_layers",
        "text_config.rms_norm_eps",
        "quantization_config.quant_method",
        "quantization_config.fmt",
    ),
    genesis_layout=GENESIS,
    frozen_dtype_prefixes=("model.visual.",),
    genesis_weight_bytes=families.weight_bytes(GENESIS, ("model.visual.",)),
    embeddings=("model.embed_tokens", "lm_head"),
    uncounted_prefixes=("model.visual.",),
)
CONFIG = FAMILY.genesis_config


def with_quant(**changes: object) -> dict:
    return {**CONFIG, "quantization_config": {**QUANT, **changes}}


def changed(**tensors: tuple[str, tuple[int, ...]] | None) -> Layout:
    """Genesis with tensors replaced (`lm_head__weight=...`) or removed (`=None`)."""
    layout = dict(GENESIS)
    for key, value in tensors.items():
        name = key.replace("__", ".")
        if value is None:
            layout.pop(name)
        else:
            layout[name] = value
    return layout


def layout_reason(layout: Layout, config: dict = CONFIG) -> str:
    result = check_layout(layout, FAMILY, config)
    assert result is not None, "expected a rejection"
    return result


def test_architecture_keys_are_looked_up_nested():
    assert check_architecture(CONFIG, FAMILY) is None
    flat = {"architectures": ["Qwen"], "num_hidden_layers": 64}
    assert "text_config.num_hidden_layers" in str(check_architecture(flat, FAMILY))
    assert "architectures" in str(check_architecture({**CONFIG, "architectures": ["X"]}, FAMILY))


def test_free_keys_may_change_and_frozen_ones_may_not():
    free = {**CONFIG, "transformers_version": "6.0"}
    free["text_config"] = {**CONFIG["text_config"], "use_cache": False}
    assert check_architecture(with_quant(weight_block_size=[256, 256]), FAMILY) is None
    assert check_architecture(free, FAMILY) is None
    eps = {**CONFIG, "text_config": {**CONFIG["text_config"], "rms_norm_eps": 1e-5}}
    assert "rms_norm_eps" in str(check_architecture(eps, FAMILY))
    assert "fmt" in str(check_architecture(with_quant(fmt="e5m2"), FAMILY))


def test_the_genesis_layout_itself_passes():
    assert check_layout(GENESIS, FAMILY, CONFIG) is None


def test_text_weights_may_move_between_fp8_bf16_and_f32_if_the_config_agrees():
    bf16_up = changed(
        **{UP.replace(".", "__"): ("BF16", (256, 64))}, **{SCALE.replace(".", "__"): None}
    )
    assert "missing from modules_to_not_convert" in layout_reason(bf16_up)
    listed = with_quant(
        modules_to_not_convert=[*QUANT["modules_to_not_convert"], "model.layers.0.mlp.up_proj"]
    )
    assert check_layout(bf16_up, FAMILY, listed) is None
    f32_up = {**bf16_up, UP: ("F32", (256, 64))}
    assert check_layout(f32_up, FAMILY, listed) is None


def test_only_weights_genesis_quantizes_may_be_fp8():
    fp8_embed = changed(
        model__embed_tokens__weight=("F8_E4M3", (1000, 64)),
        model__embed_tokens__weight_scale_inv=("BF16", (8, 1)),
    )
    unlisted = with_quant(modules_to_not_convert=["lm_head", "model.visual", "model.norm"])
    assert layout_reason(fp8_embed, unlisted) == (
        "model.embed_tokens.weight: FP8, but genesis keeps it BF16; vLLM cannot load it quantized"
    )
    fp8_head = changed(
        lm_head__weight=("F8_E4M3", (1000, 64)), lm_head__weight_scale_inv=("F32", (8, 1))
    )
    assert "genesis keeps it BF16" in layout_reason(fp8_head)


def test_projections_vllm_fuses_share_one_precision():
    gate = "model.layers.0.mlp.gate_proj.weight"
    genesis = {**GENESIS, gate: ("F8_E4M3", (256, 64)), gate + "_scale_inv": ("BF16", (2, 1))}
    family = replace(FAMILY, genesis_layout=genesis, fused=(("gate_proj", "up_proj"),))
    assert check_layout(genesis, family, CONFIG) is None
    split = {k: v for k, v in genesis.items() if k != gate + "_scale_inv"}
    split[gate] = ("BF16", (256, 64))
    listed = with_quant(
        modules_to_not_convert=[*QUANT["modules_to_not_convert"], "model.layers.0.mlp.gate_proj"]
    )
    assert check_layout(split, family, listed) == (
        "model.layers.0.mlp: gate_proj not, up_proj FP8; vLLM loads them as one layer,"
        " so all FP8 or none"
    )


def test_frozen_parts_keep_the_genesis_dtype():
    vision = changed(**{"model__visual__blocks__0__attn__qkv__weight": ("F8_E4M3", (64, 64))})
    vision["model.visual.blocks.0.attn.qkv.weight_scale_inv"] = ("BF16", (1, 1))
    assert "frozen part" in layout_reason(vision)
    assert "frozen part" in layout_reason(
        changed(**{"model__visual__blocks__0__attn__qkv__weight": ("F32", (64, 64))})
    )


def test_scales_follow_the_declared_block_and_fp8_needs_a_matrix():
    assert "does not match block" in layout_reason(
        GENESIS, with_quant(weight_block_size=[256, 256])
    )
    assert (
        check_layout(
            {**GENESIS, SCALE: ("BF16", (1, 1))}, FAMILY, with_quant(weight_block_size=[256, 256])
        )
        is None
    )
    assert "need two ints >= 128" in layout_reason(GENESIS, with_quant(weight_block_size=[64, 64]))
    assert "without" in layout_reason(changed(**{SCALE.replace(".", "__"): None}))
    assert "scale on a non-FP8" in layout_reason(
        changed(lm_head__weight_scale_inv=("BF16", (8, 1)))
    )
    assert "without a weight" in layout_reason(
        changed(model__layers__9__weight_scale_inv=("BF16", (1, 1)))
    )
    assert "not allowed" in layout_reason(changed(model__norm__weight=("F8_E4M3", (64,))))
    assert "not allowed" in layout_reason(changed(**{UP.replace(".", "__"): ("I8", (256, 64))}))


def test_tensor_set_and_shapes_must_match_genesis():
    assert "shape" in layout_reason(changed(lm_head__weight=("BF16", (999, 64))))
    assert "missing" in layout_reason(changed(lm_head__weight=None))
    assert "unexpected" in layout_reason(
        changed(model__layers__0__self_attn__q_proj__bias=("BF16", (64,)))
    )


def test_config_must_say_the_declared_quant():
    assert check_quant(CONFIG, "fp8") is None
    assert check_quant(CONFIG, "int4") == (
        "config.json quantization_config.quant_method is 'fp8', declared 'int4'"
    )
    assert "is None" in str(check_quant({**CONFIG, "quantization_config": None}, "fp8"))


def test_config_may_only_hold_genesis_keys():
    assert check_config_keys(CONFIG, CONFIG) is None
    assert check_config_keys({**CONFIG, "quantization_config": None}, CONFIG) is None
    for bad, path in (
        ({**CONFIG, "auto_map": {"AutoModel": "evil/repo--modeling.M"}}, "auto_map"),
        ({**CONFIG, "text_config": {"num_hidden_layers": 64, "x": 1}}, "text_config.x"),
        (
            {**CONFIG, "quantization_config": {"quant_method": "fp8", "url": "http://x"}},
            "quantization_config.url",
        ),
    ):
        assert check_config_keys(bad, CONFIG) == f"config.json has a key genesis does not: {path}"
    assert "must be an object" in str(check_config_keys({**CONFIG, "text_config": "x"}, CONFIG))


def test_json_files_are_small_and_the_index_plain(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": {"x": "a"}}))
    assert check_json(tmp_path) is None
    index.write_text(json.dumps({"weight_map": {}, "auto_map": {}}))
    assert "may only hold" in str(check_json(tmp_path))
    index.write_text(json.dumps({"metadata": {"source": "http://x"}, "weight_map": {}}))
    assert "unexpected keys: source" in str(check_json(tmp_path))
    index.write_text(json.dumps({"weight_map": {"x": {"url": "http://x"}}}))
    assert "shard names" in str(check_json(tmp_path))
    (tmp_path / "config.json").write_bytes(b" " * 1_000_001)
    assert "at most" in str(check_json(tmp_path))


def test_bucket_names_allowed_during_upload():
    spec = chain_spec()
    for ok in ("config.json", "model.safetensors.index.json", "a.safetensors"):
        assert spec.allows_file(ok)
    for bad in ("modeling.py", "tokenizer.json", "x/a.safetensors", "README.md", "manifest.json"):
        assert not spec.allows_file(bad)


def shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """A safetensors file from (dtype, shape, raw bytes) per tensor."""
    header, data, at = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        data, at = data + raw, at + len(raw)
    head = json.dumps(header).encode()
    path.write_bytes(len(head).to_bytes(8, "little") + head + data)


def test_every_weight_must_be_finite(tmp_path: Path):
    one_bf16, nan_bf16 = (0x3F80).to_bytes(2, "little"), (0x7FC0).to_bytes(2, "little")
    shard(
        tmp_path / "a.safetensors",
        {"w": ("BF16", [2, 2], one_bf16 * 4), "s": ("F8_E4M3", [1, 2], b"\x38\x40")},
    )
    assert check_values(tmp_path) is None
    shard(tmp_path / "b.safetensors", {"x": ("BF16", [1, 4], one_bf16 * 2 + nan_bf16 * 2)})
    assert check_values(tmp_path) == "x: 2 of its values are NaN or infinite"
    shard(tmp_path / "b.safetensors", {"y": ("F8_E4M3", [1, 3], b"\x38\x7f\xff")})
    assert check_values(tmp_path) == "y: 2 of its values are NaN or infinite"
    shard(tmp_path / "b.safetensors", {"z": ("F32", [1, 2], struct.pack("<ff", 1.0, float("inf")))})
    assert check_values(tmp_path) == "z: 1 of its values are NaN or infinite"


def test_text_weights_may_not_outweigh_genesis_text_weights():
    assert check_weight_bytes(GENESIS, FAMILY) is None
    assert check_weight_bytes(changed(lm_head__weight=("BF16", (1001, 64))), FAMILY) == (
        "counted weights total 272644 bytes, genesis has 272516"
    )
    # the vision tower is shipped as genesis and counted on neither side
    bigger_tower = changed(model__visual__blocks__0__attn__qkv__weight=("BF16", (640, 64)))
    assert check_weight_bytes(bigger_tower, FAMILY) is None


INDEX = "model.safetensors.index.json"
SUBMISSION = ["config.json", INDEX, "layers-0.safetensors", "layers-1.safetensors"]


def test_a_submission_is_config_index_and_shards_only():
    spec = chain_spec()
    assert check_files(SUBMISSION, spec) is None
    for extra in (
        "tokenizer.json",
        "README.md",
        "crc32.txt",
        "modeling.py",
        ".gitattributes",
        "manifest.json",
    ):
        assert f"unexpected files: {extra}" == check_files([*SUBMISSION, extra], spec)
    assert "subfolders" in str(check_files([*SUBMISSION, "draft/model.safetensors"], spec))


def test_missing_pieces_are_named():
    spec = chain_spec()
    assert check_files(SUBMISSION[1:], spec) == "missing required files: config.json"
    assert check_files([f for f in SUBMISSION if f != INDEX], spec) == (
        f"missing required files: {INDEX}"
    )
    assert check_files(["config.json", INDEX], spec) == "no safetensors weights"


def test_index_must_name_exactly_the_shards(tmp_path: Path):
    for shard in ("a.safetensors", "b.safetensors"):
        (tmp_path / shard).write_bytes(b"")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}))
    assert check_index(tmp_path) is None
    index.write_text(json.dumps({"weight_map": {"x": "a.safetensors"}}))
    assert "not in" in str(check_index(tmp_path))
    index.write_text(json.dumps({"weight_map": {"x": "a.safetensors", "z": "c.safetensors"}}))
    assert "missing" in str(check_index(tmp_path))


def test_pinned_qwen38_family_accepts_its_own_genesis():
    family = families.load(chain_spec().family)
    assert check_architecture(family.genesis_config, family) is None
    assert len(family.genesis_layout) == 1606
    assert check_layout(family.genesis_layout, family, family.genesis_config) is None
    assert family.genesis_weight_bytes == 29_468_003_328
    for frozen in (
        "language_model_only",
        "text_config.rms_norm_eps",
        "text_config.dtype",
        "text_config.rope_parameters.rope_theta",
        "text_config.attention_bias",
        "vision_config.patch_size",
        "quantization_config.fmt",
        "quantization_config.activation_scheme",
    ):
        assert frozen in family.arch_keys
    assert "quantization_config.weight_block_size" not in family.arch_keys


def test_upload_limit_is_per_declared_family_and_quantization():
    spec = chain_spec()
    assert spec.upload_limit("qwen38_27b", "fp8") == 35_000_000_000
    with pytest.raises(ValueError, match="qwen38_27b.gptq is not an allowed pair"):
        spec.upload_limit("qwen38_27b", "gptq")
    raw = tomllib.loads((REPO_ROOT / "chain.toml").read_text())
    raw["genesis"]["quant"] = "gptq"
    with pytest.raises(ValueError, match="no entry for genesis qwen38_27b.gptq"):
        _upload_limits(raw)


def test_king_names_follow_the_declared_pairs(spec):
    assert spec.king_name("qwen38_27b", "fp8") == "27B-FP8"
    raw = tomllib.loads((REPO_ROOT / "chain.toml").read_text())
    raw["upload_limits"]["qwen38_27b"]["fp4"] = 20_000_000_000
    with pytest.raises(ValueError, match="same pairs"):
        _king_names(raw)


def test_activation_payload_carries_the_pair_and_fits_a_commitment():
    from private_store.contracts import activation_signal_payload, parse_activation

    payload = activation_signal_payload(bytes(range(32)), family="qwen38_27b", quant="fp8")
    assert payload.startswith("r2activate:v2:qwen38_27b:fp8:") and len(payload) <= 128
    parsed = parse_activation(payload)
    assert (parsed.family, parsed.quant, parsed.pubkey) == ("qwen38_27b", "fp8", bytes(range(32)))
    with pytest.raises(ValueError):
        activation_signal_payload(bytes(32), family="Qwen:3.8", quant="fp8")


def test_every_unquantized_linear_must_be_listed_embeddings_and_norms_need_not():
    # vLLM reads the list for linear layers only: an unlisted norm, embedding or lm_head loads
    assert (
        check_layout(GENESIS, FAMILY, with_quant(modules_to_not_convert=["model.visual"])) is None
    )
    # the vision tower is a linear layer too (served whole on the benchmark pod and from HF)
    dropped = with_quant(modules_to_not_convert=["lm_head", "model.norm"])
    assert layout_reason(GENESIS, dropped) == (
        "not FP8 but missing from modules_to_not_convert: model.visual.blocks.N.attn.qkv"
    )
    # every missing module at once, one name per layer stack, so one fix covers them
    bf16_up = changed(
        **{UP.replace(".", "__"): ("BF16", (256, 64))}, **{SCALE.replace(".", "__"): None}
    )
    assert layout_reason(bf16_up, dropped) == (
        "not FP8 but missing from modules_to_not_convert: "
        "model.layers.N.mlp.up_proj, model.visual.blocks.N.attn.qkv"
    )
