"""The PQ2_0 -> MLX pack conversion: MLX's 2-bit packing, the converter's value-head reorder
undone, and a tiny q2 file converted into a template's layout without loss."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from common.gguf.mlx import (
    INDEX,
    WEIGHTS,
    Geometry,
    convert,
    header,
    name_in_pack,
    pack_2bit,
    unpack_2bit,
    verify,
)
from common.gguf.reader import ARRAY, STRING, U32, read_header
from common.gguf.reader import F32 as KV_F32
from common.gguf.types import BF16, F32, PQ2_0
from tests.test_gguf import _kv, _str, write_gguf
from tests.test_gguf_ptq1 import pq2_blocks

HIDDEN, FFN, NK, NV, HK, HV, CONV = 128, 256, 2, 4, 32, 32, 4  # a two-layer qwen35 in miniature
QK, V = 2 * NK * HK, NV * HV


def test_packing_is_mlx_affine_2bit() -> None:
    codes = np.random.default_rng(0).integers(0, 4, (5, 128)).astype(np.uint8)
    words = pack_2bit(codes)
    reference = [
        [sum(int(c) << (2 * k) for k, c in enumerate(row[w * 16 : w * 16 + 16])) for w in range(8)]
        for row in codes
    ]
    assert words.dtype == np.uint32 and words.tolist() == reference
    assert (unpack_2bit(words) == codes).all()


def tiled(x: np.ndarray, width: int) -> np.ndarray:
    """convert_hf_to_gguf's _reorder_v_heads: rows (key head, its value heads) -> (value head
    index, key head)."""
    s = x.shape
    return x.reshape(NK, NV // NK, width, *s[1:]).swapaxes(0, 1).reshape(s)


def test_rows_undo_the_converters_tiling() -> None:
    geo = Geometry(NK, NV, HK, HV, CONV)
    rng = np.random.default_rng(1)
    for name, n, width, top in (
        ("blk.0.attn_gate.weight", V, HV, 0),
        ("blk.0.ssm_a", NV, 1, 0),
        ("blk.0.ssm_alpha.weight", NV, 1, 0),
        ("blk.0.attn_qkv.weight", QK + V, HV, QK),
        ("blk.0.ssm_conv1d.weight", QK + V, HV, QK),
    ):
        grouped = rng.standard_normal((n, 3))
        stored = np.concatenate([grouped[:top], tiled(grouped[top:], width)])
        assert (stored[geo.rows(name, n)] == grouped).all(), name
    assert (geo.rows("blk.0.ffn_up.weight", 7) == np.arange(7)).all()
    assert (geo.rows("output.weight", 7) == np.arange(7)).all()


def _array_kv(key: str, etype: int, fmt: str, values: list) -> bytes:
    out = _str(key) + struct.pack("<IIQ", ARRAY, etype, len(values))
    return out + b"".join(struct.pack(fmt, v) for v in values)


def q2_model(path: Path, rng: np.random.Generator, *, bad_code: bool = False) -> Path:
    """A q2 gguf with one linear-attention layer and one attention layer, every tensor kind the
    converter maps, PQ2_0 weights of random ternary codes, MOSTLY_PQ2_0."""
    ternary = {
        "token_embd.weight": (HIDDEN, 6),
        "output.weight": (HIDDEN, 6),
        "blk.0.attn_qkv.weight": (HIDDEN, QK + V),
        "blk.0.attn_gate.weight": (HIDDEN, V),
        "blk.0.ssm_out.weight": (V, HIDDEN),
        "blk.0.ffn_gate.weight": (HIDDEN, FFN),
        "blk.0.ffn_up.weight": (HIDDEN, FFN),
        "blk.0.ffn_down.weight": (FFN, HIDDEN),
        "blk.1.attn_q.weight": (HIDDEN, HIDDEN),
        "blk.1.attn_k.weight": (HIDDEN, HIDDEN),
        "blk.1.attn_v.weight": (HIDDEN, HIDDEN),
        "blk.1.attn_output.weight": (HIDDEN, HIDDEN),
        "blk.1.ffn_gate.weight": (HIDDEN, FFN),
        "blk.1.ffn_up.weight": (HIDDEN, FFN),
        "blk.1.ffn_down.weight": (FFN, HIDDEN),
    }
    wide = {
        "output_norm.weight": (F32, (HIDDEN,)),
        "blk.0.attn_norm.weight": (F32, (HIDDEN,)),
        "blk.0.post_attention_norm.weight": (F32, (HIDDEN,)),
        "blk.0.ssm_norm.weight": (F32, (HV,)),
        "blk.0.ssm_a": (F32, (NV,)),
        "blk.0.ssm_dt.bias": (F32, (NV,)),
        "blk.0.ssm_alpha.weight": (BF16, (HIDDEN, NV)),
        "blk.0.ssm_beta.weight": (BF16, (HIDDEN, NV)),
        "blk.0.ssm_conv1d.weight": (F32, (CONV, QK + V)),
        "blk.1.attn_norm.weight": (F32, (HIDDEN,)),
        "blk.1.post_attention_norm.weight": (F32, (HIDDEN,)),
        "blk.1.attn_q_norm.weight": (F32, (HK,)),
        "blk.1.attn_k_norm.weight": (F32, (HK,)),
    }
    tensors, data = [], bytearray()
    for name, (cols, rows) in ternary.items():
        codes = rng.integers(0, 3, (rows * cols // 128, 128)).astype(np.uint8)
        if bad_code and name == "blk.0.ffn_up.weight":
            codes[2, 7] = 3
        scale = rng.integers(0, 60, (len(codes), 2)).astype(np.uint8)
        tensors.append((name, PQ2_0, (cols, rows), len(data)))
        data += pq2_blocks(codes, scale).tobytes()
        data += bytes(-len(data) % 32)
    for name, (ggml_type, shape) in wide.items():
        values = rng.standard_normal(int(np.prod(shape))).astype(np.float32)
        if name.endswith("ssm_a"):
            values = -np.exp(values)
        tensors.append((name, ggml_type, shape, len(data)))
        data += (
            values.tobytes()
            if ggml_type == F32
            else (values.view("<u4") >> 16).astype("<u2").tobytes()
        )
        data += bytes(-len(data) % 32)
    widths = sorted({cols for cols, _ in ternary.values()})
    signs = [float(s) for s in rng.choice([-1.0, 1.0], sum(widths))]
    kv = [
        _kv("general.alignment", U32, 32),
        _kv("general.file_type", U32, 141),
        _kv("qwen35.ssm.group_count", U32, NK),
        _kv("qwen35.ssm.time_step_rank", U32, NV),
        _kv("qwen35.ssm.state_size", U32, HK),
        _kv("qwen35.ssm.inner_size", U32, V),
        _kv("qwen35.ssm.conv_kernel", U32, CONV),
        _kv("prism.hadamard.block_size", U32, 128),
        _kv("prism.hadamard.sign_mode", STRING, "explicit"),
        _array_kv("prism.hadamard.sign_widths", U32, "<I", widths),
        _array_kv("prism.hadamard.sign_values", KV_F32, "<f", signs),
        _kv("prism.hadamard.weight_names", ARRAY, [n for n in ternary if n != "token_embd.weight"]),
        _kv("prism.hadamard.inverse_weight_names", ARRAY, ["token_embd.weight"]),
    ]
    return write_gguf(path, kv=kv, tensors=tensors, data=bytes(data))


def template(path: Path, g_path: Path, rng: np.random.Generator, *, vision: bool = True) -> Path:
    """The pack layout a q2_model converts into: a header of every expected tensor (zeros), a
    vision tower tensor of random values, and the json files a template brings."""
    g = read_header(g_path)
    entries: dict[str, tuple[str, list[int]]] = {}
    for t in g.tensors.values():
        name = name_in_pack(t.name)
        cols, rows = t.shape[0], int(np.prod(t.shape[1:]))
        if t.ggml_type == PQ2_0:
            base = name.removesuffix(".weight")
            entries[base + ".weight"] = ("U32", [rows, cols // 16])
            entries[base + ".scales"] = ("F16", [rows, cols // 128])
            entries[base + ".biases"] = ("F16", [rows, cols // 128])
            entries[base + ".signs"] = ("F32", [cols])
        elif t.name.endswith("ssm_conv1d.weight"):
            entries[name] = ("F32", [rows, cols, 1])
        else:
            entries[name] = ("F32", list(reversed(t.shape)))
    if vision:
        entries["vision_tower.patch_embed.weight"] = ("F16", [3, 16])
    sizes = {"U32": 4, "F16": 2, "F32": 4}
    head, at, body = {}, 0, bytearray()
    for name, (dtype, shape) in entries.items():
        n = int(np.prod(shape)) * sizes[dtype]
        head[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
        at += n
        body += rng.bytes(n) if name.startswith("vision_tower.") else bytes(n)
    text = json.dumps(head).encode()
    path.mkdir()
    (path / WEIGHTS).write_bytes(struct.pack("<Q", len(text)) + text + body)
    (path / "config.json").write_text('{"model_type": "prism_hadamard_qwen35"}')
    (path / "hadamard.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    return path


def test_a_q2_file_converts_without_loss(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    src = q2_model(tmp_path / "q2.gguf", rng)
    tpl = template(tmp_path / "template", src, rng)
    out = tmp_path / "pack"
    weights = convert(src, tpl, out)
    assert weights == sum(
        int(np.prod(t.shape)) for t in read_header(src).tensors.values() if t.ggml_type == PQ2_0
    )
    verify(out, src, tpl)

    got, start = header(out / WEIGHTS)
    tpl_head, _ = header(tpl / WEIGHTS)
    assert list(got) == list(tpl_head)
    offsets = [info["data_offsets"] for info in got.values()]
    assert offsets[0][0] == 0 and all(
        a[1] == b[0] for a, b in zip(offsets, offsets[1:], strict=False)
    )
    assert (out / WEIGHTS).stat().st_size == start + offsets[-1][1]
    index = json.loads((out / INDEX).read_text())
    assert index["weight_map"] == {n: WEIGHTS for n in got}
    assert (out / "config.json").read_text() == (tpl / "config.json").read_text()
    assert (out / "hadamard.json").exists() and (out / "tokenizer.json").exists()


def _flip(path: Path, info: dict, start: int, at: int, byte: int) -> None:
    with path.open("r+b") as fh:
        fh.seek(start + info["data_offsets"][0] + at)
        fh.write(bytes([byte]))


def test_verify_sees_every_change(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    src = q2_model(tmp_path / "q2.gguf", rng)
    tpl = template(tmp_path / "template", src, rng)
    out = tmp_path / "pack"
    convert(src, tpl, out)
    got, start = header(out / WEIGHTS)
    weight = "language_model.model.layers.0.mlp.up_proj.weight"

    original = (out / WEIGHTS).read_bytes()
    _flip(out / WEIGHTS, got[weight], start, 5, 0xFF)  # a code
    with pytest.raises(ValueError, match="up_proj.weight: a block does not dequantize"):
        verify(out, src, tpl)
    (out / WEIGHTS).write_bytes(original)
    _flip(out / WEIGHTS, got[weight.replace(".weight", ".scales")], start, 1, 0x7B)  # a scale
    with pytest.raises(ValueError, match="up_proj.weight: a block does not dequantize"):
        verify(out, src, tpl)
    (out / WEIGHTS).write_bytes(original)
    _flip(out / WEIGHTS, got["language_model.model.layers.0.linear_attn.A_log"], start, 2, 0x00)
    with pytest.raises(ValueError, match="A_log"):
        verify(out, src, tpl)
    (out / WEIGHTS).write_bytes(original)

    without = template(tmp_path / "template2", src, rng, vision=False)
    with pytest.raises(ValueError, match="vision_tower.patch_embed.weight: not the template"):
        verify(out, src, without)


def test_a_fourth_level_is_refused(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    src = q2_model(tmp_path / "q2.gguf", rng, bad_code=True)
    with pytest.raises(ValueError, match="ffn_up.weight: code 3"):
        convert(src, template(tmp_path / "template", src, rng), tmp_path / "pack")
