"""A Prism-layout MLX 2-bit pack from a ternary PQ2_0 gguf, without loss. Each 128-weight block's
codes become MLX affine 2-bit (code = trit + 1, 16 a uint32, the first in the low bits) with
scales = d and biases = -d, so the dequantized weight is trit * d as the gguf kernel computes it;
the Hadamard sign vectors come from prism.hadamard.sign_values. Rows the gguf converter tiled
(convert_hf_to_gguf's value-head reorder) are grouped again, and A_log = log(-ssm_a). Everything
the gguf lacks (config, tokenizer, hadamard.json, the vision tower) is the template's: the
genesis's MLX twin."""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.gguf.dequant import pq2_codes, rotation, tensor
from common.gguf.ptq1 import is_pq2
from common.gguf.reader import Gguf, TensorInfo, read_header
from common.gguf.types import PQ2_0

# genesis's MLX twin: it supplies the config, tokenizer, hadamard.json and vision tower a gguf
# lacks; the language-model weights are the king's
TEMPLATE_REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
TEMPLATE_REVISION = "fcba37d2117a7077eac6b613b2668d14d9779edd"
PREFIX = "language_model."
WEIGHTS = "model.safetensors"
INDEX = "model.safetensors.index.json"
TEMPLATE_FILES = (
    "config.json",
    "hadamard.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
DTYPES = {
    "U32": np.dtype("<u4"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "BF16": np.dtype("<u2"),
}
GROUP = 128
CHUNK = 1 << 19  # blocks per numpy pass
ROWS = 1 << 12  # rows per verification pass

_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
}
_TOP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}
# gguf tensors whose rows the converter tiled: (rows left alone at the top, head width)
_TILED = {
    "attn_qkv.weight": ("qk", "hv"),
    "attn_gate.weight": (0, "hv"),
    "ssm_conv1d.weight": ("qk", "hv"),
    "ssm_a": (0, 1),
    "ssm_dt.bias": (0, 1),
    "ssm_alpha.weight": (0, 1),
    "ssm_beta.weight": (0, 1),
}


_SSM_KEYS = ("group_count", "time_step_rank", "state_size", "inner_size", "conv_kernel")


@dataclass(frozen=True)
class Geometry:
    """The linear-attention head counts the row reorders need (qwen35.ssm.*)."""

    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv: int

    @classmethod
    def read(cls, g: Gguf) -> Geometry:
        kv = {k: int(g.kv["qwen35.ssm." + k].value) for k in _SSM_KEYS}
        heads = kv["time_step_rank"]
        return cls(
            kv["group_count"], heads, kv["state_size"], kv["inner_size"] // heads, kv["conv_kernel"]
        )

    def rows(self, name: str, n: int) -> np.ndarray:
        """Source row of each output row: the converter's grouped -> tiled value-head reorder
        undone over the tiled rows, identity for every other tensor."""
        order = np.arange(n)
        tiled = _TILED.get(name.split(".", 2)[2]) if name.startswith("blk.") else None
        if tiled is None:
            return order
        top = 2 * self.key_heads * self.key_dim if tiled[0] == "qk" else 0
        width = self.value_dim if tiled[1] == "hv" else 1
        per = self.value_heads // self.key_heads
        rest = order[top:].reshape(per, self.key_heads, width).transpose(1, 0, 2).reshape(-1)
        return np.concatenate([order[:top], rest])


def name_in_pack(name: str) -> str:
    """A gguf tensor's name in the pack (blk.5.ffn_down.weight ->
    language_model.model.layers.5.mlp.down_proj.weight)."""
    if name in _TOP:
        return PREFIX + _TOP[name]
    if name.startswith("blk."):
        _, layer, rest = name.split(".", 2)
        if rest in _LAYER:
            return f"{PREFIX}model.layers.{layer}.{_LAYER[rest]}"
    raise ValueError(f"{name}: no place in an MLX pack")


def header(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Tensor name -> {dtype, shape, data_offsets} of a safetensors file, and where its data
    starts; the entries keep the file's order."""
    with path.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        head = json.loads(fh.read(size))
    head.pop("__metadata__", None)
    return head, 8 + size


def convert(src: Path, template: Path, out: Path) -> int:
    """Writes the pack at out: model.safetensors with every template tensor in the template's
    order (the vision tower copied, the rest converted from src), its index, and the template's
    config, tokenizer and rotation files. Returns the ternary weights converted. Raises
    ValueError on a file that is not PQ2_0, a code 3, a tensor the mapping does not know, or a
    tensor whose shape or dtype is not the template's."""
    g = read_header(src)
    if not is_pq2(g):
        raise ValueError(f"{src}: general.file_type is not MOSTLY_PQ2_0")
    geo = Geometry.read(g)
    tpl, tpl_start = header(template / WEIGHTS)
    produced = _producers(g)
    missing = sorted(set(produced) - set(tpl))
    if missing:
        raise ValueError(f"{missing[0]}: the gguf has it, the template does not")

    out.mkdir(parents=True, exist_ok=True)
    head: dict[str, dict[str, Any]] = {"__metadata__": {"format": "mlx"}}
    at = 0
    for name, info in tpl.items():
        size = int(np.prod(info["shape"])) * _dtype(name, info["dtype"]).itemsize
        head[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [at, at + size],
        }
        at += size
    text = json.dumps(head, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)

    weights = 0
    with open(out / WEIGHTS, "wb") as fout, open(template / WEIGHTS, "rb") as ftpl:
        fout.write(struct.pack("<Q", len(text)) + text)
        for name, info in tpl.items():
            if name not in produced:
                start, end = info["data_offsets"]
                ftpl.seek(tpl_start + start)
                for pos in range(start, end, CHUNK * 34):
                    fout.write(ftpl.read(min(CHUNK * 34, end - pos)))
                continue
            array = _produce(src, g, geo, produced[name], name)
            if list(array.shape) != info["shape"] or array.dtype != _dtype(name, info["dtype"]):
                raise ValueError(
                    f"{name}: converted to {array.dtype} {list(array.shape)}, the template has "
                    f"{info['dtype']} {info['shape']}"
                )
            fout.write(array.tobytes())
            if name.endswith(".weight") and produced[name].ggml_type == PQ2_0:
                weights += array.size * 16
    index = {
        "metadata": {"total_size": at, "format": "mlx"},
        "weight_map": {n: WEIGHTS for n in sorted(tpl)},
    }
    (out / INDEX).write_text(json.dumps(index, indent=2))
    for file in TEMPLATE_FILES:
        if (template / file).is_file():
            shutil.copyfile(template / file, out / file)
    return weights


def verify(out: Path, src: Path, template: Path) -> None:
    """Raises ValueError, naming the tensor, unless the pack at out is src: the same tensor set,
    shapes and dtypes as the template; every ternary tensor dequantizing (code * scales + biases)
    to (code - 1) * d of the gguf exactly, with the gguf's sign vector; -exp(A_log) equal to
    ssm_a within float rounding; every other converted tensor exact; the vision tower the
    template's bytes; the index listing every tensor."""
    g = read_header(src)
    geo = Geometry.read(g)
    tpl, tpl_start = header(template / WEIGHTS)
    got, got_start = header(out / WEIGHTS)
    if list(got) != list(tpl):
        extra = sorted(set(got) ^ set(tpl))
        raise ValueError(f"{extra[0] if extra else next(iter(got))}: not the template's tensors")
    for name, info in tpl.items():
        if got[name]["shape"] != info["shape"] or got[name]["dtype"] != info["dtype"]:
            raise ValueError(f"{name}: shape or dtype differs from the template")
    index = json.loads((out / INDEX).read_text())
    if index.get("weight_map") != {n: WEIGHTS for n in tpl}:
        raise ValueError(f"{INDEX}: it does not list every tensor")

    produced = _producers(g)
    for name, info in got.items():
        if name not in produced:
            if not _same_bytes(
                out / WEIGHTS, got_start, info, template / WEIGHTS, tpl_start, tpl[name]
            ):
                raise ValueError(f"{name}: differs from the template")
            continue
        t = produced[name]
        if t.ggml_type != PQ2_0:
            array = _produce(src, g, geo, t, name)
            packed = _array(out / WEIGHTS, got_start, info, name)
            if name.endswith(".A_log"):
                back = -np.exp(packed.astype(np.float32))
                a = tensor(src, g, t.name)[geo.rows(t.name, t.shape[-1])]
                if not (np.abs(back - a) <= 1e-6 * np.abs(a)).all():
                    raise ValueError(f"{name}: -exp(A_log) is not ssm_a")
            elif packed.tobytes() != array.tobytes():
                raise ValueError(f"{name}: differs from the gguf")
        elif name.endswith(".signs"):
            packed = _array(out / WEIGHTS, got_start, info, name)
            if not np.array_equal(packed, _signs(g, t)):
                raise ValueError(f"{name}: the sign vector is not the gguf's")
        elif name.endswith(".weight"):
            _check_ternary(out, got, got_start, src, g, geo, t, name)


def _producers(g: Gguf) -> dict[str, TensorInfo]:
    """Pack tensor name -> the gguf tensor it comes from; a ternary tensor makes four."""
    produced = {}
    for t in g.tensors.values():
        name = name_in_pack(t.name)
        if t.ggml_type == PQ2_0:
            base = name.removesuffix(".weight")
            for part in (".weight", ".scales", ".biases", ".signs"):
                produced[base + part] = t
        else:
            produced[name] = t
    return produced


def _dtype(name: str, dtype: str) -> np.dtype:
    if dtype not in DTYPES:
        raise ValueError(f"{name}: a {dtype} tensor cannot be written")
    return DTYPES[dtype]


def _produce(src: Path, g: Gguf, geo: Geometry, t: TensorInfo, name: str) -> np.ndarray:
    """The pack tensor `name` from gguf tensor t, in the pack's dtype."""
    rows = geo.rows(t.name, t.shape[-1])
    if t.ggml_type == PQ2_0:
        if name.endswith(".signs"):
            return _signs(g, t)
        weight, scales = _ternary(src, g, t)
        if name.endswith(".weight"):
            return weight[rows]
        return scales[rows] if name.endswith(".scales") else -scales[rows]
    a = tensor(src, g, t.name)[rows]
    if name.endswith(".A_log"):
        return _a_log(a)
    if name.endswith(".conv1d.weight"):
        a = a[:, :, None]
    return np.ascontiguousarray(a, np.float32)


def _ternary(src: Path, g: Gguf, t: TensorInfo) -> tuple[np.ndarray, np.ndarray]:
    """The packed codes [rows, cols/16] uint32 and scales [rows, cols/128] fp16 of a PQ2_0
    tensor, in the gguf's row order."""
    cols, rows = t.shape[0], int(np.prod(t.shape[1:]))
    blocks = rows * cols // GROUP
    raw = np.memmap(src, np.uint8, "r", g.data_start + t.offset, (blocks * 34,)).reshape(-1, 34)
    weight = np.empty((blocks, GROUP // 16), np.uint32)
    for start in range(0, blocks, CHUNK):
        part = raw[start : start + CHUNK]
        codes = pq2_codes(part)
        if (codes == 3).any():
            raise ValueError(f"{t.name}: code 3 (+2) is not ternary")
        weight[start : start + CHUNK] = pack_2bit(codes)
    scales = raw[:, :2].copy().view("<f2")
    return weight.reshape(rows, cols // 16), scales.reshape(rows, cols // GROUP)


def pack_2bit(codes: np.ndarray) -> np.ndarray:
    """[n, 128] codes 0..3 as [n, 8] uint32, 16 codes a word, the first in the low bits."""
    c = codes.astype(np.uint32).reshape(len(codes), GROUP // 16, 16)
    return (c << (2 * np.arange(16, dtype=np.uint32))).sum(axis=2, dtype=np.uint32)


def unpack_2bit(words: np.ndarray) -> np.ndarray:
    """The codes of [n, w] uint32 words as [n, 16 w] uint8."""
    codes = np.stack([(words >> (2 * k)) & 3 for k in range(16)], axis=-1)
    return codes.reshape(len(words), -1).astype(np.uint8)


def _signs(g: Gguf, t: TensorInfo) -> np.ndarray:
    rot = rotation(g, t.name)
    if rot is None:
        raise ValueError(f"{t.name}: not under prism.hadamard, no sign vector")
    return np.ascontiguousarray(rot[1], np.float32)


def _a_log(a: np.ndarray) -> np.ndarray:
    """float32 A_log with -exp(A_log) back to ssm_a exactly where a float32 can, as the gguf
    converter's exp is undone by searching a few ulps around log(-a)."""
    out = np.log(-a.astype(np.float64)).astype(np.float32)
    for i in np.flatnonzero(np.float32(-np.exp(out)) != a):
        for step in range(1, 9):
            up = np.nextafter(out[i], np.float32(np.inf), dtype=np.float32)
            down = np.nextafter(out[i], np.float32(-np.inf), dtype=np.float32)
            for _ in range(step - 1):
                up = np.nextafter(up, np.float32(np.inf), dtype=np.float32)
                down = np.nextafter(down, np.float32(-np.inf), dtype=np.float32)
            hit = next((c for c in (up, down) if np.float32(-np.exp(c)) == a[i]), None)
            if hit is not None:
                out[i] = hit
                break
    return out


def _array(path: Path, start: int, info: dict[str, Any], name: str) -> np.ndarray:
    lo, hi = info["data_offsets"]
    dtype = _dtype(name, info["dtype"])
    array = np.memmap(path, dtype, "r", start + lo, ((hi - lo) // dtype.itemsize,))
    return array.reshape(info["shape"])


def _same_bytes(a: Path, a_start: int, ai: dict, b: Path, b_start: int, bi: dict) -> bool:
    (a_lo, a_hi), (b_lo, b_hi) = ai["data_offsets"], bi["data_offsets"]
    if a_hi - a_lo != b_hi - b_lo:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        fa.seek(a_start + a_lo)
        fb.seek(b_start + b_lo)
        left = a_hi - a_lo
        while left:
            n = min(CHUNK * 34, left)
            if fa.read(n) != fb.read(n):
                return False
            left -= n
    return True


def _check_ternary(
    out: Path,
    got: dict,
    got_start: int,
    src: Path,
    g: Gguf,
    geo: Geometry,
    t: TensorInfo,
    name: str,
) -> None:
    """scales * code + biases of the pack against (code - 1) * d of the gguf, some rows at a
    time."""
    base = name.removesuffix(".weight")
    words = _array(out / WEIGHTS, got_start, got[name], name)
    scales = _array(out / WEIGHTS, got_start, got[base + ".scales"], name)
    biases = _array(out / WEIGHTS, got_start, got[base + ".biases"], name)
    cols, rows = t.shape[0], int(np.prod(t.shape[1:]))
    raw = np.memmap(src, np.uint8, "r", g.data_start + t.offset, (rows * cols // GROUP * 34,))
    raw = raw.reshape(rows, cols // GROUP, 34)
    order = geo.rows(t.name, rows)
    for lo in range(0, rows, ROWS):
        hi = min(lo + ROWS, rows)
        codes = unpack_2bit(words[lo:hi].reshape(-1, GROUP // 16)).reshape(hi - lo, -1)
        d = raw[order[lo:hi]]
        source = (pq2_codes(d.reshape(-1, 34)).astype(np.float32) - 1.0).reshape(hi - lo, -1)
        scale = d[:, :, :2].copy().view("<f2").reshape(hi - lo, -1).astype(np.float32)
        source *= np.repeat(scale, GROUP, axis=1)
        s = np.repeat(scales[lo:hi].astype(np.float32), GROUP, axis=1)
        b = np.repeat(biases[lo:hi].astype(np.float32), GROUP, axis=1)
        if not np.array_equal(codes.astype(np.float32) * s + b, source):
            raise ValueError(f"{name}: a block does not dequantize to the gguf's weights")
