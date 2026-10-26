"""Prism's PTQ1_0, 1.75 bits per weight, from a ternary PQ2_0 gguf (2.125) without loss: each
128-weight block keeps its fp16 scale and its codes, base-3 packed as llama-prism's
quantize_row_ptq1_0_ref packs them; every other byte is copied, and general.file_type becomes
MOSTLY_PTQ1_0. Bonsai 2's PQ2_0 file repacks to Prism's published PTQ1_0 file byte for byte."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from common.gguf.dequant import pq2_codes
from common.gguf.reader import Gguf, TensorInfo, read_header
from common.gguf.types import PQ2_0, TYPES

# block_ptq1_0 (llama-prism ggml-common.h): qs[24], 5 trits a byte; qh[2], 4 a byte; fp16 d.
# Not in TYPES, which lists what a submission may use.
PTQ1_0, PTQ1_0_BYTES = 143, 28
# general.file_type holds a llama_ftype (llama.h), not a ggml type: 141 MOSTLY_PQ2_0 (142 its
# legacy value), 143 MOSTLY_PTQ1_0
PQ2_0_FILE_TYPES, PTQ1_0_FILE_TYPE = (141, 142), 143
CHUNK = 1 << 19  # blocks per numpy pass: about 0.5 GB of working arrays


def is_pq2(g: Gguf) -> bool:
    file_type = g.kv.get("general.file_type")
    return file_type is not None and file_type.value in PQ2_0_FILE_TYPES


def pack(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """[n, 128] ternary codes and [n, 2] fp16 scale bytes as [n, 28] block_ptq1_0 rows."""
    n, c = len(codes), codes.astype(np.uint32)
    out = np.empty((n, PTQ1_0_BYTES), np.uint8)
    # qs: values 0..79 in bytes 0..15, 80..119 in bytes 16..23; byte m holds x[m + k*width] for
    # k = 0..4, the first trit most significant, stored as ceil(q * 256 / 243)
    for lo, hi, width, at in ((0, 80, 16, 0), (80, 120, 8, 16)):
        v = c[:, lo:hi].reshape(n, 5, width)
        q = v[:, 0] * 81 + v[:, 1] * 27 + v[:, 2] * 9 + v[:, 3] * 3 + v[:, 4]
        out[:, at : at + width] = (q * 256 + 242) // 243
    # qh: byte h holds x[120 + h + 2m] for m = 0..3, shifted up one trit
    v = c[:, 120:128].reshape(n, 4, 2)
    q = (v[:, 0] * 27 + v[:, 1] * 9 + v[:, 2] * 3 + v[:, 3]) * 3
    out[:, 24:26] = (q * 256 + 242) // 243
    out[:, 26:28] = scale
    return out


def unpack(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The codes and scale bytes of [n, 28] block_ptq1_0 rows, read as dequantize_row_ptq1_0
    reads them."""
    n, b = len(blocks), blocks.astype(np.uint32)

    def trits(byte: np.ndarray, count: int) -> np.ndarray:
        return np.stack([(((byte * 3**k) & 0xFF) * 3) >> 8 for k in range(count)], 1)

    parts = [
        trits(b[:, 0:16], 5).reshape(n, 80),
        trits(b[:, 16:24], 5).reshape(n, 40),
        trits(b[:, 24:26], 4).reshape(n, 8),
    ]
    return np.concatenate(parts, 1).astype(np.uint8), blocks[:, 26:28]


def repack(src: Path, dst: Path) -> int:
    """Writes the ternary PQ2_0 gguf src as PTQ1_0 at dst, each block unpacked again and compared
    with the source before it is written; returns the weights repacked. A file that is not PQ2_0,
    or a code 3 (+2, not ternary), raises ValueError."""
    g = read_header(src)
    if not is_pq2(g):
        raise ValueError(f"{src}: general.file_type is not MOSTLY_PQ2_0")
    with open(src, "rb") as f:
        head = bytearray(f.read(g.header_end))
    original = _tensor_infos(g, {t.name: (t.ggml_type, t.offset) for t in g.tensors.values()})
    kv_end = g.header_end - len(original)
    if head[kv_end:] != original:
        raise ValueError(f"{src}: its tensor infos do not re-serialize to the file's bytes")
    key = b"general.file_type"
    at = head.index(struct.pack("<Q", len(key)) + key + struct.pack("<I", 4)) + len(key) + 12
    struct.pack_into("<I", head, at, PTQ1_0_FILE_TYPE)

    # every tensor in its place in the file's order, each start aligned as llama.cpp writes them
    placed, end = {}, 0
    for t in sorted(g.tensors.values(), key=lambda t: t.offset):
        placed[t.name] = (PTQ1_0 if t.ggml_type == PQ2_0 else t.ggml_type, end)
        end = -(-(end + _size(t, placed[t.name][0])) // g.alignment) * g.alignment
    head[kv_end:] = _tensor_infos(g, placed)

    weights = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(head + bytes(g.data_start - len(head)))
        for t in g.tensors.values():
            fin.seek(g.data_start + t.offset)
            fout.seek(g.data_start + placed[t.name][1])
            if t.ggml_type != PQ2_0:
                fout.write(fin.read(_size(t, t.ggml_type)))
                continue
            blocks = _size(t, PQ2_0) // 34
            for start in range(0, blocks, CHUNK):
                rows = np.frombuffer(fin.read(min(CHUNK, blocks - start) * 34), np.uint8)
                rows = rows.reshape(-1, 34)
                codes = pq2_codes(rows)
                if (codes == 3).any():
                    raise ValueError(f"{t.name}: code 3 (+2) is not ternary")
                packed = pack(codes, rows[:, :2])
                back, scale = unpack(packed)
                if not ((back == codes).all() and (scale == rows[:, :2]).all()):
                    raise ValueError(f"{t.name}: a block does not unpack to its source")
                fout.write(packed.tobytes())
            weights += blocks * 128
        fout.truncate(g.data_start + end)
    return weights


def _size(t: TensorInfo, ggml_type: int) -> int:
    weights = int(np.prod(t.shape))
    if ggml_type == PTQ1_0:
        return weights // 128 * PTQ1_0_BYTES
    kind = TYPES[ggml_type]
    return weights // kind.block_elems * kind.block_bytes


def _tensor_infos(g: Gguf, placed: dict[str, tuple[int, int]]) -> bytes:
    """The header's tensor infos in the file's order, each with its (ggml type, offset)."""
    out = bytearray()
    for t in g.tensors.values():
        name = t.name.encode()
        out += struct.pack(f"<Q{len(name)}sI", len(name), name, len(t.shape))
        out += struct.pack(f"<{len(t.shape)}QIQ", *t.shape, *placed[t.name])
    return bytes(out)
