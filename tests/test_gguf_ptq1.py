"""The PQ2_0 -> PTQ1_0 repack: llama-prism's packing exactly, a q2 file repacked without loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.gguf.dequant import pq2_codes
from common.gguf.ptq1 import PTQ1_0, PTQ1_0_BYTES, PTQ1_0_FILE_TYPE, pack, repack, unpack
from common.gguf.reader import STRING, U32, read_header
from common.gguf.types import F32, PQ2_0
from tests.test_gguf import _kv, write_gguf


def c_reference(codes: list[int]) -> bytes:
    """llama-prism's quantize_row_ptq1_0_ref for one block of codes 0..2, line by line."""
    qs, qh, pos, j = [0] * 24, [0] * 2, 0, 0
    for width in (32, 16, 8):
        while j + width <= 24:
            for m in range(width):
                q = 0
                for n in range(5):
                    q = q * 3 + codes[pos + m + n * width]
                qs[j + m] = (q * 256 + 242) // 243
            pos, j = pos + 5 * width, j + width
    for h in range(2):
        q = 0
        for m in range(4):
            q = q * 3 + codes[pos + h + m * 2]
        qh[h] = (q * 3 * 256 + 242) // 243
    return bytes(qs + qh)


def pq2_blocks(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """block_pq2_0 rows as quantize_row_pq2_0_ref lays the bits out."""
    qs = np.zeros((len(codes), 32), np.uint8)
    for j in range(128):
        qs[:, j // 4] |= codes[:, j] << (2 * (j % 4))
    return np.concatenate([scale, qs], 1)


def test_packing_is_the_c_reference_and_unpacks_to_its_source() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, (300, 128)).astype(np.uint8)
    codes[0], codes[1] = 0, 2  # all -1, all +1
    scale = rng.integers(0, 256, (300, 2)).astype(np.uint8)
    packed = pack(codes, scale)
    assert [row[:26].tobytes() for row in packed] == [c_reference(c.tolist()) for c in codes]
    back, back_scale = unpack(packed)
    assert (back == codes).all() and (back_scale == scale).all()
    assert (pq2_codes(pq2_blocks(codes, scale)) == codes).all()


def q2_file(path: Path, codes: np.ndarray) -> Path:
    """A q2-like file: a F32 tensor, then PQ2_0 blocks of `codes`, MOSTLY_PQ2_0."""
    rng = np.random.default_rng(1)
    f32 = rng.standard_normal(8).astype("<f4").tobytes()
    blocks = pq2_blocks(codes, rng.integers(0, 60, (len(codes), 2)).astype(np.uint8)).tobytes()
    data = f32 + bytes(32 - len(f32)) + blocks + bytes(-len(blocks) % 32)
    kv = [
        _kv("general.alignment", U32, 32),
        _kv("general.name", STRING, "tiny"),
        _kv("general.file_type", U32, 141),
    ]
    tensors = [("a.weight", F32, (4, 2), 0), ("b.weight", PQ2_0, (128, len(codes)), 32)]
    return write_gguf(path, kv=kv, tensors=tensors, data=data)


def test_a_q2_file_repacks_without_loss(tmp_path: Path) -> None:
    codes = np.random.default_rng(2).integers(0, 3, (6, 128)).astype(np.uint8)
    src = q2_file(tmp_path / "q2.gguf", codes)
    assert repack(src, tmp_path / "q1.gguf") == 6 * 128
    raw, out = src.read_bytes(), (tmp_path / "q1.gguf").read_bytes()
    start = read_header(src).data_start

    def u32_after(marker: bytes, skip: int) -> int:
        return int.from_bytes(out[out.index(marker) + len(marker) + skip :][:4], "little")

    assert u32_after(b"general.file_type", 4) == PTQ1_0_FILE_TYPE
    assert u32_after(b"b.weight", 4 + 16) == PTQ1_0  # past n_dims and two dims
    assert out[start : start + 32] == raw[start : start + 32], "the F32 tensor is copied"
    blocks = np.frombuffer(out, np.uint8, 6 * PTQ1_0_BYTES, start + 32).reshape(6, -1)
    back, scale = unpack(blocks)
    source = np.frombuffer(raw, np.uint8, 6 * 34, start + 32).reshape(6, 34)
    assert (back == codes).all() and (scale == source[:, :2]).all()
    assert len(out) == start + 32 + 6 * PTQ1_0_BYTES + (-6 * PTQ1_0_BYTES % 32)


def test_a_fourth_level_is_refused(tmp_path: Path) -> None:
    codes = np.ones((2, 128), np.uint8)
    codes[1, 5] = 3
    with pytest.raises(ValueError, match="not ternary"):
        repack(q2_file(tmp_path / "q2.gguf", codes), tmp_path / "q1.gguf")
