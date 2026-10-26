"""Exact float32 values of a gguf tensor, and the fork's Hadamard rotation (prism.hadamard).

A folded weight W' = W · R stores the rotation of its input side; the fork turns each activation
x into R x before the matmul: x' = H (s ⊙ x) per block, H the normalized Sylvester-Walsh-Hadamard
matrix of block_size, s the explicit sign vector of the weight's input width
(llama-prism src/llama-model.cpp, llm_graph_context::build_lora_mm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.gguf.reader import Gguf
from common.gguf.types import BF16, F16, F32, PQ2_0, Q8_0, TYPES

ROTATION_PREFIX = "prism.hadamard."


def tensor(path: Path, g: Gguf, name: str) -> np.ndarray:
    """float32 [rows, row_len] for a 2-D tensor, [n] for a 1-D one (GGUF dim 0 is the row)."""
    t = g.tensors[name]
    kind = TYPES[t.ggml_type]
    n = int(np.prod(t.shape))
    size = n // kind.block_elems * kind.block_bytes
    raw = np.memmap(path, np.uint8, "r", g.data_start + t.offset, (size,))
    if t.ggml_type == F32:
        out = raw.view("<f4").astype(np.float32)
    elif t.ggml_type == F16:
        out = raw.view("<f2").astype(np.float32)
    elif t.ggml_type == BF16:
        out = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    elif t.ggml_type == Q8_0:
        blocks = raw.reshape(-1, 34)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        out = (blocks[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)
    elif t.ggml_type == PQ2_0:
        # value (code - 1) * d (ggml-quants.c dequantize_row_pq2_0)
        blocks = raw.reshape(-1, 34)
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)
        out = ((pq2_codes(blocks).astype(np.float32) - 1.0) * d).reshape(-1)
    else:
        raise ValueError(f"{name}: no dequantizer for {kind.name}")
    return out.reshape(tuple(reversed(t.shape))) if len(t.shape) > 1 else out


def pq2_codes(blocks: np.ndarray) -> np.ndarray:
    """[n, 128] 2-bit codes of [n, 34] block_pq2_0 rows (fp16 d, then 32 bytes of codes): weight
    j at bits 2*(j%4) of byte j/4."""
    qs = blocks[:, 2:]
    return np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=-1).reshape(-1, 128)


def rotation(g: Gguf, name: str) -> tuple[int, np.ndarray] | None:
    """(block_size, signs of the tensor's row width) when `name` is stored folded (a weight under
    weight_names) or latent (an embedding under inverse_weight_names), else None."""
    kv = g.kv
    listed = [
        kv[ROTATION_PREFIX + key].value
        for key in ("weight_names", "inverse_weight_names")
        if ROTATION_PREFIX + key in kv
    ]
    if not any(name in names for names in listed):
        return None
    block = int(kv[ROTATION_PREFIX + "block_size"].value)
    width = g.tensors[name].shape[0]
    if kv[ROTATION_PREFIX + "sign_mode"].value != "explicit":
        return block, np.ones(width, np.float32)
    widths = kv[ROTATION_PREFIX + "sign_widths"].value
    values = kv[ROTATION_PREFIX + "sign_values"].value
    at = 0
    for w in widths:
        if w == width:
            return block, np.asarray(values[at : at + w], np.float32)
        at += w
    raise ValueError(f"{name}: no prism.hadamard sign vector of width {width}")


def hadamard(block: int) -> np.ndarray:
    """The normalized Sylvester-Walsh-Hadamard matrix: ±1/√block by the parity of row & col."""
    i = np.arange(block, dtype=np.uint32)
    bits = np.bitwise_and.outer(i, i)
    parity = np.zeros_like(bits)
    while bits.any():
        parity ^= bits & 1
        bits >>= 1
    return np.where(parity == 1, -1.0, 1.0).astype(np.float32) / np.sqrt(np.float32(block))
