"""The ggml tensor types we accept, id -> (name, block_elems, block_bytes). PQ2_0 is the
PrismML fork's group-128 2-bit type: one fp16 scale and 32 bytes of 2-bit codes per 128
weights (ggml/src/ggml-common.h block_pq2_0, branch `prism`)."""

from __future__ import annotations

from typing import NamedTuple


class GgmlType(NamedTuple):
    name: str
    block_elems: int
    block_bytes: int


F32 = 0
F16 = 1
Q8_0 = 8
BF16 = 30
PQ2_0 = 142

TYPES: dict[int, GgmlType] = {
    F32: GgmlType("F32", 1, 4),
    F16: GgmlType("F16", 1, 2),
    Q8_0: GgmlType("Q8_0", 32, 34),
    BF16: GgmlType("BF16", 1, 2),
    PQ2_0: GgmlType("PQ2_0", 128, 34),
}

# Unquantized types: a tensor genesis keeps in one of these keeps its type.
WIDE = frozenset({F32, F16, BF16})

# Block types: byte offset of the block's fp16 scale d (block_q8_0 and block_pq2_0 both start
# with it), and the largest |code| a weight is d times: Q8_0's int8 reaches -128, PQ2_0's ternary
# codes ±1 (its code 3, +2, is rejected by validation). A new block type adds both here, and its
# dequantizer in dequant.py.
SCALE_AT = {Q8_0: 0, PQ2_0: 0}
MAX_CODE = {Q8_0: 128, PQ2_0: 1}
