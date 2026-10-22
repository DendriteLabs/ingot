"""Container checks on a parsed header: tensors packed back to back in offset order with only
alignment padding, each the exact byte size its type implies, the padding zero, and nothing
after the data."""

from __future__ import annotations

import math
from pathlib import Path

from common.gguf.reader import Gguf
from common.gguf.types import TYPES


def tensor_bytes(shape: tuple[int, ...], ggml_type: int) -> int | None:
    """Byte size of a tensor, or None when its rows do not split into whole blocks."""
    t = TYPES[ggml_type]
    if shape[0] % t.block_elems:
        return None
    return math.prod(shape) // t.block_elems * t.block_bytes


def check_container(g: Gguf, path: str | Path) -> str | None:
    a = g.alignment
    if a <= 0 or a & (a - 1):
        return f"alignment {a} is not a power of two"
    spans: list[tuple[int, int]] = [(g.header_end, g.data_start)]
    pos = 0
    for t in sorted(g.tensors.values(), key=lambda t: t.offset):
        if t.offset % a:
            return f"{t.name}: offset {t.offset} is not a multiple of the alignment {a}"
        size = tensor_bytes(t.shape, t.ggml_type)
        if size is None:
            name, block = TYPES[t.ggml_type].name, TYPES[t.ggml_type].block_elems
            return f"{t.name}: rows of {t.shape[0]} do not split into {name} blocks of {block}"
        expected = -(-pos // a) * a
        if t.offset < expected:
            return f"{t.name}: offset {t.offset} overlaps the tensor before it"
        if t.offset > expected:
            return f"{t.name}: offset {t.offset} leaves a gap after the tensor before it"
        spans.append((g.data_start + pos, g.data_start + t.offset))
        pos = t.offset + size
    end = g.data_start + pos
    padded = g.data_start + -(-pos // a) * a
    if g.file_size not in (end, padded):
        return f"the data ends at byte {end} but the file has {g.file_size} bytes"
    spans.append((end, g.file_size))
    with open(path, "rb") as f:
        for lo, hi in spans:
            if hi > lo:
                f.seek(lo)
                if f.read(hi - lo).strip(b"\0"):
                    return f"padding at bytes {lo}..{hi} is not zero"
    return None
