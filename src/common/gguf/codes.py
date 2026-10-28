"""How many ternary codes one PQ2_0 gguf changed against another of the same layout: a q2 king
against genesis. Only the codes count; a block's fp16 scale and every non-PQ2_0 tensor (norms)
are the rest of the weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.gguf.dequant import pq2_codes
from common.gguf.reader import read_header
from common.gguf.types import PQ2_0

BLOCK_BYTES = 34  # block_pq2_0: fp16 d, then 32 bytes of 2-bit codes
CHUNK = 1 << 19  # blocks per numpy pass


def code_changes(path: Path, base: Path) -> tuple[int, int]:
    """(weights whose code differs, PQ2_0 weights compared) between path and base. A PQ2_0
    tensor of path that base lacks, or holds with another type or shape, raises ValueError."""
    g, h = read_header(path), read_header(base)
    changed = total = 0
    for name, t in g.tensors.items():
        if t.ggml_type != PQ2_0:
            continue
        u = h.tensors.get(name)
        if u is None or u.ggml_type != PQ2_0 or u.shape != t.shape:
            raise ValueError(f"{name}: {base.name} does not hold the same PQ2_0 tensor")
        n = int(np.prod(t.shape)) // 128
        a = np.memmap(path, np.uint8, "r", g.data_start + t.offset, (n, BLOCK_BYTES))
        b = np.memmap(base, np.uint8, "r", h.data_start + u.offset, (n, BLOCK_BYTES))
        for lo in range(0, n, CHUNK):
            x, y = np.asarray(a[lo : lo + CHUNK]), np.asarray(b[lo : lo + CHUNK])
            rows = (x[:, 2:] != y[:, 2:]).any(1)
            if rows.any():
                changed += int((pq2_codes(x[rows]) != pq2_codes(y[rows])).sum())
        total += n * 128
    return changed, total
