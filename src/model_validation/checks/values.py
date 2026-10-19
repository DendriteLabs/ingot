"""Every weight is a number: a NaN or an infinity anywhere makes a model the servers cannot run
and the eval cannot score, and that is the file's fault. Read straight from the shards' bytes,
in slices, so the check needs neither torch nor the model in memory."""

from __future__ import annotations

import mmap
from collections.abc import Callable
from pathlib import Path

import numpy as np

from common.models import safetensors_header

SLICE = 1 << 26  # elements per slice: a few hundred MB of temporaries at most
# dtype -> (the bytes as numpy sees them, which elements are not finite)
_NON_FINITE: dict[str, tuple[type, Callable[[np.ndarray], np.ndarray]]] = {
    "BF16": (np.uint16, lambda v: (v & 0x7F80) == 0x7F80),  # exponent all ones: inf or NaN
    "F32": (np.float32, lambda v: ~np.isfinite(v)),
    "F8_E4M3": (np.uint8, lambda v: (v & 0x7F) == 0x7F),  # e4m3fn: NaN, there is no inf
}


def check_values(model_dir: Path) -> str | None:
    for shard in sorted(model_dir.glob("*.safetensors")):
        header = safetensors_header(shard)
        with shard.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 8 + int.from_bytes(mm[:8], "little")
            for name, info in header.items():
                kind = _NON_FINITE.get(info["dtype"])
                if kind is None:
                    continue
                view, non_finite = kind
                a, b = info["data_offsets"]
                count = (b - a) // np.dtype(view).itemsize
                bad = 0
                size = np.dtype(view).itemsize
                for at in range(0, count, SLICE):
                    # a copy of the slice: a view on the map would keep it from closing
                    lo = start + a + at * size
                    v = np.frombuffer(mm[lo : lo + min(SLICE, count - at) * size], dtype=view)
                    bad += int(non_finite(v).sum())
                if bad:
                    return f"{name}: {bad} of its values are NaN or infinite"
    return None
