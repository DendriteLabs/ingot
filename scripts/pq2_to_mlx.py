"""Convert a q2 king, a ternary PQ2_0 gguf, to a Prism-layout MLX 2-bit pack without loss
(common.gguf.mlx), completed from the genesis's MLX twin, and verify it against the gguf.

    uv run python scripts/pq2_to_mlx.py <in PQ2_0.gguf> <template dir> <out dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from common.gguf.mlx import convert, verify

if __name__ == "__main__":
    src, template, out = (Path(a) for a in sys.argv[1:4])
    weights = convert(src, template, out)
    verify(out, src, template)
    print(f"{out}: {weights:,} weights converted and verified against {src}")
