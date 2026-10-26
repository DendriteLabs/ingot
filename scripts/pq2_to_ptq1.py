"""Repack a q2 king, a ternary PQ2_0 gguf, as Prism's PTQ1_0: 1.75 bits per weight instead of
2.125, without loss (common.gguf.ptq1). The promoter publishes this file beside every q2 king.

    uv run python scripts/pq2_to_ptq1.py <in PQ2_0.gguf> <out PTQ1_0.gguf>
"""

from __future__ import annotations

import sys
from pathlib import Path

from common.gguf.ptq1 import repack

if __name__ == "__main__":
    weights = repack(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{sys.argv[2]}: {weights:,} weights repacked, each block checked against the source")
