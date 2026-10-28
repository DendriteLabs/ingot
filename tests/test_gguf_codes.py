"""A king's ternary codes against genesis's: only the codes count, never the scales or norms."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.gguf.codes import code_changes
from tests.test_gguf_ptq1 import q2_file


def test_only_changed_codes_count(tmp_path: Path) -> None:
    codes = np.random.default_rng(3).integers(0, 3, (6, 128)).astype(np.uint8)
    genesis = q2_file(tmp_path / "genesis.gguf", codes)
    # other scales and F32 values, the same codes
    assert code_changes(q2_file(tmp_path / "rescaled.gguf", codes, seed=2), genesis) == (0, 768)
    changed = codes.copy()
    changed[0, :3] = (changed[0, :3] + 1) % 3
    assert code_changes(q2_file(tmp_path / "changed.gguf", changed, seed=2), genesis) == (3, 768)
