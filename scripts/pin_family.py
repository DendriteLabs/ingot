"""Regenerate a family's pinned facts from the hub: the genesis config.json and tensor layout
named in chain.toml for a safetensors quant (gguf quants: scripts/pin_gguf.py).

    uv run scripts/pin_family.py qwen38_27b fp8
"""

from __future__ import annotations

import json
import sys

from common import models
from common.config import REPO_ROOT, chain_spec


def main(name: str, quant: str) -> None:
    pair = chain_spec().pair(name, quant)
    root = REPO_ROOT / "src" / "model_validation" / "families" / name
    config = models.hub_file(pair.genesis_repo, "config.json", pair.genesis_revision)
    (root / "genesis_config.json").write_bytes(config.read_bytes())
    layout = models.hub_layout(pair.genesis_repo, pair.genesis_revision)
    (root / "genesis_layout.json").write_text(
        json.dumps(
            {
                "genesis": pair.genesis_repo,
                "tensors": {k: [d, list(s)] for k, (d, s) in sorted(layout.items())},
            },
            separators=(",", ":"),
        )
    )
    print(f"pinned genesis config and {len(layout)} tensors for {name} in {root}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fp8")
