"""Regenerate a family's pinned facts from the hub: the genesis config.json and tensor layout
named in chain.toml.

    uv run scripts/pin_family.py qwen38_27b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from common import models
from common.config import REPO_ROOT, chain_spec


def main(name: str) -> None:
    spec = chain_spec()
    root = REPO_ROOT / "src" / "model_validation" / "families" / name
    config = models.hub_file(spec.genesis_repo, "config.json", spec.genesis_revision)
    (root / "genesis_config.json").write_bytes(config.read_bytes())
    layout = models.hub_layout(spec.genesis_repo, spec.genesis_revision)
    (root / "genesis_layout.json").write_text(
        json.dumps(
            {
                "genesis": spec.genesis_repo,
                "tensors": {k: [d, list(s)] for k, (d, s) in sorted(layout.items())},
            },
            separators=(",", ":"),
        )
    )
    print(f"pinned genesis config and {len(layout)} tensors for {name} in {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).stem)
