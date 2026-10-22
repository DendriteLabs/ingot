"""Pin a gguf quant's genesis header in its family package (genesis_<quant>.json): fetches only the
header with HTTP range requests (with INGOT_HF_TOKEN / HF_TOKEN for a private genesis).

    uv run scripts/pin_gguf.py qwen38_27b q2
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_url

from common.config import REPO_ROOT, chain_spec, settings
from common.gguf import GgufError, read_header
from model_validation.gguf_checks import pin_file, pinned

MAX_HEADER = 64 * 1024 * 1024


def fetch_header(url: str, size: int, path: Path, token: str | None = None) -> None:
    """A growing prefix until the header parses, written into a sparse file of the real size."""
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    n = 8 * 1024 * 1024
    while True:
        req = urllib.request.Request(url, headers={**auth, "Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            head = resp.read()
        with open(path, "wb") as f:
            f.write(head)
            f.truncate(size)
        try:
            read_header(path, max_header=len(head))
            return
        except GgufError:
            if n >= MAX_HEADER:
                raise
            n *= 2


def main(family: str, quant: str) -> None:
    pair = chain_spec().pair(family, quant)
    if pair.format != "gguf" or pair.genesis_file is None:
        raise SystemExit(f"{family}.{quant} is not a gguf quant")
    url = hf_hub_url(pair.genesis_repo, pair.genesis_file, revision=pair.genesis_revision)
    token = settings().hf_token or None
    size = get_hf_file_metadata(url, token=token).size
    assert size is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / pair.genesis_file
        fetch_header(url, size, path, token)
        pin = pinned(read_header(path))
    pin["genesis"] = f"{pair.genesis_repo}@{pair.genesis_revision}/{pair.genesis_file}"
    out = REPO_ROOT / "src" / "model_validation" / "families" / family / pin_file(quant)
    out.write_text(json.dumps(pin, separators=(",", ":"), ensure_ascii=False))
    print(f"pinned {len(pin['kv'])} keys and {len(pin['tensors'])} tensors in {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
