"""The hash dedup mode (a quant's `dedup = "hash"`): a submission is a copy only when its weight
files are byte for byte another banked model's. No GPU, no teacher, no secret: one sha256 pass
over the files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from model_dedup.verdict import Verdict

VERSION = "files-1"
WEIGHTS = (".safetensors", ".gguf")


def file_fingerprint(model_dir: Path) -> dict:
    """sha256 over the sorted digests of the weight files, so shard names do not matter."""
    digests = []
    for path in sorted(p for p in Path(model_dir).iterdir() if p.suffix in WEIGHTS):
        with path.open("rb") as fh:
            digests.append(hashlib.file_digest(fh, "sha256").hexdigest())
    if not digests:
        raise ValueError(f"{model_dir}: no weight files")
    combined = hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()
    return {"version": VERSION, "key_id": "", "tensors_hash": combined}


def identical(fp: dict, shortlist: dict[str, dict]) -> Verdict:
    for bank_id, other in shortlist.items():
        if other["tensors_hash"] == fp["tensors_hash"]:
            return Verdict("COPY", bank_id, f"identical files to {bank_id}", 1.0, 1.0, ())
    return Verdict(None, None, "no identical files", 0.0, 0.0, ())
