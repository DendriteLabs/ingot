"""Tensor layout read from safetensors headers; no weights are loaded."""

from __future__ import annotations

from pathlib import Path

from common.models import safetensors_header
from model_validation.families import Layout


def layout_from_dir(model_dir: Path) -> Layout:
    out: Layout = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        for name, info in safetensors_header(path).items():
            out[name] = (info["dtype"], tuple(info["shape"]))
    return out
