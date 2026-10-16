"""The safetensors files together may not be larger than genesis's. How the bytes are spent,
which quantization, which layers stay wide, is the miner's choice."""

from __future__ import annotations

from pathlib import Path

from model_validation.families import Family


def weight_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))


def check_weight_bytes(model_dir: Path, family: Family) -> str | None:
    size = weight_bytes(model_dir)
    if size > family.genesis_weight_bytes:
        return f"safetensors total {size} bytes, genesis has {family.genesis_weight_bytes}"
    return None
