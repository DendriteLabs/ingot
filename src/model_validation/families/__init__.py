"""One package per family; `load(name)` returns its pinned facts."""

from __future__ import annotations

import importlib
import json
from importlib.resources import files
from typing import Any

from model_validation.families.base import Family, Layout, lookup, weight_bytes

__all__ = ["Family", "Layout", "load", "lookup", "weight_bytes"]


def load(name: str) -> Family:
    package = importlib.import_module(f"model_validation.families.{name}")
    root = files(package)
    genesis = json.loads(root.joinpath("genesis_config.json").read_text())
    free = package.FREE_KEYS
    pinned = json.loads(root.joinpath("genesis_layout.json").read_text())["tensors"]
    layout = {k: (d, tuple(s)) for k, (d, s) in pinned.items()}
    return Family(
        name=name,
        genesis_config=genesis,
        arch_keys=tuple(
            k for k in _leaf_keys(genesis) if not any(k == f or k.startswith(f + ".") for f in free)
        ),
        genesis_layout=layout,
        frozen_dtype_prefixes=package.FROZEN_DTYPE_PREFIXES,
        genesis_weight_bytes=weight_bytes(layout, package.UNCOUNTED_PREFIXES),
        fused=package.FUSED,
        embeddings=package.EMBEDDINGS,
        uncounted_prefixes=package.UNCOUNTED_PREFIXES,
    )


def _leaf_keys(config: dict[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in config.items():
        if isinstance(value, dict) and value:
            keys += _leaf_keys(value, f"{prefix}{key}.")
        else:
            keys.append(prefix + key)
    return keys
