"""A model family: the genesis config and tensor layout every submission must reproduce, the
keys it may change, and the genesis safetensors size it may not exceed. Facts only; the rules
that use them live in checks/."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Layout = dict[str, tuple[str, tuple[int, ...]]]  # tensor name -> (dtype, shape)


@dataclass(frozen=True)
class Family:
    name: str
    genesis_config: dict[str, Any]
    arch_keys: tuple[str, ...]
    genesis_layout: Layout
    frozen_dtype_prefixes: tuple[str, ...]
    genesis_weight_bytes: int

    def arch_value(self, key: str) -> Any:
        return lookup(self.genesis_config, key)

    def dtype_frozen(self, tensor: str) -> bool:
        return tensor.startswith(self.frozen_dtype_prefixes)


def lookup(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node
