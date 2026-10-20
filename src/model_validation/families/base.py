"""A model family: the genesis config and tensor layout every submission must reproduce, the
keys it may change, and the genesis tensor bytes it may not exceed. Facts only; the rules
that use them live in checks/."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Layout = dict[str, tuple[str, tuple[int, ...]]]  # tensor name -> (dtype, shape)
_BYTES = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}


@dataclass(frozen=True)
class Family:
    name: str
    genesis_config: dict[str, Any]
    arch_keys: tuple[str, ...]
    genesis_layout: Layout
    frozen_dtype_prefixes: tuple[str, ...]
    genesis_weight_bytes: int
    # projections vLLM loads as one layer (its packed_modules_mapping): one precision for all
    fused: tuple[tuple[str, ...], ...] = ()
    # modules vLLM builds as embedding layers (lm_head too): never read off modules_to_not_convert
    embeddings: tuple[str, ...] = ()
    # parts the eval never reads, shipped as genesis: left out of the size count on both sides
    uncounted_prefixes: tuple[str, ...] = ()

    def arch_value(self, key: str) -> Any:
        return lookup(self.genesis_config, key)

    def dtype_frozen(self, tensor: str) -> bool:
        return tensor.startswith(self.frozen_dtype_prefixes)


def weight_bytes(layout: Layout, uncounted: tuple[str, ...] = ()) -> int:
    return sum(
        _BYTES[d] * math.prod(s) for n, (d, s) in layout.items() if not n.startswith(uncounted)
    )


def lookup(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node
