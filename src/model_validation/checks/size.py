"""The text model may not hold more bytes than genesis's: the parts the eval never reads are
shipped as genesis and left out of the count on both sides. How the bytes are spent, which
quantization, which layers stay wide, is the miner's."""

from __future__ import annotations

from model_validation.families import Family, Layout, weight_bytes


def check_weight_bytes(layout: Layout, family: Family) -> str | None:
    size = weight_bytes(layout, family.uncounted_prefixes)
    if size > family.genesis_weight_bytes:
        return f"counted weights total {size} bytes, genesis has {family.genesis_weight_bytes}"
    return None
